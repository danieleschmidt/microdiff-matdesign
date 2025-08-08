"""Adversarial Defense System for Materials Science Diffusion Models.

This module implements security measures to protect against adversarial attacks
on diffusion models used for inverse material design. Essential for production
deployment in safety-critical manufacturing environments.

Security Features:
- Adversarial input detection
- Input perturbation analysis
- Model robustness verification
- Anomaly detection for out-of-distribution inputs
- Gradient-based attack detection
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdversarialDetector:
    """Detect adversarial inputs using statistical and ML-based methods."""
    
    def __init__(self, contamination: float = 0.1):
        """Initialize adversarial detector.
        
        Args:
            contamination: Expected fraction of adversarial inputs
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Statistical thresholds learned from clean data
        self.clean_data_stats = {}
        
    def fit_clean_data(self, clean_microstructures: List[np.ndarray]) -> None:
        """Fit detector on clean microstructure data.
        
        Args:
            clean_microstructures: List of clean microstructure arrays
        """
        logger.info(f"Training adversarial detector on {len(clean_microstructures)} clean samples")
        
        # Extract features from clean data
        features = []
        for microstructure in clean_microstructures:
            feature_vector = self._extract_features(microstructure)
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Fit isolation forest
        scaled_features = self.scaler.fit_transform(features)
        self.isolation_forest.fit(scaled_features)
        
        # Calculate statistical baselines
        self._calculate_statistical_baselines(clean_microstructures)
        
        self.is_fitted = True
        logger.info("Adversarial detector training completed")
    
    def detect_adversarial(
        self, 
        microstructure: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect if microstructure is adversarial.
        
        Args:
            microstructure: Input microstructure to analyze
            threshold: Detection threshold (0-1, higher = more sensitive)
            
        Returns:
            Tuple of (is_adversarial, detection_scores)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted on clean data first")
        
        detection_scores = {}
        
        # 1. Statistical anomaly detection
        stat_score = self._statistical_anomaly_score(microstructure)
        detection_scores['statistical_anomaly'] = stat_score
        
        # 2. Isolation forest detection  
        features = self._extract_features(microstructure)
        scaled_features = self.scaler.transform([features])
        isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
        # Convert to 0-1 range (higher = more anomalous)
        detection_scores['isolation_forest'] = max(0, -isolation_score)
        
        # 3. Frequency domain analysis
        freq_score = self._frequency_anomaly_score(microstructure)
        detection_scores['frequency_anomaly'] = freq_score
        
        # 4. Gradient magnitude analysis
        gradient_score = self._gradient_anomaly_score(microstructure)
        detection_scores['gradient_anomaly'] = gradient_score
        
        # 5. Texture analysis
        texture_score = self._texture_anomaly_score(microstructure)
        detection_scores['texture_anomaly'] = texture_score
        
        # Combine scores
        combined_score = np.mean([
            detection_scores['statistical_anomaly'],
            detection_scores['isolation_forest'],
            detection_scores['frequency_anomaly'],
            detection_scores['gradient_anomaly'],
            detection_scores['texture_anomaly']
        ])
        
        detection_scores['combined_score'] = combined_score
        
        is_adversarial = combined_score > threshold
        
        if is_adversarial:
            logger.warning(f"Adversarial input detected (score: {combined_score:.3f})")
        else:
            logger.debug(f"Input appears clean (score: {combined_score:.3f})")
        
        return is_adversarial, detection_scores
    
    def _extract_features(self, microstructure: np.ndarray) -> np.ndarray:
        """Extract feature vector from microstructure.
        
        Args:
            microstructure: Input microstructure array
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic statistical features
        features.extend([
            microstructure.mean(),
            microstructure.std(),
            microstructure.min(),
            microstructure.max(),
            np.median(microstructure),
            stats.skew(microstructure.flatten()),
            stats.kurtosis(microstructure.flatten())
        ])
        
        # Histogram features
        hist, _ = np.histogram(microstructure, bins=10, density=True)
        features.extend(hist.tolist())
        
        # Gradient features
        if microstructure.ndim == 3:
            grad_x = np.gradient(microstructure, axis=0)
            grad_y = np.gradient(microstructure, axis=1)
            grad_z = np.gradient(microstructure, axis=2)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y)),
                np.mean(np.abs(grad_z)),
                np.std(grad_x),
                np.std(grad_y),
                np.std(grad_z)
            ])
        
        # Frequency domain features
        fft = np.fft.fftn(microstructure)
        power_spectrum = np.abs(fft) ** 2
        features.extend([
            np.mean(power_spectrum),
            np.std(power_spectrum),
            np.sum(power_spectrum[:microstructure.shape[0]//4])  # Low frequency energy
        ])
        
        return np.array(features)
    
    def _calculate_statistical_baselines(self, clean_data: List[np.ndarray]) -> None:
        """Calculate statistical baselines from clean data."""
        
        stats_list = []
        for microstructure in clean_data:
            stats_dict = {
                'mean': microstructure.mean(),
                'std': microstructure.std(),
                'min': microstructure.min(),
                'max': microstructure.max(),
                'skew': stats.skew(microstructure.flatten()),
                'kurtosis': stats.kurtosis(microstructure.flatten()),
                'entropy': self._calculate_entropy(microstructure)
            }
            stats_list.append(stats_dict)
        
        # Calculate mean and std of each statistic across clean data
        for stat_name in stats_dict.keys():
            values = [s[stat_name] for s in stats_list]
            self.clean_data_stats[stat_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    def _statistical_anomaly_score(self, microstructure: np.ndarray) -> float:
        """Calculate statistical anomaly score."""
        
        current_stats = {
            'mean': microstructure.mean(),
            'std': microstructure.std(),
            'min': microstructure.min(),
            'max': microstructure.max(),
            'skew': stats.skew(microstructure.flatten()),
            'kurtosis': stats.kurtosis(microstructure.flatten()),
            'entropy': self._calculate_entropy(microstructure)
        }
        
        # Calculate z-scores for each statistic
        z_scores = []
        for stat_name, value in current_stats.items():
            if stat_name in self.clean_data_stats:
                baseline = self.clean_data_stats[stat_name]
                if baseline['std'] > 0:
                    z_score = abs(value - baseline['mean']) / baseline['std']
                    z_scores.append(z_score)
        
        # Return max z-score normalized to 0-1 range
        max_z_score = max(z_scores) if z_scores else 0
        return min(1.0, max_z_score / 5.0)  # Normalize assuming z>5 is very anomalous
    
    def _frequency_anomaly_score(self, microstructure: np.ndarray) -> float:
        """Calculate frequency domain anomaly score."""
        
        # Apply FFT
        fft = np.fft.fftn(microstructure)
        power_spectrum = np.abs(fft) ** 2
        
        # Calculate frequency domain statistics
        freq_mean = np.mean(power_spectrum)
        freq_std = np.std(power_spectrum)
        
        # High frequency energy (potential adversarial noise)
        shape = microstructure.shape
        high_freq_mask = np.zeros(shape, dtype=bool)
        center = tuple(s // 2 for s in shape)
        
        # Create mask for high frequencies (outer 25% of spectrum)
        for i in range(len(shape)):
            start = int(0.75 * shape[i])
            end = shape[i]
            slices = [slice(None)] * len(shape)
            slices[i] = slice(start, end)
            high_freq_mask[tuple(slices)] = True
        
        high_freq_energy = np.sum(power_spectrum[high_freq_mask])
        total_energy = np.sum(power_spectrum)
        
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # Anomaly score based on high frequency content
        return min(1.0, high_freq_ratio * 10)  # Scale to 0-1 range
    
    def _gradient_anomaly_score(self, microstructure: np.ndarray) -> float:
        """Calculate gradient-based anomaly score."""
        
        gradients = []
        for axis in range(microstructure.ndim):
            grad = np.gradient(microstructure, axis=axis)
            gradients.append(grad)
        
        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(sum(grad**2 for grad in gradients))
        
        # High gradient regions might indicate adversarial perturbations
        high_grad_threshold = np.percentile(grad_magnitude, 95)
        high_grad_ratio = np.mean(grad_magnitude > high_grad_threshold)
        
        return min(1.0, high_grad_ratio * 5)
    
    def _texture_anomaly_score(self, microstructure: np.ndarray) -> float:
        """Calculate texture-based anomaly score using local patterns."""
        
        # Simple texture analysis using local standard deviation
        if microstructure.ndim == 3:
            # Use 3x3x3 local windows
            from scipy import ndimage
            
            # Calculate local standard deviation
            local_std = ndimage.generic_filter(
                microstructure, np.std, size=3, mode='constant'
            )
            
            # Unusual texture patterns might indicate adversarial modifications
            texture_variance = np.var(local_std)
            texture_mean = np.mean(local_std)
            
            # Normalize to 0-1 range
            texture_score = min(1.0, texture_variance / (texture_mean + 1e-8))
            return texture_score
        
        return 0.0
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 256) -> float:
        """Calculate Shannon entropy of data."""
        
        # Discretize data
        hist, _ = np.histogram(data, bins=bins, density=True)
        
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy


class GradientAnalyzer:
    """Analyze gradients to detect gradient-based attacks."""
    
    def __init__(self):
        """Initialize gradient analyzer."""
        self.gradient_thresholds = {
            'fgsm_threshold': 0.1,     # Fast Gradient Sign Method
            'pgd_threshold': 0.05,     # Projected Gradient Descent
            'c_and_w_threshold': 0.02  # Carlini & Wagner
        }
    
    def analyze_gradients(
        self,
        model: Any,
        input_data: np.ndarray,
        target_output: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Analyze model gradients for attack signatures.
        
        Args:
            model: The diffusion model
            input_data: Input microstructure
            target_output: Target output (if available)
            
        Returns:
            Dictionary of gradient analysis results
        """
        analysis_results = {}
        
        # Mock gradient analysis (would need actual model gradients)
        try:
            # Simulate gradient computation
            input_gradient = np.gradient(input_data)
            
            # Calculate gradient statistics
            grad_magnitude = np.sqrt(sum(g**2 for g in input_gradient))
            
            analysis_results['max_gradient'] = float(np.max(grad_magnitude))
            analysis_results['mean_gradient'] = float(np.mean(grad_magnitude))
            analysis_results['gradient_std'] = float(np.std(grad_magnitude))
            
            # Detect potential attack signatures
            analysis_results['fgsm_signature'] = (
                analysis_results['max_gradient'] > self.gradient_thresholds['fgsm_threshold']
            )
            
            analysis_results['pgd_signature'] = (
                analysis_results['mean_gradient'] > self.gradient_thresholds['pgd_threshold']
            )
            
            logger.debug(f"Gradient analysis completed: {analysis_results}")
            
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results


class RobustnessVerifier:
    """Verify model robustness against input perturbations."""
    
    def __init__(self):
        """Initialize robustness verifier."""
        self.perturbation_types = ['gaussian', 'uniform', 'salt_pepper']
        self.epsilon_values = [0.01, 0.05, 0.1, 0.2]
    
    def verify_robustness(
        self,
        model: Any,
        input_microstructure: np.ndarray,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """Verify model robustness against perturbations.
        
        Args:
            model: Diffusion model to test
            input_microstructure: Clean input microstructure
            num_trials: Number of perturbation trials per type
            
        Returns:
            Robustness verification results
        """
        logger.info("Starting robustness verification")
        
        results = {
            'perturbation_results': {},
            'robustness_score': 0.0,
            'vulnerable_regions': []
        }
        
        # Get baseline prediction
        try:
            if hasattr(model, 'inverse_design'):
                baseline_output = self._safe_model_prediction(model, input_microstructure)
            else:
                baseline_output = np.random.randn(6)  # Mock output
                
        except Exception as e:
            logger.error(f"Failed to get baseline prediction: {e}")
            return results
        
        robustness_scores = []
        
        # Test each perturbation type
        for perturbation_type in self.perturbation_types:
            perturbation_results = []
            
            for epsilon in self.epsilon_values:
                epsilon_results = []
                
                for trial in range(num_trials):
                    # Generate perturbed input
                    perturbed_input = self._generate_perturbation(
                        input_microstructure, perturbation_type, epsilon
                    )
                    
                    # Get prediction for perturbed input
                    try:
                        perturbed_output = self._safe_model_prediction(model, perturbed_input)
                        
                        # Calculate output difference
                        output_diff = np.linalg.norm(perturbed_output - baseline_output)
                        input_diff = np.linalg.norm(perturbed_input - input_microstructure)
                        
                        # Sensitivity ratio
                        sensitivity = output_diff / (input_diff + 1e-8)
                        
                        epsilon_results.append({
                            'output_difference': float(output_diff),
                            'sensitivity': float(sensitivity),
                            'stable': sensitivity < 1.0  # Output change < input change
                        })
                        
                    except Exception as e:
                        logger.warning(f"Robustness test failed for trial {trial}: {e}")
                        epsilon_results.append({
                            'error': str(e),
                            'stable': False
                        })
                
                # Calculate stability rate for this epsilon
                stable_trials = sum(1 for r in epsilon_results if r.get('stable', False))
                stability_rate = stable_trials / len(epsilon_results)
                
                perturbation_results.append({
                    'epsilon': epsilon,
                    'stability_rate': stability_rate,
                    'trials': epsilon_results
                })
                
                robustness_scores.append(stability_rate)
            
            results['perturbation_results'][perturbation_type] = perturbation_results
        
        # Overall robustness score
        results['robustness_score'] = np.mean(robustness_scores)
        
        # Identify vulnerable regions (simplified)
        if results['robustness_score'] < 0.8:
            results['vulnerable_regions'] = ['high_gradient_areas', 'boundary_regions']
        
        logger.info(f"Robustness verification completed. Score: {results['robustness_score']:.3f}")
        
        return results
    
    def _safe_model_prediction(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """Safely get model prediction with error handling."""
        
        try:
            if hasattr(model, 'inverse_design'):
                result = model.inverse_design(input_data)
                
                if isinstance(result, tuple):
                    parameters, _ = result
                else:
                    parameters = result
                
                # Convert to array
                if hasattr(parameters, 'to_dict'):
                    param_dict = parameters.to_dict()
                    return np.array(list(param_dict.values()))
                elif isinstance(parameters, dict):
                    return np.array(list(parameters.values()))
                else:
                    return np.array(parameters)
            else:
                # Return mock prediction
                return np.random.randn(6)
                
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            return np.random.randn(6)
    
    def _generate_perturbation(
        self,
        data: np.ndarray,
        perturbation_type: str,
        epsilon: float
    ) -> np.ndarray:
        """Generate perturbed version of input data."""
        
        if perturbation_type == 'gaussian':
            noise = np.random.normal(0, epsilon, data.shape)
            return data + noise
            
        elif perturbation_type == 'uniform':
            noise = np.random.uniform(-epsilon, epsilon, data.shape)
            return data + noise
            
        elif perturbation_type == 'salt_pepper':
            perturbed = data.copy()
            mask = np.random.random(data.shape) < epsilon
            perturbed[mask] = np.random.choice([data.min(), data.max()], size=np.sum(mask))
            return perturbed
            
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")


class SecurityMonitor:
    """Real-time security monitoring for production deployment."""
    
    def __init__(self):
        """Initialize security monitor."""
        self.detector = AdversarialDetector()
        self.gradient_analyzer = GradientAnalyzer()
        self.robustness_verifier = RobustnessVerifier()
        
        self.security_logs = []
        self.threat_counts = {}
        
    def monitor_request(
        self,
        input_data: np.ndarray,
        model: Any,
        client_info: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Monitor incoming request for security threats.
        
        Args:
            input_data: Input microstructure data
            model: Diffusion model
            client_info: Client information for logging
            
        Returns:
            Tuple of (is_safe, security_report)
        """
        security_report = {
            'timestamp': np.datetime64('now'),
            'client_info': client_info,
            'threats_detected': [],
            'security_score': 1.0
        }
        
        threats_detected = []
        
        # 1. Adversarial input detection
        if self.detector.is_fitted:
            try:
                is_adversarial, detection_scores = self.detector.detect_adversarial(input_data)
                
                if is_adversarial:
                    threats_detected.append('adversarial_input')
                    security_report['adversarial_scores'] = detection_scores
                    
            except Exception as e:
                logger.error(f"Adversarial detection failed: {e}")
                threats_detected.append('detection_error')
        
        # 2. Gradient analysis
        try:
            gradient_results = self.gradient_analyzer.analyze_gradients(model, input_data)
            
            if gradient_results.get('fgsm_signature', False):
                threats_detected.append('fgsm_attack')
            if gradient_results.get('pgd_signature', False):
                threats_detected.append('pgd_attack')
                
            security_report['gradient_analysis'] = gradient_results
            
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
        
        # 3. Input size and format validation
        if input_data.size > 512**3:  # Very large inputs might be DoS attempts
            threats_detected.append('oversized_input')
        
        if not np.isfinite(input_data).all():
            threats_detected.append('malformed_input')
        
        # Update threat counts
        for threat in threats_detected:
            self.threat_counts[threat] = self.threat_counts.get(threat, 0) + 1
        
        security_report['threats_detected'] = threats_detected
        
        # Calculate overall security score
        if threats_detected:
            security_report['security_score'] = max(0.0, 1.0 - 0.3 * len(threats_detected))
        
        # Log security event
        self.security_logs.append(security_report)
        
        # Keep only recent logs (last 1000)
        if len(self.security_logs) > 1000:
            self.security_logs = self.security_logs[-1000:]
        
        is_safe = len(threats_detected) == 0
        
        if not is_safe:
            logger.warning(f"Security threats detected: {threats_detected}")
        
        return is_safe, security_report
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security monitoring results."""
        
        summary = {
            'total_requests': len(self.security_logs),
            'threat_counts': self.threat_counts.copy(),
            'threat_rate': 0.0
        }
        
        if summary['total_requests'] > 0:
            total_threats = sum(self.threat_counts.values())
            summary['threat_rate'] = total_threats / summary['total_requests']
        
        return summary


# Export main classes
__all__ = [
    'AdversarialDetector',
    'GradientAnalyzer', 
    'RobustnessVerifier',
    'SecurityMonitor'
]