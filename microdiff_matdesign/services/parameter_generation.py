"""Service for intelligent parameter generation and recommendation."""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import warnings
from dataclasses import dataclass
from enum import Enum

from ..core import MicrostructureDiffusion, ProcessParameters
from ..utils.validation import validate_parameters, validate_alloy_compatibility


class OptimizationObjective(Enum):
    """Optimization objectives for parameter generation."""
    DENSITY = "density"
    STRENGTH = "strength"
    SURFACE_QUALITY = "surface_quality"
    BUILD_SPEED = "build_speed"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COST = "cost"


@dataclass
class ParameterConstraints:
    """Container for parameter constraints."""
    laser_power_range: Tuple[float, float] = (50.0, 500.0)
    scan_speed_range: Tuple[float, float] = (200.0, 2000.0)
    layer_thickness_range: Tuple[float, float] = (10.0, 100.0)
    hatch_spacing_range: Tuple[float, float] = (50.0, 300.0)
    powder_bed_temp_range: Tuple[float, float] = (20.0, 200.0)
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """Convert to dictionary format."""
        return {
            'laser_power': self.laser_power_range,
            'scan_speed': self.scan_speed_range,
            'layer_thickness': self.layer_thickness_range,
            'hatch_spacing': self.hatch_spacing_range,
            'powder_bed_temp': self.powder_bed_temp_range
        }


@dataclass
class GenerationRequest:
    """Request for parameter generation."""
    target_microstructure: Optional[np.ndarray] = None
    target_properties: Optional[Dict[str, float]] = None
    constraints: Optional[ParameterConstraints] = None
    objectives: Optional[List[OptimizationObjective]] = None
    alloy: str = "Ti-6Al-4V"
    process: str = "laser_powder_bed_fusion"
    num_candidates: int = 10
    uncertainty_quantification: bool = True


class ParameterGenerationService:
    """Service for generating optimal process parameters."""
    
    def __init__(self, model: Optional[MicrostructureDiffusion] = None):
        """Initialize the parameter generation service.
        
        Args:
            model: Pre-trained diffusion model (if None, will load default)
        """
        self.model = model
        self._property_models = {}
        self._constraint_validators = {}
        
        # Initialize property prediction models
        self._initialize_property_models()
        
    def _initialize_property_models(self):
        """Initialize physics-based property prediction models."""
        
        # Simplified property models (in practice, these would be more sophisticated)
        self._property_models = {
            'density': self._predict_density,
            'strength': self._predict_strength,
            'surface_quality': self._predict_surface_quality,
            'porosity': self._predict_porosity,
            'hardness': self._predict_hardness
        }
    
    def generate_parameters(self, request: GenerationRequest) -> Dict[str, Any]:
        """Generate optimal parameters based on request.
        
        Args:
            request: Parameter generation request
            
        Returns:
            Dictionary containing generated parameters and metadata
        """
        
        # Validate inputs
        self._validate_request(request)
        
        # Initialize model if needed
        if self.model is None:
            self.model = MicrostructureDiffusion(
                alloy=request.alloy,
                process=request.process,
                pretrained=True
            )
        
        results = {
            'candidates': [],
            'best_candidate': None,
            'optimization_metrics': {},
            'constraints_satisfied': True,
            'warnings': []
        }
        
        try:
            if request.target_microstructure is not None:
                # Microstructure-driven generation
                candidates = self._generate_from_microstructure(request)
            elif request.target_properties is not None:
                # Property-driven generation
                candidates = self._generate_from_properties(request)
            else:
                # Constraint-based generation
                candidates = self._generate_from_constraints(request)
            
            # Evaluate and rank candidates
            evaluated_candidates = self._evaluate_candidates(candidates, request)
            
            # Select best candidate
            best_candidate = self._select_best_candidate(evaluated_candidates, request)
            
            results['candidates'] = evaluated_candidates
            results['best_candidate'] = best_candidate
            results['optimization_metrics'] = self._compute_optimization_metrics(
                evaluated_candidates, request
            )
            
        except Exception as e:
            warnings.warn(f"Parameter generation failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _validate_request(self, request: GenerationRequest):
        """Validate the generation request."""
        
        # Check that at least one target is specified
        if (request.target_microstructure is None and 
            request.target_properties is None and 
            request.constraints is None):
            raise ValueError("Must specify at least one of: target_microstructure, target_properties, or constraints")
        
        # Validate alloy-process compatibility
        validate_alloy_compatibility(request.alloy, request.process)
        
        # Validate target properties if specified
        if request.target_properties is not None:
            for prop, value in request.target_properties.items():
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Property {prop} must be numeric, got {type(value)}")
        
        # Validate constraints
        if request.constraints is not None:
            self._validate_constraints(request.constraints)
    
    def _validate_constraints(self, constraints: ParameterConstraints):
        """Validate parameter constraints."""
        
        constraint_dict = constraints.to_dict()
        
        for param, (min_val, max_val) in constraint_dict.items():
            if min_val >= max_val:
                raise ValueError(f"Invalid constraint for {param}: min ({min_val}) >= max ({max_val})")
            
            if min_val < 0:
                raise ValueError(f"Negative minimum value for {param}: {min_val}")
    
    def _generate_from_microstructure(self, request: GenerationRequest) -> List[ProcessParameters]:
        """Generate parameters from target microstructure."""
        
        # Use diffusion model for inverse design
        result = self.model.inverse_design(
            target_microstructure=request.target_microstructure,
            num_samples=request.num_candidates,
            uncertainty_quantification=request.uncertainty_quantification
        )
        
        if request.uncertainty_quantification:
            base_params, uncertainty = result
            # Generate variations based on uncertainty
            candidates = self._generate_uncertainty_variations(base_params, uncertainty, request.num_candidates)
        else:
            base_params = result
            candidates = [base_params]
            
            # Generate additional candidates through perturbation
            for _ in range(request.num_candidates - 1):
                perturbed = self._perturb_parameters(base_params)
                candidates.append(perturbed)
        
        return candidates
    
    def _generate_from_properties(self, request: GenerationRequest) -> List[ProcessParameters]:
        """Generate parameters from target properties."""
        
        # Use optimization to find parameters that achieve target properties
        candidates = []
        
        for _ in range(request.num_candidates):
            # Start with random initial parameters
            initial_params = self._generate_random_parameters(request.constraints)
            
            # Optimize towards target properties
            optimized_params = self._optimize_for_properties(
                initial_params, request.target_properties, request.constraints
            )
            
            candidates.append(optimized_params)
        
        return candidates
    
    def _generate_from_constraints(self, request: GenerationRequest) -> List[ProcessParameters]:
        """Generate parameters based only on constraints."""
        
        candidates = []
        
        for _ in range(request.num_candidates):
            params = self._generate_random_parameters(request.constraints)
            candidates.append(params)
        
        return candidates
    
    def _generate_random_parameters(self, constraints: Optional[ParameterConstraints] = None) -> ProcessParameters:
        """Generate random parameters within constraints."""
        
        if constraints is None:
            constraints = ParameterConstraints()
        
        constraint_dict = constraints.to_dict()
        
        # Sample randomly within constraints
        laser_power = np.random.uniform(*constraint_dict['laser_power'])
        scan_speed = np.random.uniform(*constraint_dict['scan_speed'])
        layer_thickness = np.random.uniform(*constraint_dict['layer_thickness'])
        hatch_spacing = np.random.uniform(*constraint_dict['hatch_spacing'])
        powder_bed_temp = np.random.uniform(*constraint_dict['powder_bed_temp'])
        
        return ProcessParameters(
            laser_power=laser_power,
            scan_speed=scan_speed,
            layer_thickness=layer_thickness,
            hatch_spacing=hatch_spacing,
            powder_bed_temp=powder_bed_temp
        )
    
    def _perturb_parameters(self, base_params: ProcessParameters, 
                           perturbation_scale: float = 0.1) -> ProcessParameters:
        """Generate perturbed version of parameters."""
        
        # Gaussian perturbations
        laser_power = base_params.laser_power * (1 + np.random.normal(0, perturbation_scale))
        scan_speed = base_params.scan_speed * (1 + np.random.normal(0, perturbation_scale))
        layer_thickness = base_params.layer_thickness * (1 + np.random.normal(0, perturbation_scale))
        hatch_spacing = base_params.hatch_spacing * (1 + np.random.normal(0, perturbation_scale))
        powder_bed_temp = base_params.powder_bed_temp * (1 + np.random.normal(0, perturbation_scale))
        
        # Ensure positive values
        laser_power = max(laser_power, 50.0)
        scan_speed = max(scan_speed, 200.0)
        layer_thickness = max(layer_thickness, 10.0)
        hatch_spacing = max(hatch_spacing, 50.0)
        powder_bed_temp = max(powder_bed_temp, 20.0)
        
        return ProcessParameters(
            laser_power=laser_power,
            scan_speed=scan_speed,
            layer_thickness=layer_thickness,
            hatch_spacing=hatch_spacing,
            powder_bed_temp=powder_bed_temp
        )
    
    def _generate_uncertainty_variations(self, base_params: ProcessParameters, 
                                       uncertainty: Dict[str, float],
                                       num_samples: int) -> List[ProcessParameters]:
        """Generate parameter variations based on uncertainty estimates."""
        
        candidates = [base_params]
        
        for _ in range(num_samples - 1):
            # Sample from uncertainty distribution
            laser_power = np.random.normal(
                base_params.laser_power, 
                uncertainty.get('laser_power_std', 0.1 * base_params.laser_power)
            )
            scan_speed = np.random.normal(
                base_params.scan_speed,
                uncertainty.get('scan_speed_std', 0.1 * base_params.scan_speed)
            )
            layer_thickness = np.random.normal(
                base_params.layer_thickness,
                uncertainty.get('layer_thickness_std', 0.1 * base_params.layer_thickness)
            )
            hatch_spacing = np.random.normal(
                base_params.hatch_spacing,
                uncertainty.get('hatch_spacing_std', 0.1 * base_params.hatch_spacing)
            )
            
            # Ensure positive values and reasonable ranges
            laser_power = np.clip(laser_power, 50.0, 500.0)
            scan_speed = np.clip(scan_speed, 200.0, 2000.0)
            layer_thickness = np.clip(layer_thickness, 10.0, 100.0)
            hatch_spacing = np.clip(hatch_spacing, 50.0, 300.0)
            
            candidate = ProcessParameters(
                laser_power=laser_power,
                scan_speed=scan_speed,
                layer_thickness=layer_thickness,
                hatch_spacing=hatch_spacing,
                powder_bed_temp=base_params.powder_bed_temp
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _optimize_for_properties(self, initial_params: ProcessParameters,
                                target_properties: Dict[str, float],
                                constraints: Optional[ParameterConstraints] = None) -> ProcessParameters:
        """Optimize parameters to achieve target properties."""
        
        # Simplified optimization using random search
        # In practice, would use more sophisticated methods like genetic algorithms or Bayesian optimization
        
        best_params = initial_params
        best_score = float('inf')
        
        for _ in range(100):  # Optimization iterations
            # Generate candidate parameters
            if np.random.random() < 0.5:
                # Perturbation of current best
                candidate = self._perturb_parameters(best_params, 0.2)
            else:
                # Random restart
                candidate = self._generate_random_parameters(constraints)
            
            # Apply constraints
            if constraints is not None:
                candidate = self._apply_constraints(candidate, constraints)
            
            # Evaluate candidate
            predicted_properties = self._predict_properties(candidate)
            score = self._compute_property_mismatch(predicted_properties, target_properties)
            
            if score < best_score:
                best_score = score
                best_params = candidate
        
        return best_params
    
    def _apply_constraints(self, params: ProcessParameters, 
                          constraints: ParameterConstraints) -> ProcessParameters:
        """Apply constraints to parameters."""
        
        constraint_dict = constraints.to_dict()
        
        laser_power = np.clip(params.laser_power, *constraint_dict['laser_power'])
        scan_speed = np.clip(params.scan_speed, *constraint_dict['scan_speed'])
        layer_thickness = np.clip(params.layer_thickness, *constraint_dict['layer_thickness'])
        hatch_spacing = np.clip(params.hatch_spacing, *constraint_dict['hatch_spacing'])
        powder_bed_temp = np.clip(params.powder_bed_temp, *constraint_dict['powder_bed_temp'])
        
        return ProcessParameters(
            laser_power=laser_power,
            scan_speed=scan_speed,
            layer_thickness=layer_thickness,
            hatch_spacing=hatch_spacing,
            powder_bed_temp=powder_bed_temp,
            atmosphere=params.atmosphere
        )
    
    def _evaluate_candidates(self, candidates: List[ProcessParameters], 
                           request: GenerationRequest) -> List[Dict[str, Any]]:
        """Evaluate and score parameter candidates."""
        
        evaluated = []
        
        for i, candidate in enumerate(candidates):
            evaluation = {
                'index': i,
                'parameters': candidate,
                'predicted_properties': self._predict_properties(candidate),
                'constraint_violations': self._check_constraint_violations(candidate, request.constraints),
                'scores': {}
            }
            
            # Compute various scores
            if request.target_properties is not None:
                evaluation['scores']['property_match'] = self._compute_property_match_score(
                    evaluation['predicted_properties'], request.target_properties
                )
            
            if request.objectives is not None:
                evaluation['scores']['objectives'] = self._compute_objective_scores(
                    candidate, request.objectives
                )
            
            # Overall manufacturability score
            evaluation['scores']['manufacturability'] = self._compute_manufacturability_score(candidate)
            
            # Energy efficiency score
            evaluation['scores']['energy_efficiency'] = self._compute_energy_efficiency_score(candidate)
            
            evaluated.append(evaluation)
        
        return evaluated
    
    def _predict_properties(self, params: ProcessParameters) -> Dict[str, float]:
        """Predict material properties from process parameters."""
        
        properties = {}
        
        for prop_name, predictor in self._property_models.items():
            properties[prop_name] = predictor(params)
        
        return properties
    
    def _predict_density(self, params: ProcessParameters) -> float:
        """Predict relative density from process parameters."""
        
        # Simplified model based on energy density
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Empirical relationship (simplified)
        if energy_density < 40:
            density = 0.85 + 0.01 * energy_density
        elif energy_density > 120:
            density = 0.98 - 0.001 * (energy_density - 120)
        else:
            density = 0.85 + 0.0175 * (energy_density - 40)
        
        return np.clip(density, 0.0, 1.0)
    
    def _predict_strength(self, params: ProcessParameters) -> float:
        """Predict tensile strength from process parameters."""
        
        # Simplified model
        base_strength = 900  # MPa for Ti-6Al-4V
        
        # Effect of energy density
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density < 60:
            strength_factor = 0.8 + 0.003 * energy_density
        else:
            strength_factor = 1.0 + 0.001 * (energy_density - 60)
        
        # Effect of layer thickness (thinner layers -> finer microstructure -> higher strength)
        layer_factor = 1.0 + 0.002 * (50 - params.layer_thickness)
        
        return base_strength * strength_factor * layer_factor
    
    def _predict_surface_quality(self, params: ProcessParameters) -> float:
        """Predict surface roughness (Ra in micrometers)."""
        
        # Simplified model - lower values are better
        base_roughness = 10.0  # micrometers
        
        # Effect of layer thickness
        layer_effect = params.layer_thickness / 30.0  # Normalized to typical value
        
        # Effect of energy density
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density > 100:
            energy_effect = 1.0 + 0.01 * (energy_density - 100)  # Overmelting increases roughness
        else:
            energy_effect = 1.0
        
        return base_roughness * layer_effect * energy_effect
    
    def _predict_porosity(self, params: ProcessParameters) -> float:
        """Predict porosity percentage."""
        
        density = self._predict_density(params)
        return (1.0 - density) * 100  # Convert to percentage
    
    def _predict_hardness(self, params: ProcessParameters) -> float:
        """Predict hardness (HV)."""
        
        # Simplified relationship with strength
        strength = self._predict_strength(params)
        hardness = strength / 3.0  # Rough conversion from MPa to HV
        
        return hardness
    
    def _check_constraint_violations(self, params: ProcessParameters,
                                   constraints: Optional[ParameterConstraints] = None) -> List[str]:
        """Check for constraint violations."""
        
        violations = []
        
        if constraints is None:
            return violations
        
        constraint_dict = constraints.to_dict()
        param_dict = params.to_dict()
        
        for param_name, (min_val, max_val) in constraint_dict.items():
            if param_name in param_dict:
                value = param_dict[param_name]
                if isinstance(value, (int, float)):
                    if value < min_val:
                        violations.append(f"{param_name} too low: {value} < {min_val}")
                    elif value > max_val:
                        violations.append(f"{param_name} too high: {value} > {max_val}")
        
        return violations
    
    def _compute_property_match_score(self, predicted: Dict[str, float], 
                                    target: Dict[str, float]) -> float:
        """Compute how well predicted properties match targets."""
        
        total_error = 0.0
        num_properties = 0
        
        for prop_name, target_value in target.items():
            if prop_name in predicted:
                predicted_value = predicted[prop_name]
                # Normalized relative error
                relative_error = abs(predicted_value - target_value) / (abs(target_value) + 1e-6)
                total_error += relative_error
                num_properties += 1
        
        if num_properties == 0:
            return 0.0
        
        # Convert to score (higher is better)
        avg_error = total_error / num_properties
        score = 1.0 / (1.0 + avg_error)
        
        return score
    
    def _compute_property_mismatch(self, predicted: Dict[str, float], 
                                 target: Dict[str, float]) -> float:
        """Compute mismatch between predicted and target properties (for optimization)."""
        
        total_error = 0.0
        
        for prop_name, target_value in target.items():
            if prop_name in predicted:
                predicted_value = predicted[prop_name]
                relative_error = abs(predicted_value - target_value) / (abs(target_value) + 1e-6)
                total_error += relative_error ** 2
        
        return total_error
    
    def _compute_objective_scores(self, params: ProcessParameters, 
                                objectives: List[OptimizationObjective]) -> Dict[str, float]:
        """Compute scores for optimization objectives."""
        
        scores = {}
        
        for objective in objectives:
            if objective == OptimizationObjective.DENSITY:
                scores['density'] = self._predict_density(params)
            elif objective == OptimizationObjective.STRENGTH:
                # Normalize to 0-1 scale
                strength = self._predict_strength(params)
                scores['strength'] = min(strength / 1200.0, 1.0)  # Assume max target of 1200 MPa
            elif objective == OptimizationObjective.SURFACE_QUALITY:
                # Lower roughness is better
                roughness = self._predict_surface_quality(params)
                scores['surface_quality'] = max(0.0, 1.0 - roughness / 20.0)  # Assume max acceptable of 20 μm
            elif objective == OptimizationObjective.BUILD_SPEED:
                # Higher scan speed is faster
                scores['build_speed'] = min(params.scan_speed / 2000.0, 1.0)
            elif objective == OptimizationObjective.ENERGY_EFFICIENCY:
                scores['energy_efficiency'] = self._compute_energy_efficiency_score(params)
        
        return scores
    
    def _compute_manufacturability_score(self, params: ProcessParameters) -> float:
        """Compute overall manufacturability score."""
        
        # Factors that affect manufacturability
        scores = []
        
        # Energy density should be in optimal range
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if 60 <= energy_density <= 100:
            energy_score = 1.0
        else:
            energy_score = max(0.0, 1.0 - 0.01 * abs(energy_density - 80))
        
        scores.append(energy_score)
        
        # Aspect ratios should be reasonable
        aspect_ratio = params.hatch_spacing / params.layer_thickness
        if 2 <= aspect_ratio <= 6:
            aspect_score = 1.0
        else:
            aspect_score = max(0.0, 1.0 - 0.1 * abs(aspect_ratio - 4))
        
        scores.append(aspect_score)
        
        return np.mean(scores)
    
    def _compute_energy_efficiency_score(self, params: ProcessParameters) -> float:
        """Compute energy efficiency score."""
        
        # Lower energy per unit volume is more efficient
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Optimal range around 60-80 J/mm³
        if 60 <= energy_density <= 80:
            return 1.0
        else:
            return max(0.0, 1.0 - 0.01 * abs(energy_density - 70))
    
    def _select_best_candidate(self, evaluated_candidates: List[Dict[str, Any]], 
                              request: GenerationRequest) -> Dict[str, Any]:
        """Select the best candidate based on multiple criteria."""
        
        if not evaluated_candidates:
            return None
        
        # Compute overall scores
        for candidate in evaluated_candidates:
            overall_score = 0.0
            weight_sum = 0.0
            
            # Property match (high weight if specified)
            if 'property_match' in candidate['scores']:
                overall_score += 3.0 * candidate['scores']['property_match']
                weight_sum += 3.0
            
            # Manufacturability (always important)
            overall_score += 2.0 * candidate['scores']['manufacturability']
            weight_sum += 2.0
            
            # Energy efficiency
            overall_score += 1.0 * candidate['scores']['energy_efficiency']
            weight_sum += 1.0
            
            # Penalty for constraint violations
            num_violations = len(candidate['constraint_violations'])
            violation_penalty = min(num_violations * 0.2, 0.8)
            
            candidate['overall_score'] = (overall_score / weight_sum) * (1.0 - violation_penalty)
        
        # Sort by overall score
        evaluated_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return evaluated_candidates[0]
    
    def _compute_optimization_metrics(self, evaluated_candidates: List[Dict[str, Any]], 
                                    request: GenerationRequest) -> Dict[str, Any]:
        """Compute metrics about the optimization process."""
        
        metrics = {
            'num_candidates': len(evaluated_candidates),
            'num_feasible': len([c for c in evaluated_candidates if not c['constraint_violations']]),
            'score_statistics': {},
            'property_ranges': {}
        }
        
        if evaluated_candidates:
            # Score statistics
            overall_scores = [c['overall_score'] for c in evaluated_candidates]
            metrics['score_statistics'] = {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': np.min(overall_scores),
                'max': np.max(overall_scores)
            }
            
            # Property ranges
            if evaluated_candidates[0]['predicted_properties']:
                for prop_name in evaluated_candidates[0]['predicted_properties'].keys():
                    values = [c['predicted_properties'][prop_name] for c in evaluated_candidates]
                    metrics['property_ranges'][prop_name] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        
        return metrics