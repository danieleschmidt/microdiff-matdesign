"""Core diffusion model functionality."""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from .models.diffusion import DiffusionModel
from .models.encoders import MicrostructureEncoder
from .models.decoders import ParameterDecoder
from .utils.validation import validate_microstructure, validate_parameters
from .utils.preprocessing import normalize_microstructure, denormalize_parameters
from .utils.error_handling import handle_errors, error_context, validate_input, ValidationError, ModelError, ProcessingError
from .utils.logging_config import get_logger, with_logging, LoggingContextManager
from .utils.robust_validation import RobustValidator

# Optional imports for Generation 3 features with graceful fallback
try:
    from .utils.performance import PerformanceConfig, ResourceManager, ParallelProcessor
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PerformanceConfig = ResourceManager = ParallelProcessor = None
    PERFORMANCE_AVAILABLE = False

try:
    from .utils.caching import CacheManager, CacheConfig
    CACHING_AVAILABLE = True
except ImportError:
    CacheManager = CacheConfig = None
    CACHING_AVAILABLE = False

try:
    from .utils.scaling import LoadBalancer, ScalingConfig
    SCALING_AVAILABLE = True
except ImportError:
    LoadBalancer = ScalingConfig = None
    SCALING_AVAILABLE = False


class ProcessParameters:
    """Container for manufacturing process parameters."""
    
    def __init__(self, **kwargs):
        self.laser_power = kwargs.get('laser_power', 200.0)
        self.scan_speed = kwargs.get('scan_speed', 800.0) 
        self.layer_thickness = kwargs.get('layer_thickness', 30.0)
        self.hatch_spacing = kwargs.get('hatch_spacing', 120.0)
        self.powder_bed_temp = kwargs.get('powder_bed_temp', 80.0)
        self.atmosphere = kwargs.get('atmosphere', 'argon')
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'laser_power': self.laser_power,
            'scan_speed': self.scan_speed,
            'layer_thickness': self.layer_thickness,
            'hatch_spacing': self.hatch_spacing,
            'powder_bed_temp': self.powder_bed_temp,
            'atmosphere': self.atmosphere
        }
        
    def to_tensor(self) -> torch.Tensor:
        """Convert parameters to normalized tensor for model input."""
        values = [self.laser_power, self.scan_speed, self.layer_thickness, 
                 self.hatch_spacing, self.powder_bed_temp]
        normalized = [(v - mean) / std for v, mean, std in zip(values, 
                     [200, 800, 30, 120, 80], [50, 200, 10, 30, 20])]
        return torch.tensor(normalized, dtype=torch.float32)


class MicrostructureDiffusion:
    """Main diffusion model for microstructure inverse design."""
    
    def __init__(
        self,
        alloy: str = "Ti-6Al-4V",
        process: str = "laser_powder_bed_fusion",
        pretrained: bool = True,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_validation: bool = True,
        safety_checks: bool = True,
        enable_scaling: bool = True,
        enable_caching: bool = True,
        performance_config: Optional[PerformanceConfig] = None
    ):
        """Initialize the diffusion model."""
        self.logger = get_logger('core.MicrostructureDiffusion')
        
        with LoggingContextManager(self.logger, "model_initialization"):
            # Validate inputs
            if enable_validation:
                self.validator = RobustValidator()
                self._validate_initialization_params(alloy, process, device)
            
            self.alloy = alloy
            self.process = process
            self.pretrained = pretrained
            self.enable_validation = enable_validation
            self.safety_checks = safety_checks
            self.enable_scaling = enable_scaling
            self.enable_caching = enable_caching
            
            # Initialize performance and scaling components for Generation 3
            if PERFORMANCE_AVAILABLE and performance_config is None:
                self.performance_config = PerformanceConfig()
            else:
                self.performance_config = performance_config
            
            if enable_scaling and PERFORMANCE_AVAILABLE and SCALING_AVAILABLE:
                self.resource_manager = ResourceManager(self.performance_config)
                self.parallel_processor = ParallelProcessor(self.performance_config)
                self.load_balancer = LoadBalancer(ScalingConfig())
                self.logger.info("Scaling capabilities enabled")
            else:
                self.resource_manager = None
                self.parallel_processor = None
                self.load_balancer = None
                if enable_scaling:
                    self.logger.warning("Scaling requested but dependencies unavailable - using fallback mode")
            
            if enable_caching and CACHING_AVAILABLE:
                cache_config = CacheConfig(
                    max_size=1000,  # Cache up to 1000 inference results
                    ttl_seconds=3600,  # 1 hour TTL
                    enable_disk_cache=True
                )
                self.cache_manager = CacheManager(cache_config)
                self.logger.info("Intelligent caching enabled")
            else:
                self.cache_manager = None
                if enable_caching:
                    self.logger.warning("Caching requested but dependencies unavailable - using fallback mode")
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Model architecture parameters
        self.config = self._load_config()
        
        # Initialize model components
        self.encoder = MicrostructureEncoder(
            input_dim=self.config['encoder']['input_dim'],
            hidden_dim=self.config['encoder']['hidden_dim'],
            latent_dim=self.config['encoder']['latent_dim']
        ).to(self.device)
        
        self.diffusion_model = DiffusionModel(
            input_dim=self.config['diffusion']['input_dim'],
            hidden_dim=self.config['diffusion']['hidden_dim'],
            num_steps=self.config['diffusion']['num_steps']
        ).to(self.device)
        
        self.decoder = ParameterDecoder(
            latent_dim=self.config['decoder']['latent_dim'],
            hidden_dim=self.config['decoder']['hidden_dim'],
            output_dim=self.config['decoder']['output_dim']
        ).to(self.device)
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(model_path)
            
        # Set to evaluation mode by default
        self.eval()
            
    def _validate_initialization_params(self, alloy: str, process: str, device: Optional[str]):
        """Validate initialization parameters."""
        valid_alloys = {"Ti-6Al-4V", "Inconel-718", "AlSi10Mg", "316L"}
        valid_processes = {"laser_powder_bed_fusion", "electron_beam_melting", "directed_energy_deposition"}
        
        validate_input(
            alloy in valid_alloys,
            f"Unsupported alloy: {alloy}. Supported: {valid_alloys}",
            ValidationError
        )
        
        validate_input(
            process in valid_processes,
            f"Unsupported process: {process}. Supported: {valid_processes}",
            ValidationError
        )
        
        if device is not None:
            validate_input(
                device in {"cpu", "cuda", "auto"},
                f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'auto'",
                ValidationError
            )
    
    def _validate_inverse_design_inputs(self, microstructure: np.ndarray, 
                                      num_samples: int, guidance_scale: float):
        """Validate inputs for inverse design."""
        # Microstructure validation
        validate_input(
            isinstance(microstructure, np.ndarray),
            "Microstructure must be a numpy array",
            ValidationError
        )
        
        validate_input(
            len(microstructure.shape) == 3,
            f"Microstructure must be 3D, got shape {microstructure.shape}",
            ValidationError
        )
        
        validate_input(
            microstructure.size > 0,
            "Microstructure cannot be empty",
            ValidationError
        )
        
        # Check for NaN or infinite values
        validate_input(
            np.isfinite(microstructure).all(),
            "Microstructure contains NaN or infinite values",
            ValidationError
        )
        
        # Parameter validation
        validate_input(
            isinstance(num_samples, int) and num_samples > 0,
            f"num_samples must be positive integer, got {num_samples}",
            ValidationError
        )
        
        validate_input(
            isinstance(guidance_scale, (int, float)) and guidance_scale > 0,
            f"guidance_scale must be positive number, got {guidance_scale}",
            ValidationError
        )
        
        # Performance warnings
        if num_samples > 50:
            self.logger.warning(f"Large num_samples ({num_samples}) may be slow")
        
        if guidance_scale > 20:
            self.logger.warning(f"High guidance_scale ({guidance_scale}) may cause artifacts")
    
    def _get_parameter_bounds(self) -> Dict[str, np.ndarray]:
        """Get process-specific parameter bounds."""
        if self.process == "laser_powder_bed_fusion":
            return {
                'min': np.array([50.0, 200.0, 10.0, 50.0, 25.0]),
                'max': np.array([500.0, 2000.0, 100.0, 300.0, 200.0]),
                'default': np.array([200.0, 800.0, 30.0, 120.0, 80.0])
            }
        elif self.process == "electron_beam_melting":
            return {
                'min': np.array([100.0, 50.0, 20.0, 80.0, 200.0]),
                'max': np.array([1000.0, 500.0, 150.0, 400.0, 600.0]),
                'default': np.array([400.0, 200.0, 50.0, 150.0, 350.0])
            }
        else:  # Default bounds
            return {
                'min': np.array([100.0, 400.0, 20.0, 80.0, 50.0]),
                'max': np.array([400.0, 1500.0, 80.0, 250.0, 150.0]),
                'default': np.array([200.0, 800.0, 30.0, 120.0, 80.0])
            }
    
    def _assess_sample_quality(self, parameter_samples: np.ndarray) -> str:
        """Assess the quality of parameter samples."""
        if len(parameter_samples) < 2:
            return "insufficient_samples"
        
        # Calculate coefficient of variation for each parameter
        means = np.mean(parameter_samples, axis=0)
        stds = np.std(parameter_samples, axis=0)
        
        # Avoid division by zero
        cvs = np.where(means != 0, stds / np.abs(means), np.inf)
        avg_cv = np.mean(cvs[np.isfinite(cvs)])
        
        if avg_cv < 0.1:
            return "high_confidence"
        elif avg_cv < 0.3:
            return "medium_confidence"
        elif avg_cv < 0.5:
            return "low_confidence"
        else:
            return "very_uncertain"
    
    def _generate_cache_key(self, microstructure: np.ndarray, num_samples: int, 
                           guidance_scale: float, uncertainty_quantification: bool) -> str:
        """Generate a cache key for inverse design results."""
        import hashlib
        
        # Create hash of microstructure
        microstructure_hash = hashlib.md5(microstructure.tobytes()).hexdigest()[:16]
        
        # Create cache key from parameters
        cache_key = (
            f"inverse_design_{microstructure_hash}_{self.alloy}_{self.process}_"
            f"{num_samples}_{guidance_scale:.2f}_{uncertainty_quantification}"
        )
        return cache_key
    
    def _generate_samples_sequential(self, latent_encoding: torch.Tensor, 
                                   num_samples: int, guidance_scale: float) -> List[np.ndarray]:
        """Generate samples sequentially (original approach)."""
        parameter_samples = []
        
        for sample_idx in range(num_samples):
            try:
                # Sample noise and condition on microstructure encoding
                noise = torch.randn(latent_encoding.shape, device=self.device)
                
                # Iterative denoising with guidance
                denoised_latent = self.diffusion_model.sample(
                    noise, 
                    condition=latent_encoding,
                    guidance_scale=guidance_scale
                )
                
                # Safety check for denoised latent
                if self.safety_checks:
                    validate_input(
                        torch.isfinite(denoised_latent).all(),
                        f"Diffusion model produced non-finite values at sample {sample_idx}",
                        ModelError
                    )
                
                # Decode to process parameters
                parameter_tensor = self.decoder(denoised_latent)
                
                # Safety check for parameters
                if self.safety_checks:
                    validate_input(
                        torch.isfinite(parameter_tensor).all(),
                        f"Decoder produced non-finite values at sample {sample_idx}",
                        ModelError
                    )
                
                parameter_samples.append(parameter_tensor.cpu().numpy())
                
            except Exception as e:
                self.logger.warning(f"Failed to generate sample {sample_idx}: {e}")
                if len(parameter_samples) == 0 and sample_idx == num_samples - 1:
                    raise ProcessingError(
                        f"Failed to generate any valid samples after {num_samples} attempts"
                    ) from e
                continue
                
        return parameter_samples
    
    def _generate_samples_parallel(self, latent_encoding: torch.Tensor, 
                                 num_samples: int, guidance_scale: float,
                                 batch_size: int) -> List[np.ndarray]:
        """Generate samples in parallel batches for better performance."""
        parameter_samples = []
        
        # Split into batches for parallel processing
        batches = []
        for i in range(0, num_samples, batch_size):
            batch_samples = min(batch_size, num_samples - i)
            batches.append(batch_samples)
        
        # Process batches in parallel using ThreadPoolExecutor for I/O bound operations
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def generate_batch(batch_size: int) -> List[np.ndarray]:
            """Generate a batch of samples."""
            batch_results = []
            for _ in range(batch_size):
                try:
                    noise = torch.randn(latent_encoding.shape, device=self.device)
                    denoised_latent = self.diffusion_model.sample(
                        noise, condition=latent_encoding, guidance_scale=guidance_scale
                    )
                    
                    if self.safety_checks and not torch.isfinite(denoised_latent).all():
                        continue
                        
                    parameter_tensor = self.decoder(denoised_latent)
                    
                    if self.safety_checks and not torch.isfinite(parameter_tensor).all():
                        continue
                        
                    batch_results.append(parameter_tensor.cpu().numpy())
                except Exception as e:
                    self.logger.warning(f"Failed to generate sample in batch: {e}")
                    continue
            return batch_results
        
        # Execute batches in parallel
        max_workers = min(len(batches), 
                         self.resource_manager.optimal_thread_workers if self.resource_manager else 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(generate_batch, batch_size): batch_size 
                for batch_size in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result(timeout=30)  # 30 second timeout per batch
                    parameter_samples.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
        
        self.logger.info(f"Parallel generation completed: {len(parameter_samples)} samples from {len(batches)} batches")
        return parameter_samples
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = Path(__file__).parent / 'configs' / f'{self.alloy.lower().replace("-", "_")}_{self.process}.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration - simplified for Generation 1
            return {
                'encoder': {'input_dim': 64*64*64, 'hidden_dim': 256, 'latent_dim': 128},
                'diffusion': {'input_dim': 128, 'hidden_dim': 256, 'num_steps': 10},  # Very simple for testing
                'decoder': {'latent_dim': 128, 'hidden_dim': 256, 'output_dim': 5}
            }
    
    def _load_pretrained_weights(self, model_path: Optional[str] = None):
        """Load pretrained model weights."""
        if model_path is None:
            model_path = Path(__file__).parent / 'pretrained' / f'{self.alloy}_{self.process}.pth'
            
        if Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.diffusion_model.load_state_dict(checkpoint['diffusion'])
                self.decoder.load_state_dict(checkpoint['decoder'])
                print(f"Loaded pretrained weights from {model_path}")
            except Exception as e:
                warnings.warn(f"Failed to load pretrained weights: {e}")
        else:
            warnings.warn(f"Pretrained model not found at {model_path}. Using random initialization.")
    
    def train(self):
        """Set model to training mode."""
        self.encoder.train()
        self.diffusion_model.train()
        self.decoder.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.encoder.eval()
        self.diffusion_model.eval()
        self.decoder.eval()
        
    @handle_errors(operation="inverse_design", reraise=True)
    @with_logging("inverse_design")
    def inverse_design(
        self,
        target_microstructure: np.ndarray,
        num_samples: int = 10,
        guidance_scale: float = 7.5,
        uncertainty_quantification: bool = False,
        enable_parallel: Optional[bool] = None,
        batch_size: Optional[int] = None
    ) -> Union[ProcessParameters, Tuple[ProcessParameters, Dict[str, float]]]:
        """Generate process parameters from target microstructure."""
        
        # Generation 3: Check cache first for performance
        cache_key = None
        if self.enable_caching and self.cache_manager is not None:
            cache_key = self._generate_cache_key(
                target_microstructure, num_samples, guidance_scale, uncertainty_quantification
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info("Returning cached result for inverse design")
                return cached_result
        
        # Determine parallel processing settings
        use_parallel = enable_parallel if enable_parallel is not None else (
            self.enable_scaling and self.resource_manager is not None and num_samples >= 4
        )
        effective_batch_size = batch_size or min(num_samples, 4)
        
        if use_parallel:
            self.logger.info(f"Using parallel processing with batch_size={effective_batch_size}")
        
        # Enhanced validation for Generation 2
        if self.enable_validation:
            self._validate_inverse_design_inputs(
                target_microstructure, num_samples, guidance_scale
            )
        
        with error_context("microstructure_validation"):
            validate_microstructure(target_microstructure)
        
        try:
            # Preprocess microstructure with error handling
            with error_context("microstructure_preprocessing"):
                microstructure_tensor = normalize_microstructure(target_microstructure)
                microstructure_tensor = torch.from_numpy(microstructure_tensor).float().to(self.device)
                
                if len(microstructure_tensor.shape) == 3:
                    microstructure_tensor = microstructure_tensor.unsqueeze(0)  # Add batch dimension
            
            # Model inference with comprehensive error handling    
            with torch.no_grad():
                with error_context("microstructure_encoding"):
                    # Encode microstructure to latent space
                    latent_encoding = self.encoder(microstructure_tensor.flatten(1))
                    
                    # Validate latent encoding
                    if self.safety_checks:
                        validate_input(
                            torch.isfinite(latent_encoding).all(),
                            "Encoder produced non-finite values",
                            ModelError
                        )
                
                # Generate multiple samples with parallel processing for Generation 3
                parameter_samples = []
                
                with error_context("parameter_generation"):
                    if use_parallel and self.enable_scaling and self.resource_manager is not None:
                        # Parallel sample generation for better performance
                        parameter_samples = self._generate_samples_parallel(
                            latent_encoding, num_samples, guidance_scale, effective_batch_size
                        )
                    else:
                        # Sequential generation (original approach)
                        parameter_samples = self._generate_samples_sequential(
                            latent_encoding, num_samples, guidance_scale
                        )
        
        except torch.cuda.OutOfMemoryError as e:
            raise ResourceError(
                "CUDA out of memory during inference. Try reducing num_samples or input size."
            ) from e
        except Exception as e:
            raise ProcessingError(f"Model inference failed: {str(e)}") from e
                
        # Robust parameter processing
        if len(parameter_samples) == 0:
            raise ProcessingError("No valid parameter samples generated")
        
        with error_context("parameter_processing"):
            # Convert to ProcessParameters objects
            parameter_samples = np.array(parameter_samples)
            self.logger.info(f"Generated {len(parameter_samples)} parameter samples")
            
            # Validate parameter samples
            if self.safety_checks:
                validate_input(
                    np.isfinite(parameter_samples).all(),
                    "Parameter samples contain non-finite values",
                    ModelError
                )
            
            # Denormalize parameters with error handling
            try:
                denormalized_params = denormalize_parameters(parameter_samples)
            except Exception as e:
                raise ProcessingError(f"Parameter denormalization failed: {e}") from e
            
            # Calculate statistics
            try:
                mean_params = np.mean(denormalized_params, axis=0)
                
                # Handle both single sample and batch cases
                if mean_params.ndim > 1:
                    mean_params = mean_params[0]  # Take first sample if batch
                
                # Robust parameter bounds for different processes
                bounds = self._get_parameter_bounds()
                
                # Clip parameters to valid ranges
                clipped_params = np.clip(mean_params, bounds['min'], bounds['max'])
                
                # Log if significant clipping occurred
                if np.any(np.abs(clipped_params - mean_params) > 0.1 * mean_params):
                    self.logger.warning("Significant parameter clipping occurred - model may need retraining")
                
                # Create ProcessParameters object with robust defaults
                result_params = ProcessParameters(
                    laser_power=float(clipped_params[0]) if len(clipped_params) > 0 else bounds['default'][0],
                    scan_speed=float(clipped_params[1]) if len(clipped_params) > 1 else bounds['default'][1],
                    layer_thickness=float(clipped_params[2]) if len(clipped_params) > 2 else bounds['default'][2],
                    hatch_spacing=float(clipped_params[3]) if len(clipped_params) > 3 else bounds['default'][3],
                    powder_bed_temp=float(clipped_params[4]) if len(clipped_params) > 4 else bounds['default'][4]
                )
                
            except Exception as e:
                raise ProcessingError(f"Parameter statistics calculation failed: {e}") from e
        
        # Validate final parameters
        with error_context("parameter_validation"):
            validate_parameters(result_params.to_dict(), self.process)
        
        # Enhanced uncertainty quantification
        if uncertainty_quantification:
            with error_context("uncertainty_calculation"):
                try:
                    # Calculate robust uncertainty metrics
                    std_params = np.std(denormalized_params, axis=0)
                    
                    # Calculate confidence intervals (assuming normal distribution)
                    confidence_level = 0.95
                    z_score = 1.96  # For 95% confidence
                    
                    # Create comprehensive uncertainty dictionary
                    uncertainty = {
                        'laser_power_std': float(std_params[0]) if len(std_params) > 0 else 0.0,
                        'scan_speed_std': float(std_params[1]) if len(std_params) > 1 else 0.0,
                        'layer_thickness_std': float(std_params[2]) if len(std_params) > 2 else 0.0,
                        'hatch_spacing_std': float(std_params[3]) if len(std_params) > 3 else 0.0,
                        'powder_bed_temp_std': float(std_params[4]) if len(std_params) > 4 else 0.0,
                        'confidence_level': confidence_level,
                        'num_samples': len(parameter_samples),
                        'sample_quality': self._assess_sample_quality(denormalized_params)
                    }
                    
                    # Add confidence intervals
                    param_names = ['laser_power', 'scan_speed', 'layer_thickness', 'hatch_spacing', 'powder_bed_temp']
                    for i, param_name in enumerate(param_names[:len(mean_params)]):
                        margin_of_error = z_score * std_params[i] / np.sqrt(len(parameter_samples))
                        uncertainty[f'{param_name}_ci_lower'] = float(mean_params[i] - margin_of_error)
                        uncertainty[f'{param_name}_ci_upper'] = float(mean_params[i] + margin_of_error)
                    
                    # Log uncertainty assessment
                    avg_uncertainty = np.mean(std_params[:len(mean_params)])
                    if avg_uncertainty > 0.2 * np.mean(mean_params):
                        self.logger.warning(f"High parameter uncertainty detected (avg std: {avg_uncertainty:.2f})")
                    
                    self.logger.info(f"Uncertainty quantification completed for {len(parameter_samples)} samples")
                    
                    return result_params, uncertainty
                    
                except Exception as e:
                    self.logger.error(f"Uncertainty quantification failed: {e}")
                    # Return parameters without uncertainty if calculation fails
                    return result_params, {'error': f'Uncertainty calculation failed: {str(e)}'}
        
        # Cache the result for future use (Generation 3)
        if self.enable_caching and cache_key is not None and self.cache_manager is not None:
            result_to_cache = (result_params, uncertainty) if uncertainty_quantification else result_params
            self.cache_manager.set(cache_key, result_to_cache)
            self.logger.info("Cached inverse design result")
        
        self.logger.info("Inverse design completed successfully")
        return result_params
    
    def predict_microstructure(
        self,
        parameters: ProcessParameters,
        num_samples: int = 1
    ) -> np.ndarray:
        """Forward prediction: generate microstructure from parameters."""
        
        parameter_tensor = parameters.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use decoder in reverse mode (this is a simplified approach)
            latent = self.decoder.inverse(parameter_tensor)
            
            # Generate microstructure samples
            microstructures = []
            for _ in range(num_samples):
                noise = torch.randn_like(latent)
                conditioned_latent = latent + 0.1 * noise  # Add controlled noise
                
                # Decode to microstructure
                microstructure = self.encoder.decode(conditioned_latent)
                microstructures.append(microstructure.cpu().numpy())
                
        if num_samples == 1:
            return microstructures[0].reshape(128, 128, 128)
        else:
            return np.array(microstructures).reshape(num_samples, 128, 128, 128)
    
    def optimize_parameters(
        self,
        target_properties: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        num_iterations: int = 100
    ) -> ProcessParameters:
        """Optimize parameters for target material properties."""
        
        # Initialize with random parameters
        best_params = ProcessParameters()
        best_score = float('inf')
        
        for _ in range(num_iterations):
            # Generate random perturbation
            perturbation = {
                'laser_power': np.random.normal(0, 10),
                'scan_speed': np.random.normal(0, 50),
                'layer_thickness': np.random.normal(0, 2),
                'hatch_spacing': np.random.normal(0, 10)
            }
            
            # Apply perturbation
            trial_params = ProcessParameters(
                laser_power=best_params.laser_power + perturbation['laser_power'],
                scan_speed=best_params.scan_speed + perturbation['scan_speed'],
                layer_thickness=best_params.layer_thickness + perturbation['layer_thickness'],
                hatch_spacing=best_params.hatch_spacing + perturbation['hatch_spacing']
            )
            
            # Apply constraints
            if constraints:
                trial_params = self._apply_constraints(trial_params, constraints)
            
            # Evaluate parameters (simplified scoring)
            score = self._evaluate_parameters(trial_params, target_properties)
            
            if score < best_score:
                best_score = score
                best_params = trial_params
                
        return best_params
    
    def _apply_constraints(
        self, 
        params: ProcessParameters, 
        constraints: Dict[str, Tuple[float, float]]
    ) -> ProcessParameters:
        """Apply parameter constraints."""
        
        result = ProcessParameters()
        
        for param_name, (min_val, max_val) in constraints.items():
            if hasattr(params, param_name):
                current_val = getattr(params, param_name)
                constrained_val = np.clip(current_val, min_val, max_val)
                setattr(result, param_name, constrained_val)
                
        return result
    
    def _evaluate_parameters(
        self, 
        params: ProcessParameters, 
        target_properties: Dict[str, float]
    ) -> float:
        """Evaluate parameter quality based on target properties."""
        
        # Simplified property prediction (in practice, this would use 
        # physics-based models or additional ML models)
        predicted_properties = {
            'density': 0.99 - (params.scan_speed - 800) / 2000,
            'roughness': 5 + (params.laser_power - 200) / 40,
            'strength': 1000 + (params.laser_power - 200) * 2
        }
        
        # Calculate weighted MSE
        score = 0.0
        for prop, target_val in target_properties.items():
            if prop in predicted_properties:
                error = (predicted_properties[prop] - target_val) ** 2
                score += error
                
        return score
    
    def save_model(self, save_path: str):
        """Save model weights and configuration."""
        
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'diffusion': self.diffusion_model.state_dict(),
            'decoder': self.decoder.state_dict(),
            'config': self.config,
            'alloy': self.alloy,
            'process': self.process
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model weights and configuration."""
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        self.config = checkpoint['config']
        self.alloy = checkpoint['alloy']
        self.process = checkpoint['process']
        
        print(f"Model loaded from {load_path}")


def train_diffusion_model(
    dataset,
    architecture: str = "unet3d",
    diffusion_steps: int = 1000,
    batch_size: int = 8,
    epochs: int = 500,
    learning_rate: float = 1e-4,
    device: Optional[str] = None
) -> MicrostructureDiffusion:
    """Train a diffusion model on the provided dataset."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = MicrostructureDiffusion(pretrained=False, device=device)
    model.train()
    
    # Setup optimization
    params = list(model.encoder.parameters()) + \
             list(model.diffusion_model.parameters()) + \
             list(model.decoder.parameters())
    
    optimizer = AdamW(params, lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            microstructures, parameters = batch
            microstructures = microstructures.to(device)
            parameters = parameters.to(device)
            
            optimizer.zero_grad()
            
            # Encode microstructures
            latent_micro = model.encoder(microstructures.flatten(1))
            
            # Forward diffusion process
            t = torch.randint(0, diffusion_steps, (microstructures.shape[0],), device=device)
            noise = torch.randn_like(latent_micro)
            
            # Add noise to latent representations
            noisy_latent = model.diffusion_model.add_noise(latent_micro, noise, t)
            
            # Predict noise
            predicted_noise = model.diffusion_model(noisy_latent, t, condition=parameters)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Validation
        if hasattr(dataset, 'val_loader'):
            val_loss = _validate_model(model, dataset.val_loader, device)
            print(f"Validation Loss = {val_loss:.4f}")
    
    model.eval()
    return model


def _validate_model(model, val_loader, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            microstructures, parameters = batch
            microstructures = microstructures.to(device)
            parameters = parameters.to(device)
            
            # Encode microstructures
            latent_micro = model.encoder(microstructures.flatten(1))
            
            # Sample random timestep
            t = torch.randint(0, model.config['diffusion']['num_steps'], 
                            (microstructures.shape[0],), device=device)
            noise = torch.randn_like(latent_micro)
            
            # Add noise
            noisy_latent = model.diffusion_model.add_noise(latent_micro, noise, t)
            
            # Predict noise
            predicted_noise = model.diffusion_model(noisy_latent, t, condition=parameters)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0