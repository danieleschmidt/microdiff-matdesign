"""Validation functions for inputs and outputs."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import warnings


def validate_microstructure(microstructure: np.ndarray, 
                           min_size: Tuple[int, int, int] = (32, 32, 32),
                           max_size: Tuple[int, int, int] = (512, 512, 512)) -> bool:
    """Validate microstructure input data."""
    
    if not isinstance(microstructure, np.ndarray):
        raise TypeError(f"Microstructure must be numpy array, got {type(microstructure)}")
    
    # Check dimensionality
    if microstructure.ndim != 3:
        raise ValueError(f"Microstructure must be 3D array, got {microstructure.ndim}D")
    
    # Check size constraints
    shape = microstructure.shape
    for i, (current, min_val, max_val) in enumerate(zip(shape, min_size, max_size)):
        if current < min_val:
            raise ValueError(f"Dimension {i} too small: {current} < {min_val}")
        if current > max_val:
            warnings.warn(f"Dimension {i} very large: {current} > {max_val}, this may impact performance")
    
    # Check data type
    if not np.issubdtype(microstructure.dtype, np.floating):
        warnings.warn(f"Converting microstructure from {microstructure.dtype} to float32")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(microstructure)):
        raise ValueError("Microstructure contains NaN values")
    
    if np.any(np.isinf(microstructure)):
        raise ValueError("Microstructure contains infinite values")
    
    # Check value range
    min_val, max_val = np.min(microstructure), np.max(microstructure)
    if min_val == max_val:
        warnings.warn("Microstructure has constant values - this may indicate preprocessing issues")
    
    # Check for reasonable dynamic range
    if max_val - min_val < 1e-6:
        warnings.warn("Microstructure has very small dynamic range")
    
    return True


def validate_parameters(parameters: Dict[str, Any], process: str = "laser_powder_bed_fusion") -> bool:
    """Validate process parameters for manufacturing feasibility."""
    
    # Define parameter ranges for different processes
    parameter_ranges = {
        "laser_powder_bed_fusion": {
            "laser_power": (50.0, 500.0, "W"),
            "scan_speed": (200.0, 2000.0, "mm/s"),
            "layer_thickness": (10.0, 100.0, "μm"),
            "hatch_spacing": (50.0, 300.0, "μm"),
            "powder_bed_temp": (20.0, 200.0, "°C")
        },
        "electron_beam_melting": {
            "beam_power": (100.0, 3000.0, "W"),
            "scan_speed": (100.0, 8000.0, "mm/s"),
            "layer_thickness": (20.0, 200.0, "μm"),
            "hatch_spacing": (100.0, 500.0, "μm"),
            "bed_temperature": (600.0, 1100.0, "°C")
        },
        "directed_energy_deposition": {
            "laser_power": (100.0, 2000.0, "W"),
            "feed_rate": (1.0, 20.0, "g/min"),
            "travel_speed": (100.0, 1000.0, "mm/min"),
            "layer_height": (100.0, 1000.0, "μm"),
            "powder_flow_rate": (0.5, 10.0, "g/min")
        }
    }
    
    if process not in parameter_ranges:
        warnings.warn(f"Unknown process '{process}', using default validation")
        process = "laser_powder_bed_fusion"
    
    ranges = parameter_ranges[process]
    
    # Validate each parameter
    for param_name, value in parameters.items():
        if param_name in ranges:
            min_val, max_val, unit = ranges[param_name]
            
            # Type check
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {param_name} must be numeric, got {type(value)}")
            
            # Range check
            if value < min_val:
                raise ValueError(f"{param_name} too low: {value} < {min_val} {unit}")
            
            if value > max_val:
                raise ValueError(f"{param_name} too high: {value} > {max_val} {unit}")
            
            # Physical feasibility checks
            if param_name == "layer_thickness" and "hatch_spacing" in parameters:
                hatch_spacing = parameters["hatch_spacing"]
                if value > hatch_spacing:
                    warnings.warn(f"Layer thickness ({value}) > hatch spacing ({hatch_spacing}) - unusual parameter combination")
    
    # Process-specific validation
    if process == "laser_powder_bed_fusion":
        _validate_lpbf_parameters(parameters)
    elif process == "electron_beam_melting":
        _validate_ebm_parameters(parameters)
    elif process == "directed_energy_deposition":
        _validate_ded_parameters(parameters)
    
    return True


def _validate_lpbf_parameters(params: Dict[str, Any]) -> None:
    """Specific validation for LPBF parameters."""
    
    # Energy density calculation
    if all(key in params for key in ["laser_power", "scan_speed", "hatch_spacing", "layer_thickness"]):
        power = params["laser_power"]
        speed = params["scan_speed"]
        hatch = params["hatch_spacing"]
        layer = params["layer_thickness"]
        
        # Volumetric energy density (J/mm³)
        energy_density = power / (speed * hatch * layer / 1000)  # Convert μm to mm
        
        # Typical range for Ti-6Al-4V: 40-120 J/mm³
        if energy_density < 20:
            warnings.warn(f"Low energy density: {energy_density:.1f} J/mm³ - may cause lack of fusion")
        elif energy_density > 200:
            warnings.warn(f"High energy density: {energy_density:.1f} J/mm³ - may cause overmelting")
    
    # Scan speed vs. layer thickness relationship
    if "scan_speed" in params and "layer_thickness" in params:
        speed = params["scan_speed"]
        layer = params["layer_thickness"]
        
        # High speed with thick layers can cause incomplete melting
        if speed > 1000 and layer > 50:
            warnings.warn("High scan speed with thick layers may cause incomplete melting")


def _validate_ebm_parameters(params: Dict[str, Any]) -> None:
    """Specific validation for EBM parameters."""
    
    # Beam power vs. scan speed relationship
    if "beam_power" in params and "scan_speed" in params:
        power = params["beam_power"]
        speed = params["scan_speed"]
        
        # Line energy (J/mm)
        line_energy = power / speed
        
        if line_energy < 0.1:
            warnings.warn(f"Low line energy: {line_energy:.3f} J/mm - may cause lack of fusion")
        elif line_energy > 2.0:
            warnings.warn(f"High line energy: {line_energy:.3f} J/mm - may cause overheating")
    
    # Temperature constraints
    if "bed_temperature" in params:
        temp = params["bed_temperature"]
        
        # For Ti alloys, bed temperature should be below beta transus
        if temp > 1000:
            warnings.warn(f"High bed temperature: {temp}°C - may affect phase transformation")


def _validate_ded_parameters(params: Dict[str, Any]) -> None:
    """Specific validation for DED parameters."""
    
    # Powder flow vs. travel speed relationship
    if "powder_flow_rate" in params and "travel_speed" in params:
        flow_rate = params["powder_flow_rate"]
        travel_speed = params["travel_speed"]
        
        # Powder utilization efficiency
        if flow_rate / travel_speed > 0.1:
            warnings.warn("High powder flow rate relative to travel speed - may cause powder waste")


def validate_alloy_compatibility(alloy: str, process: str) -> bool:
    """Validate alloy-process compatibility."""
    
    compatible_combinations = {
        "Ti-6Al-4V": ["laser_powder_bed_fusion", "electron_beam_melting", "directed_energy_deposition"],
        "Inconel 718": ["laser_powder_bed_fusion", "directed_energy_deposition"],
        "AlSi10Mg": ["laser_powder_bed_fusion"],
        "SS 316L": ["laser_powder_bed_fusion", "directed_energy_deposition"],
        "CoCrMo": ["laser_powder_bed_fusion", "electron_beam_melting"]
    }
    
    if alloy not in compatible_combinations:
        warnings.warn(f"Unknown alloy '{alloy}' - compatibility not verified")
        return True
    
    if process not in compatible_combinations[alloy]:
        available_processes = ", ".join(compatible_combinations[alloy])
        warnings.warn(f"Alloy '{alloy}' not typically used with '{process}'. "
                     f"Compatible processes: {available_processes}")
        return False
    
    return True


def validate_model_inputs(microstructure: np.ndarray, alloy: str, process: str) -> bool:
    """Comprehensive validation of model inputs."""
    
    # Validate microstructure
    validate_microstructure(microstructure)
    
    # Validate alloy-process compatibility
    validate_alloy_compatibility(alloy, process)
    
    # Additional consistency checks
    volume = np.prod(microstructure.shape)
    if volume > 256**3:
        warnings.warn("Large microstructure volume may require significant computational resources")
    
    return True


def validate_training_data(microstructures: List[np.ndarray], 
                          parameters: List[Dict[str, Any]],
                          process: str) -> bool:
    """Validate training dataset."""
    
    if len(microstructures) != len(parameters):
        raise ValueError(f"Mismatch between microstructures ({len(microstructures)}) "
                        f"and parameters ({len(parameters)})")
    
    if len(microstructures) < 10:
        warnings.warn("Very small training dataset - consider collecting more data")
    
    # Validate each sample
    for i, (micro, params) in enumerate(zip(microstructures, parameters)):
        try:
            validate_microstructure(micro)
            validate_parameters(params, process)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid training sample {i}: {e}")
    
    # Check parameter diversity
    if len(parameters) > 1:
        _check_parameter_diversity(parameters)
    
    return True


def _check_parameter_diversity(parameters: List[Dict[str, Any]]) -> None:
    """Check if parameter dataset has sufficient diversity."""
    
    # Extract numeric parameters
    param_arrays = {}
    
    for params in parameters:
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if key not in param_arrays:
                    param_arrays[key] = []
                param_arrays[key].append(value)
    
    # Check diversity for each parameter
    for param_name, values in param_arrays.items():
        values = np.array(values)
        
        # Check for constant values
        if np.std(values) < 1e-6:
            warnings.warn(f"Parameter '{param_name}' has very low variance in training data")
        
        # Check for reasonable coverage
        param_range = np.max(values) - np.min(values)
        if param_range < 0.1 * np.mean(values):
            warnings.warn(f"Parameter '{param_name}' has limited range in training data")


def validate_uncertainty_quantification(mean: np.ndarray, std: np.ndarray, 
                                       confidence_level: float = 0.95) -> bool:
    """Validate uncertainty quantification outputs."""
    
    if mean.shape != std.shape:
        raise ValueError(f"Mean and std shapes don't match: {mean.shape} vs {std.shape}")
    
    # Check for negative standard deviations
    if np.any(std < 0):
        raise ValueError("Standard deviations cannot be negative")
    
    # Check for unreasonably large uncertainties
    relative_uncertainty = std / (np.abs(mean) + 1e-8)
    if np.any(relative_uncertainty > 1.0):
        warnings.warn("Very large relative uncertainties detected (>100%)")
    
    # Validate confidence level
    if not 0 < confidence_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
    
    return True