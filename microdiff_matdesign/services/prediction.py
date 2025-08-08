"""Prediction service for material properties and process outcomes."""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from ..core import ProcessParameters


class PropertyType(Enum):
    """Types of material properties that can be predicted."""
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    PHYSICAL = "physical"


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predicted_value: float
    confidence: float
    uncertainty_range: Tuple[float, float]
    model_name: str
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class PropertyPrediction:
    """Container for property prediction results."""
    property_name: str
    property_type: PropertyType
    result: PredictionResult
    units: str
    description: str


class BasePropertyPredictor(ABC):
    """Base class for property prediction models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def predict(self, parameters: ProcessParameters, **kwargs) -> PredictionResult:
        """Predict property value from process parameters."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for the prediction model."""
        pass


class MechanicalPropertyPredictor(BasePropertyPredictor):
    """Predictor for mechanical properties."""
    
    def __init__(self):
        super().__init__("MechanicalPropertyPredictor")
        self.property_models = {
            'tensile_strength': self._predict_tensile_strength,
            'yield_strength': self._predict_yield_strength,
            'elongation': self._predict_elongation,
            'hardness': self._predict_hardness,
            'elastic_modulus': self._predict_elastic_modulus,
            'fatigue_strength': self._predict_fatigue_strength
        }
    
    def predict(self, parameters: ProcessParameters, property_name: str = 'tensile_strength', **kwargs) -> PredictionResult:
        """Predict mechanical property."""
        
        if property_name not in self.property_models:
            raise ValueError(f"Unknown mechanical property: {property_name}")
        
        predictor = self.property_models[property_name]
        predicted_value, uncertainty = predictor(parameters)
        
        # Calculate confidence based on parameter validity
        confidence = self._calculate_confidence(parameters, property_name)
        
        # Uncertainty range
        uncertainty_range = (
            predicted_value - uncertainty,
            predicted_value + uncertainty
        )
        
        return PredictionResult(
            predicted_value=predicted_value,
            confidence=confidence,
            uncertainty_range=uncertainty_range,
            model_name=self.model_name,
            feature_importance=self.get_feature_importance()
        )
    
    def _predict_tensile_strength(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict tensile strength and uncertainty."""
        
        # Energy density effect
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Base strength for Ti-6Al-4V
        base_strength = 900  # MPa
        
        # Energy density effect (parabolic relationship)
        if energy_density < 60:
            ed_factor = 0.8 + 0.003 * energy_density
        elif energy_density > 120:
            ed_factor = 1.1 - 0.001 * (energy_density - 120)
        else:
            ed_factor = 0.98 + 0.002 * (energy_density - 60)
        
        # Layer thickness effect (Hall-Petch relationship)
        layer_factor = 1.0 + 0.002 * (50 - params.layer_thickness)
        
        # Scan speed effect (cooling rate)
        speed_factor = 1.0 + 0.0001 * (params.scan_speed - 800)
        
        # Powder bed temperature effect
        temp_factor = 1.0 + 0.001 * (params.powder_bed_temp - 80)
        
        predicted_strength = base_strength * ed_factor * layer_factor * speed_factor * temp_factor
        
        # Uncertainty estimation
        uncertainty = 0.05 * predicted_strength  # 5% uncertainty
        
        return predicted_strength, uncertainty
    
    def _predict_yield_strength(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict yield strength and uncertainty."""
        
        tensile_strength, _ = self._predict_tensile_strength(params)
        
        # Typical yield strength is 85-90% of tensile strength for Ti-6Al-4V
        yield_strength = tensile_strength * 0.87
        
        # Energy density fine-tuning
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density > 100:
            # Higher energy density can increase yield strength ratio
            yield_strength *= 1.02
        
        uncertainty = 0.06 * yield_strength
        
        return yield_strength, uncertainty
    
    def _predict_elongation(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict elongation at break and uncertainty."""
        
        # Energy density effect (inverse relationship with strength)
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Base elongation for Ti-6Al-4V
        base_elongation = 12.0  # %
        
        # Energy density effect (optimal around 80 J/mm³)
        if 70 <= energy_density <= 90:
            ed_factor = 1.0
        else:
            ed_factor = max(0.7, 1.0 - 0.01 * abs(energy_density - 80))
        
        # Layer thickness effect (finer layers = more ductile)
        layer_factor = 1.0 + 0.01 * (40 - params.layer_thickness)
        
        # Scan speed effect (higher speed = faster cooling = less ductility)
        speed_factor = max(0.8, 1.0 - 0.0002 * (params.scan_speed - 800))
        
        predicted_elongation = base_elongation * ed_factor * layer_factor * speed_factor
        
        uncertainty = 0.15 * predicted_elongation
        
        return predicted_elongation, uncertainty
    
    def _predict_hardness(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict hardness and uncertainty."""
        
        tensile_strength, _ = self._predict_tensile_strength(params)
        
        # Empirical relationship: HV ≈ UTS/3 for many metals
        hardness_hv = tensile_strength / 3.0
        
        # Fine-tuning for process parameters
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Higher energy density can increase hardness slightly
        if energy_density > 100:
            hardness_hv *= 1.05
        
        uncertainty = 0.08 * hardness_hv
        
        return hardness_hv, uncertainty
    
    def _predict_elastic_modulus(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict elastic modulus and uncertainty."""
        
        # Elastic modulus is relatively insensitive to processing for Ti-6Al-4V
        base_modulus = 114000  # MPa
        
        # Small effects from porosity
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Porosity effect (simplified)
        if energy_density < 60:
            porosity_factor = 0.95  # Lower density
        elif energy_density > 120:
            porosity_factor = 0.98  # Some porosity from overmelting
        else:
            porosity_factor = 1.0
        
        predicted_modulus = base_modulus * porosity_factor
        
        uncertainty = 0.03 * predicted_modulus
        
        return predicted_modulus, uncertainty
    
    def _predict_fatigue_strength(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict fatigue strength and uncertainty."""
        
        tensile_strength, _ = self._predict_tensile_strength(params)
        
        # Fatigue strength typically 40-50% of tensile strength
        base_fatigue_ratio = 0.45
        
        # Surface finish effect (layer thickness affects surface roughness)
        if params.layer_thickness < 30:
            surface_factor = 1.1  # Better surface finish
        elif params.layer_thickness > 60:
            surface_factor = 0.9   # Rougher surface
        else:
            surface_factor = 1.0
        
        # Porosity effect (critical for fatigue)
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density < 70:
            porosity_factor = 0.8  # Porosity reduces fatigue strength significantly
        else:
            porosity_factor = 1.0
        
        fatigue_strength = tensile_strength * base_fatigue_ratio * surface_factor * porosity_factor
        
        uncertainty = 0.12 * fatigue_strength  # Higher uncertainty for fatigue
        
        return fatigue_strength, uncertainty
    
    def _calculate_confidence(self, params: ProcessParameters, property_name: str) -> float:
        """Calculate prediction confidence based on parameter validity."""
        
        # Energy density within optimal range
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if 60 <= energy_density <= 100:
            energy_confidence = 1.0
        else:
            energy_confidence = max(0.5, 1.0 - 0.01 * abs(energy_density - 80))
        
        # Parameter ranges
        param_confidences = []
        
        # Laser power
        if 100 <= params.laser_power <= 400:
            param_confidences.append(1.0)
        else:
            param_confidences.append(0.8)
        
        # Scan speed
        if 500 <= params.scan_speed <= 1500:
            param_confidences.append(1.0)
        else:
            param_confidences.append(0.8)
        
        # Layer thickness
        if 20 <= params.layer_thickness <= 80:
            param_confidences.append(1.0)
        else:
            param_confidences.append(0.7)
        
        # Overall confidence
        overall_confidence = energy_confidence * np.mean(param_confidences)
        
        return min(overall_confidence, 1.0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for mechanical properties."""
        
        return {
            'energy_density': 0.4,
            'laser_power': 0.25,
            'scan_speed': 0.15,
            'layer_thickness': 0.1,
            'hatch_spacing': 0.08,
            'powder_bed_temp': 0.02
        }


class ThermalPropertyPredictor(BasePropertyPredictor):
    """Predictor for thermal properties."""
    
    def __init__(self):
        super().__init__("ThermalPropertyPredictor")
        self.property_models = {
            'thermal_conductivity': self._predict_thermal_conductivity,
            'thermal_expansion': self._predict_thermal_expansion,
            'heat_capacity': self._predict_heat_capacity,
            'melting_point': self._predict_melting_point
        }
    
    def predict(self, parameters: ProcessParameters, property_name: str = 'thermal_conductivity', **kwargs) -> PredictionResult:
        """Predict thermal property."""
        
        if property_name not in self.property_models:
            raise ValueError(f"Unknown thermal property: {property_name}")
        
        predictor = self.property_models[property_name]
        predicted_value, uncertainty = predictor(parameters)
        
        confidence = self._calculate_confidence(parameters)
        uncertainty_range = (predicted_value - uncertainty, predicted_value + uncertainty)
        
        return PredictionResult(
            predicted_value=predicted_value,
            confidence=confidence,
            uncertainty_range=uncertainty_range,
            model_name=self.model_name,
            feature_importance=self.get_feature_importance()
        )
    
    def _predict_thermal_conductivity(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict thermal conductivity."""
        
        # Base thermal conductivity for Ti-6Al-4V
        base_conductivity = 7.0  # W/m·K
        
        # Porosity effect (reduces conductivity)
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density < 60:
            porosity_factor = 0.85  # Higher porosity
        else:
            porosity_factor = 0.95
        
        # Microstructure effect (slight)
        microstructure_factor = 1.0 + 0.001 * (params.scan_speed - 800)
        
        conductivity = base_conductivity * porosity_factor * microstructure_factor
        uncertainty = 0.1 * conductivity
        
        return conductivity, uncertainty
    
    def _predict_thermal_expansion(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict coefficient of thermal expansion."""
        
        # Thermal expansion is relatively insensitive to processing
        base_expansion = 8.6e-6  # /K for Ti-6Al-4V
        
        # Small microstructural effects
        microstructure_factor = 1.0 + 0.0001 * (params.layer_thickness - 40)
        
        expansion = base_expansion * microstructure_factor
        uncertainty = 0.05 * expansion
        
        return expansion, uncertainty
    
    def _predict_heat_capacity(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict specific heat capacity."""
        
        # Heat capacity is material property, minimally affected by processing
        base_heat_capacity = 526  # J/kg·K for Ti-6Al-4V
        
        uncertainty = 0.02 * base_heat_capacity
        
        return base_heat_capacity, uncertainty
    
    def _predict_melting_point(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict melting point."""
        
        # Melting point is intrinsic material property
        melting_point = 1660  # °C for Ti-6Al-4V
        
        uncertainty = 5.0  # °C
        
        return melting_point, uncertainty
    
    def _calculate_confidence(self, params: ProcessParameters) -> float:
        """Calculate confidence for thermal property predictions."""
        
        # Thermal properties are generally less sensitive to processing
        return 0.85
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for thermal properties."""
        
        return {
            'energy_density': 0.3,
            'layer_thickness': 0.2,
            'scan_speed': 0.2,
            'laser_power': 0.15,
            'hatch_spacing': 0.1,
            'powder_bed_temp': 0.05
        }


class ProcessOutcomePredictor(BasePropertyPredictor):
    """Predictor for process outcomes (density, surface quality, etc.)."""
    
    def __init__(self):
        super().__init__("ProcessOutcomePredictor")
        self.outcome_models = {
            'density': self._predict_density,
            'surface_roughness': self._predict_surface_roughness,
            'porosity': self._predict_porosity,
            'build_rate': self._predict_build_rate,
            'energy_efficiency': self._predict_energy_efficiency,
            'dimensional_accuracy': self._predict_dimensional_accuracy
        }
    
    def predict(self, parameters: ProcessParameters, outcome_name: str = 'density', **kwargs) -> PredictionResult:
        """Predict process outcome."""
        
        if outcome_name not in self.outcome_models:
            raise ValueError(f"Unknown process outcome: {outcome_name}")
        
        predictor = self.outcome_models[outcome_name]
        predicted_value, uncertainty = predictor(parameters)
        
        confidence = self._calculate_confidence(parameters, outcome_name)
        uncertainty_range = (predicted_value - uncertainty, predicted_value + uncertainty)
        
        return PredictionResult(
            predicted_value=predicted_value,
            confidence=confidence,
            uncertainty_range=uncertainty_range,
            model_name=self.model_name,
            feature_importance=self.get_feature_importance()
        )
    
    def _predict_density(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict relative density."""
        
        # Energy density-based model
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density < 40:
            density = 0.85 + 0.01 * energy_density
        elif energy_density > 120:
            density = 0.98 - 0.001 * (energy_density - 120)
        else:
            density = 0.85 + 0.0175 * (energy_density - 40)
        
        # Scan strategy effect
        aspect_ratio = params.hatch_spacing / params.layer_thickness
        if aspect_ratio > 6:
            density *= 0.98  # Slight penalty for large aspect ratios
        
        uncertainty = 0.02
        
        return min(density, 1.0), uncertainty
    
    def _predict_surface_roughness(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict surface roughness (Ra)."""
        
        # Base roughness
        base_roughness = 8.0  # μm
        
        # Layer thickness effect (primary)
        layer_effect = params.layer_thickness / 30.0
        
        # Energy density effect
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if energy_density > 100:
            energy_effect = 1.0 + 0.01 * (energy_density - 100)
        else:
            energy_effect = 1.0
        
        # Scan speed effect
        speed_effect = 1.0 + 0.0001 * (params.scan_speed - 800)
        
        roughness = base_roughness * layer_effect * energy_effect * speed_effect
        uncertainty = 0.2 * roughness
        
        return roughness, uncertainty
    
    def _predict_porosity(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict porosity percentage."""
        
        density, density_uncertainty = self._predict_density(params)
        porosity = (1.0 - density) * 100
        
        uncertainty = density_uncertainty * 100
        
        return porosity, uncertainty
    
    def _predict_build_rate(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict build rate (volume per time)."""
        
        # Volume rate calculation
        volume_rate = params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000  # mm³/s
        
        # Convert to cm³/h
        build_rate = volume_rate * 3.6  # cm³/h
        
        uncertainty = 0.05 * build_rate
        
        return build_rate, uncertainty
    
    def _predict_energy_efficiency(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict energy efficiency (0-1 scale)."""
        
        # Energy density
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        # Optimal efficiency around 70-80 J/mm³
        if 70 <= energy_density <= 80:
            efficiency = 1.0
        else:
            efficiency = max(0.3, 1.0 - 0.01 * abs(energy_density - 75))
        
        # Build rate effect
        build_rate, _ = self._predict_build_rate(params)
        
        # Higher build rate is more efficient (to a point)
        rate_factor = min(1.2, 1.0 + 0.001 * build_rate)
        efficiency *= rate_factor
        
        uncertainty = 0.1
        
        return min(efficiency, 1.0), uncertainty
    
    def _predict_dimensional_accuracy(self, params: ProcessParameters) -> Tuple[float, float]:
        """Predict dimensional accuracy (deviation in μm)."""
        
        # Base accuracy
        base_accuracy = 50.0  # μm
        
        # Layer thickness effect
        layer_factor = params.layer_thickness / 40.0
        
        # Energy density effect
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if 70 <= energy_density <= 90:
            energy_factor = 1.0
        else:
            energy_factor = 1.0 + 0.01 * abs(energy_density - 80)
        
        accuracy = base_accuracy * layer_factor * energy_factor
        uncertainty = 0.15 * accuracy
        
        return accuracy, uncertainty
    
    def _calculate_confidence(self, params: ProcessParameters, outcome_name: str) -> float:
        """Calculate confidence for process outcome predictions."""
        
        # Higher confidence for outcomes directly related to energy density
        energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
        
        if outcome_name in ['density', 'porosity']:
            if 60 <= energy_density <= 100:
                return 0.9
            else:
                return 0.7
        elif outcome_name in ['surface_roughness', 'dimensional_accuracy']:
            return 0.8
        else:
            return 0.75
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for process outcomes."""
        
        return {
            'energy_density': 0.5,
            'layer_thickness': 0.2,
            'scan_speed': 0.15,
            'laser_power': 0.1,
            'hatch_spacing': 0.04,
            'powder_bed_temp': 0.01
        }


class PredictionService:
    """Service for comprehensive property and outcome predictions."""
    
    def __init__(self):
        """Initialize prediction service."""
        self.predictors = {
            PropertyType.MECHANICAL: MechanicalPropertyPredictor(),
            PropertyType.THERMAL: ThermalPropertyPredictor(),
            PropertyType.PHYSICAL: ProcessOutcomePredictor()
        }
        
        # Property metadata
        self.property_metadata = {
            # Mechanical properties
            'tensile_strength': {'type': PropertyType.MECHANICAL, 'units': 'MPa', 'description': 'Ultimate tensile strength'},
            'yield_strength': {'type': PropertyType.MECHANICAL, 'units': 'MPa', 'description': '0.2% offset yield strength'},
            'elongation': {'type': PropertyType.MECHANICAL, 'units': '%', 'description': 'Elongation at break'},
            'hardness': {'type': PropertyType.MECHANICAL, 'units': 'HV', 'description': 'Vickers hardness'},
            'elastic_modulus': {'type': PropertyType.MECHANICAL, 'units': 'MPa', 'description': 'Elastic modulus'},
            'fatigue_strength': {'type': PropertyType.MECHANICAL, 'units': 'MPa', 'description': 'Fatigue strength (10^6 cycles)'},
            
            # Thermal properties
            'thermal_conductivity': {'type': PropertyType.THERMAL, 'units': 'W/m·K', 'description': 'Thermal conductivity'},
            'thermal_expansion': {'type': PropertyType.THERMAL, 'units': '/K', 'description': 'Coefficient of thermal expansion'},
            'heat_capacity': {'type': PropertyType.THERMAL, 'units': 'J/kg·K', 'description': 'Specific heat capacity'},
            'melting_point': {'type': PropertyType.THERMAL, 'units': '°C', 'description': 'Melting point'},
            
            # Process outcomes
            'density': {'type': PropertyType.PHYSICAL, 'units': '%', 'description': 'Relative density'},
            'surface_roughness': {'type': PropertyType.PHYSICAL, 'units': 'μm', 'description': 'Surface roughness (Ra)'},
            'porosity': {'type': PropertyType.PHYSICAL, 'units': '%', 'description': 'Porosity percentage'},
            'build_rate': {'type': PropertyType.PHYSICAL, 'units': 'cm³/h', 'description': 'Build rate'},
            'energy_efficiency': {'type': PropertyType.PHYSICAL, 'units': '-', 'description': 'Energy efficiency (0-1)'},
            'dimensional_accuracy': {'type': PropertyType.PHYSICAL, 'units': 'μm', 'description': 'Dimensional deviation'}
        }
    
    def predict_property(self, parameters: ProcessParameters, property_name: str) -> PropertyPrediction:
        """Predict a single property."""
        
        if property_name not in self.property_metadata:
            raise ValueError(f"Unknown property: {property_name}")
        
        metadata = self.property_metadata[property_name]
        property_type = metadata['type']
        
        if property_type not in self.predictors:
            raise ValueError(f"No predictor available for property type: {property_type}")
        
        predictor = self.predictors[property_type]
        result = predictor.predict(parameters, property_name=property_name)
        
        return PropertyPrediction(
            property_name=property_name,
            property_type=property_type,
            result=result,
            units=metadata['units'],
            description=metadata['description']
        )
    
    def predict_multiple_properties(self, parameters: ProcessParameters, 
                                  property_names: List[str]) -> List[PropertyPrediction]:
        """Predict multiple properties."""
        
        predictions = []
        
        for property_name in property_names:
            try:
                prediction = self.predict_property(parameters, property_name)
                predictions.append(prediction)
            except Exception as e:
                warnings.warn(f"Failed to predict {property_name}: {str(e)}")
        
        return predictions
    
    def predict_all_mechanical_properties(self, parameters: ProcessParameters) -> List[PropertyPrediction]:
        """Predict all available mechanical properties."""
        
        mechanical_properties = [name for name, meta in self.property_metadata.items() 
                               if meta['type'] == PropertyType.MECHANICAL]
        
        return self.predict_multiple_properties(parameters, mechanical_properties)
    
    def predict_all_thermal_properties(self, parameters: ProcessParameters) -> List[PropertyPrediction]:
        """Predict all available thermal properties."""
        
        thermal_properties = [name for name, meta in self.property_metadata.items() 
                            if meta['type'] == PropertyType.THERMAL]
        
        return self.predict_multiple_properties(parameters, thermal_properties)
    
    def predict_all_process_outcomes(self, parameters: ProcessParameters) -> List[PropertyPrediction]:
        """Predict all available process outcomes."""
        
        process_outcomes = [name for name, meta in self.property_metadata.items() 
                          if meta['type'] == PropertyType.PHYSICAL]
        
        return self.predict_multiple_properties(parameters, process_outcomes)
    
    def predict_comprehensive(self, parameters: ProcessParameters) -> Dict[str, List[PropertyPrediction]]:
        """Predict all available properties and outcomes."""
        
        results = {
            'mechanical': self.predict_all_mechanical_properties(parameters),
            'thermal': self.predict_all_thermal_properties(parameters),
            'process_outcomes': self.predict_all_process_outcomes(parameters)
        }
        
        return results
    
    def compare_parameter_sets(self, parameter_sets: List[ProcessParameters],
                             property_names: List[str],
                             labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare predictions across multiple parameter sets."""
        
        if labels is None:
            labels = [f"Set_{i+1}" for i in range(len(parameter_sets))]
        
        comparison_results = {}
        
        for property_name in property_names:
            property_results = {}
            
            for i, params in enumerate(parameter_sets):
                try:
                    prediction = self.predict_property(params, property_name)
                    property_results[labels[i]] = {
                        'predicted_value': prediction.result.predicted_value,
                        'confidence': prediction.result.confidence,
                        'uncertainty_range': prediction.result.uncertainty_range,
                        'units': prediction.units
                    }
                except Exception as e:
                    warnings.warn(f"Failed to predict {property_name} for {labels[i]}: {str(e)}")
            
            if property_results:
                # Statistical analysis
                values = [result['predicted_value'] for result in property_results.values()]
                confidences = [result['confidence'] for result in property_results.values()]
                
                comparison_results[property_name] = {
                    'individual_results': property_results,
                    'statistics': {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    },
                    'confidence_statistics': {
                        'mean_confidence': np.mean(confidences),
                        'min_confidence': np.min(confidences),
                        'max_confidence': np.max(confidences)
                    },
                    'ranking': self._rank_by_property(property_results, property_name)
                }
        
        return comparison_results
    
    def _rank_by_property(self, property_results: Dict[str, Dict], property_name: str) -> List[Tuple[str, float]]:
        """Rank parameter sets by predicted property value."""
        
        # Determine if higher or lower is better for this property
        higher_is_better = property_name in [
            'tensile_strength', 'yield_strength', 'elongation', 'hardness', 'elastic_modulus',
            'fatigue_strength', 'thermal_conductivity', 'density', 'build_rate', 'energy_efficiency'
        ]
        
        ranking = [(label, result['predicted_value']) for label, result in property_results.items()]
        ranking.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        return ranking
    
    def sensitivity_analysis(self, base_parameters: ProcessParameters,
                           property_name: str,
                           parameter_variations: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis for a property."""
        
        base_prediction = self.predict_property(base_parameters, property_name)
        base_value = base_prediction.result.predicted_value
        
        sensitivity_results = {}
        
        for param_name, variation_values in parameter_variations.items():
            param_results = []
            
            for variation in variation_values:
                # Create modified parameters
                modified_params = ProcessParameters(**base_parameters.to_dict())
                
                if hasattr(modified_params, param_name):
                    setattr(modified_params, param_name, variation)
                    
                    try:
                        prediction = self.predict_property(modified_params, property_name)
                        predicted_value = prediction.result.predicted_value
                        
                        # Calculate sensitivity
                        relative_change = (predicted_value - base_value) / base_value if base_value != 0 else 0
                        param_change = (variation - getattr(base_parameters, param_name)) / getattr(base_parameters, param_name)
                        
                        sensitivity = relative_change / param_change if param_change != 0 else 0
                        
                        param_results.append({
                            'parameter_value': variation,
                            'predicted_value': predicted_value,
                            'relative_change': relative_change,
                            'sensitivity': sensitivity
                        })
                        
                    except Exception as e:
                        warnings.warn(f"Sensitivity analysis failed for {param_name}={variation}: {str(e)}")
            
            if param_results:
                sensitivities = [result['sensitivity'] for result in param_results]
                
                sensitivity_results[param_name] = {
                    'results': param_results,
                    'mean_sensitivity': np.mean(np.abs(sensitivities)),
                    'max_sensitivity': np.max(np.abs(sensitivities)),
                    'sensitivity_range': np.max(sensitivities) - np.min(sensitivities)
                }
        
        # Rank parameters by sensitivity
        param_sensitivities = [(param, result['mean_sensitivity']) 
                             for param, result in sensitivity_results.items()]
        param_sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'base_value': base_value,
            'property_name': property_name,
            'parameter_sensitivities': sensitivity_results,
            'sensitivity_ranking': param_sensitivities
        }
    
    def optimization_recommendations(self, target_properties: Dict[str, float],
                                   current_parameters: ProcessParameters,
                                   constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Generate optimization recommendations to achieve target properties."""
        
        recommendations = {
            'current_predictions': {},
            'target_gaps': {},
            'parameter_adjustments': {},
            'feasibility_assessment': {}
        }
        
        # Current predictions
        for property_name, target_value in target_properties.items():
            try:
                prediction = self.predict_property(current_parameters, property_name)
                current_value = prediction.result.predicted_value
                
                recommendations['current_predictions'][property_name] = {
                    'current_value': current_value,
                    'target_value': target_value,
                    'gap': target_value - current_value,
                    'relative_gap': (target_value - current_value) / target_value if target_value != 0 else 0,
                    'confidence': prediction.result.confidence,
                    'units': prediction.units
                }
                
                # Assess feasibility
                uncertainty_range = prediction.result.uncertainty_range
                feasible = uncertainty_range[0] <= target_value <= uncertainty_range[1]
                
                recommendations['feasibility_assessment'][property_name] = {
                    'feasible_within_uncertainty': feasible,
                    'uncertainty_range': uncertainty_range,
                    'difficulty': self._assess_optimization_difficulty(property_name, target_value, current_value)
                }
                
            except Exception as e:
                warnings.warn(f"Failed to analyze {property_name}: {str(e)}")
        
        # Generate parameter adjustment recommendations
        recommendations['parameter_adjustments'] = self._generate_parameter_adjustments(
            target_properties, current_parameters, constraints
        )
        
        return recommendations
    
    def _assess_optimization_difficulty(self, property_name: str, target_value: float, current_value: float) -> str:
        """Assess difficulty of achieving target property."""
        
        relative_change = abs(target_value - current_value) / abs(current_value) if current_value != 0 else 1
        
        if relative_change < 0.05:
            return 'easy'
        elif relative_change < 0.15:
            return 'moderate'
        elif relative_change < 0.30:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def _generate_parameter_adjustments(self, target_properties: Dict[str, float],
                                      current_parameters: ProcessParameters,
                                      constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, str]:
        """Generate parameter adjustment recommendations."""
        
        adjustments = {}
        
        for property_name, target_value in target_properties.items():
            try:
                current_prediction = self.predict_property(current_parameters, property_name)
                current_value = current_prediction.result.predicted_value
                
                if current_value < target_value:
                    # Need to increase property
                    adjustments[property_name] = self._get_increase_recommendations(property_name)
                else:
                    # Need to decrease property
                    adjustments[property_name] = self._get_decrease_recommendations(property_name)
                    
            except Exception:
                adjustments[property_name] = "Unable to generate recommendations"
        
        return adjustments
    
    def _get_increase_recommendations(self, property_name: str) -> str:
        """Get recommendations to increase a property."""
        
        if property_name in ['tensile_strength', 'yield_strength', 'hardness']:
            return "Increase energy density by reducing scan speed or increasing laser power. Consider reducing layer thickness."
        elif property_name == 'elongation':
            return "Optimize energy density around 80 J/mm³. Consider reducing scan speed for better melting."
        elif property_name == 'density':
            return "Increase energy density. Reduce scan speed, increase laser power, or reduce hatch spacing."
        elif property_name == 'surface_roughness':
            return "Increase layer thickness or reduce energy density to avoid overmelting."
        elif property_name == 'build_rate':
            return "Increase scan speed or layer thickness while maintaining adequate energy density."
        else:
            return "Adjust process parameters based on property relationships."
    
    def _get_decrease_recommendations(self, property_name: str) -> str:
        """Get recommendations to decrease a property."""
        
        if property_name in ['tensile_strength', 'yield_strength', 'hardness']:
            return "Decrease energy density by increasing scan speed or reducing laser power."
        elif property_name == 'surface_roughness':
            return "Reduce layer thickness or optimize energy density to avoid overmelting."
        elif property_name == 'porosity':
            return "Increase energy density. Reduce scan speed, increase laser power, or reduce hatch spacing."
        else:
            return "Adjust process parameters based on property relationships."
    
    def get_available_properties(self) -> Dict[str, Dict[str, str]]:
        """Get list of all available properties and their metadata."""
        
        return {
            name: {
                'type': meta['type'].value,
                'units': meta['units'],
                'description': meta['description']
            }
            for name, meta in self.property_metadata.items()
        }