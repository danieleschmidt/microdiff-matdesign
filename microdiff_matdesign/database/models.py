"""Database models for MicroDiff-MatDesign."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

import numpy as np


@dataclass
class Experiment:
    """Experiment model."""
    
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    alloy: str = "Ti-6Al-4V"
    process: str = "laser_powder_bed_fusion"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'alloy': self.alloy,
            'process': self.process,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


@dataclass
class Microstructure:
    """Microstructure model."""
    
    id: Optional[int] = None
    experiment_id: int = 0
    name: str = ""
    volume_data_path: Optional[str] = None
    voxel_size: float = 1.0  # micrometers
    dimensions: List[int] = field(default_factory=lambda: [128, 128, 128])
    acquisition_method: str = "micro_ct"
    preprocessing_applied: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'name': self.name,
            'volume_data_path': self.volume_data_path,
            'voxel_size': self.voxel_size,
            'dimensions': self.dimensions,
            'acquisition_method': self.acquisition_method,
            'preprocessing_applied': self.preprocessing_applied,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Microstructure':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def load_volume_data(self) -> Optional[np.ndarray]:
        """Load volume data from file."""
        if not self.volume_data_path:
            return None
        
        try:
            if self.volume_data_path.endswith('.npy'):
                return np.load(self.volume_data_path)
            elif self.volume_data_path.endswith('.npz'):
                data = np.load(self.volume_data_path)
                return data['volume'] if 'volume' in data else data[list(data.keys())[0]]
            else:
                # Try to load as generic binary data
                return np.fromfile(self.volume_data_path, dtype=np.float32).reshape(self.dimensions)
        except Exception as e:
            print(f"Failed to load volume data: {e}")
            return None
    
    def save_volume_data(self, volume: np.ndarray, data_directory: str = "data/volumes") -> bool:
        """Save volume data to file."""
        from pathlib import Path
        
        try:
            data_dir = Path(data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"microstructure_{self.id}_{self.name.replace(' ', '_')}.npz"
            filepath = data_dir / filename
            
            np.savez_compressed(filepath, volume=volume)
            
            self.volume_data_path = str(filepath)
            self.dimensions = list(volume.shape)
            
            return True
            
        except Exception as e:
            print(f"Failed to save volume data: {e}")
            return False


@dataclass
class ProcessParameters:
    """Process parameters model."""
    
    id: Optional[int] = None
    experiment_id: int = 0
    microstructure_id: Optional[int] = None
    laser_power: float = 200.0  # Watts
    scan_speed: float = 800.0   # mm/s
    layer_thickness: float = 30.0  # micrometers
    hatch_spacing: float = 120.0   # micrometers
    powder_bed_temp: float = 80.0  # Celsius
    atmosphere: str = "argon"
    scan_strategy: str = "alternating"
    additional_params: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'microstructure_id': self.microstructure_id,
            'laser_power': self.laser_power,
            'scan_speed': self.scan_speed,
            'layer_thickness': self.layer_thickness,
            'hatch_spacing': self.hatch_spacing,
            'powder_bed_temp': self.powder_bed_temp,
            'atmosphere': self.atmosphere,
            'scan_strategy': self.scan_strategy,
            'additional_params': self.additional_params,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessParameters':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def calculate_energy_density(self) -> float:
        """Calculate volumetric energy density (J/mm³)."""
        return self.laser_power / (
            self.scan_speed * self.hatch_spacing * self.layer_thickness / 1000
        )
    
    def calculate_line_energy(self) -> float:
        """Calculate line energy (J/mm)."""
        return self.laser_power / self.scan_speed
    
    def calculate_area_energy(self) -> float:
        """Calculate area energy (J/mm²)."""
        return self.laser_power / (self.scan_speed * self.hatch_spacing)
    
    def validate_parameters(self) -> List[str]:
        """Validate parameter ranges."""
        warnings = []
        
        # Typical ranges for Ti-6Al-4V LPBF
        if not (100 <= self.laser_power <= 400):
            warnings.append(f"Laser power ({self.laser_power} W) outside typical range (100-400 W)")
        
        if not (200 <= self.scan_speed <= 2000):
            warnings.append(f"Scan speed ({self.scan_speed} mm/s) outside typical range (200-2000 mm/s)")
        
        if not (20 <= self.layer_thickness <= 100):
            warnings.append(f"Layer thickness ({self.layer_thickness} μm) outside typical range (20-100 μm)")
        
        if not (50 <= self.hatch_spacing <= 200):
            warnings.append(f"Hatch spacing ({self.hatch_spacing} μm) outside typical range (50-200 μm)")
        
        # Energy density check
        energy_density = self.calculate_energy_density()
        if not (40 <= energy_density <= 150):
            warnings.append(f"Energy density ({energy_density:.1f} J/mm³) outside typical range (40-150 J/mm³)")
        
        return warnings


@dataclass
class MaterialProperties:
    """Material properties model."""
    
    id: Optional[int] = None
    experiment_id: int = 0
    microstructure_id: Optional[int] = None
    tensile_strength: Optional[float] = None  # MPa
    yield_strength: Optional[float] = None    # MPa
    elongation: Optional[float] = None        # %
    hardness: Optional[float] = None          # HV
    density: Optional[float] = None           # g/cm³
    elastic_modulus: Optional[float] = None   # GPa
    thermal_conductivity: Optional[float] = None  # W/m·K
    test_method: str = "standard"
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    measured_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'microstructure_id': self.microstructure_id,
            'tensile_strength': self.tensile_strength,
            'yield_strength': self.yield_strength,
            'elongation': self.elongation,
            'hardness': self.hardness,
            'density': self.density,
            'elastic_modulus': self.elastic_modulus,
            'thermal_conductivity': self.thermal_conductivity,
            'test_method': self.test_method,
            'test_conditions': self.test_conditions,
            'measured_at': self.measured_at.isoformat() if self.measured_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialProperties':
        """Create from dictionary."""
        if 'measured_at' in data and data['measured_at']:
            data['measured_at'] = datetime.fromisoformat(data['measured_at'])
        
        return cls(**data)
    
    def calculate_quality_index(self) -> float:
        """Calculate a quality index based on properties."""
        # This is a simplified quality index
        # In practice, this would be more sophisticated
        
        score = 0.0
        count = 0
        
        # Tensile strength (normalized to Ti-6Al-4V typical: 900-1200 MPa)
        if self.tensile_strength is not None:
            normalized_strength = min(self.tensile_strength / 1050, 1.2)  # Cap at 1.2
            score += normalized_strength
            count += 1
        
        # Elongation (normalized to typical: 10-15%)
        if self.elongation is not None:
            normalized_elongation = min(self.elongation / 12.5, 1.2)
            score += normalized_elongation
            count += 1
        
        # Density (normalized to Ti-6Al-4V: 4.43 g/cm³)
        if self.density is not None:
            normalized_density = min(self.density / 4.43, 1.1)
            score += normalized_density
            count += 1
        
        return score / count if count > 0 else 0.0


@dataclass
class AnalysisResult:
    """Analysis result model."""
    
    id: Optional[int] = None
    microstructure_id: int = 0
    analysis_type: str = "comprehensive"
    features: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    analysis_version: str = "1.0"
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'microstructure_id': self.microstructure_id,
            'analysis_type': self.analysis_type,
            'features': self.features,
            'quality_metrics': self.quality_metrics,
            'recommendations': self.recommendations,
            'warnings': self.warnings,
            'confidence_scores': self.confidence_scores,
            'analysis_version': self.analysis_version,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def get_overall_quality_score(self) -> float:
        """Get overall quality score."""
        return self.quality_metrics.get('overall_quality', 0.0)
    
    def get_critical_warnings(self) -> List[str]:
        """Get critical warnings."""
        critical_keywords = ['crack', 'failure', 'critical', 'severe', 'dangerous']
        
        critical_warnings = []
        for warning in self.warnings:
            if any(keyword in warning.lower() for keyword in critical_keywords):
                critical_warnings.append(warning)
        
        return critical_warnings
    
    def get_top_recommendations(self, n: int = 3) -> List[str]:
        """Get top N recommendations."""
        return self.recommendations[:n]
    
    def summary_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            'analysis_type': self.analysis_type,
            'overall_quality': self.get_overall_quality_score(),
            'feature_count': len(self.features),
            'critical_warnings': len(self.get_critical_warnings()),
            'total_recommendations': len(self.recommendations),
            'confidence': self.confidence_scores.get('overall', 0.0),
            'analysis_date': self.created_at.isoformat() if self.created_at else None
        }