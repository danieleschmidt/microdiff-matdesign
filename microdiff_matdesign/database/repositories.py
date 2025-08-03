"""Repository classes for database operations."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

from .connection import DatabaseManager
from .models import (
    Experiment, Microstructure, ProcessParameters, 
    MaterialProperties, AnalysisResult
)


logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository class."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize repository with database manager."""
        self.db = db_manager


class ExperimentRepository(BaseRepository):
    """Repository for experiment operations."""
    
    def create(self, experiment: Experiment) -> int:
        """Create a new experiment."""
        
        query = """
        INSERT INTO experiments (name, description, alloy, process, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            experiment.name,
            experiment.description,
            experiment.alloy,
            experiment.process,
            experiment.metadata
        )
        
        experiment_id = self.db.execute_insert(query, params)
        experiment.id = experiment_id
        
        logger.info(f"Created experiment {experiment_id}: {experiment.name}")
        return experiment_id
    
    def get_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID."""
        
        query = "SELECT * FROM experiments WHERE id = ?"
        results = self.db.execute_query(query, (experiment_id,))
        
        if results:
            row = results[0]
            return Experiment(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                alloy=row['alloy'],
                process=row['process'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                metadata=row['metadata'] if row['metadata'] else {}
            )
        
        return None
    
    def get_by_alloy_process(self, alloy: str, process: str) -> List[Experiment]:
        """Get experiments by alloy and process."""
        
        query = "SELECT * FROM experiments WHERE alloy = ? AND process = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (alloy, process))
        
        experiments = []
        for row in results:
            experiment = Experiment(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                alloy=row['alloy'],
                process=row['process'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                metadata=row['metadata'] if row['metadata'] else {}
            )
            experiments.append(experiment)
        
        return experiments
    
    def update(self, experiment: Experiment) -> bool:
        """Update an experiment."""
        
        query = """
        UPDATE experiments 
        SET name = ?, description = ?, alloy = ?, process = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """
        
        params = (
            experiment.name,
            experiment.description,
            experiment.alloy,
            experiment.process,
            experiment.metadata,
            experiment.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        
        if rows_affected > 0:
            logger.info(f"Updated experiment {experiment.id}")
            return True
        
        return False
    
    def delete(self, experiment_id: int) -> bool:
        """Delete an experiment and all related data."""
        
        # This would be a cascading delete in a real implementation
        # For now, we'll just delete the experiment
        
        query = "DELETE FROM experiments WHERE id = ?"
        rows_affected = self.db.execute_update(query, (experiment_id,))
        
        if rows_affected > 0:
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        
        return False
    
    def list_all(self, limit: int = 100) -> List[Experiment]:
        """List all experiments."""
        
        query = "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?"
        results = self.db.execute_query(query, (limit,))
        
        experiments = []
        for row in results:
            experiment = Experiment(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                alloy=row['alloy'],
                process=row['process'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                metadata=row['metadata'] if row['metadata'] else {}
            )
            experiments.append(experiment)
        
        return experiments


class MicrostructureRepository(BaseRepository):
    """Repository for microstructure operations."""
    
    def create(self, microstructure: Microstructure) -> int:
        """Create a new microstructure."""
        
        query = """
        INSERT INTO microstructures 
        (experiment_id, name, volume_data_path, voxel_size, dimensions, 
         acquisition_method, preprocessing_applied)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            microstructure.experiment_id,
            microstructure.name,
            microstructure.volume_data_path,
            microstructure.voxel_size,
            microstructure.dimensions,
            microstructure.acquisition_method,
            microstructure.preprocessing_applied
        )
        
        microstructure_id = self.db.execute_insert(query, params)
        microstructure.id = microstructure_id
        
        logger.info(f"Created microstructure {microstructure_id}: {microstructure.name}")
        return microstructure_id
    
    def get_by_id(self, microstructure_id: int) -> Optional[Microstructure]:
        """Get microstructure by ID."""
        
        query = "SELECT * FROM microstructures WHERE id = ?"
        results = self.db.execute_query(query, (microstructure_id,))
        
        if results:
            row = results[0]
            return Microstructure(
                id=row['id'],
                experiment_id=row['experiment_id'],
                name=row['name'],
                volume_data_path=row['volume_data_path'],
                voxel_size=row['voxel_size'],
                dimensions=row['dimensions'] if row['dimensions'] else [128, 128, 128],
                acquisition_method=row['acquisition_method'],
                preprocessing_applied=row['preprocessing_applied'] if row['preprocessing_applied'] else {},
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
        
        return None
    
    def get_by_experiment(self, experiment_id: int) -> List[Microstructure]:
        """Get microstructures by experiment ID."""
        
        query = "SELECT * FROM microstructures WHERE experiment_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        microstructures = []
        for row in results:
            microstructure = Microstructure(
                id=row['id'],
                experiment_id=row['experiment_id'],
                name=row['name'],
                volume_data_path=row['volume_data_path'],
                voxel_size=row['voxel_size'],
                dimensions=row['dimensions'] if row['dimensions'] else [128, 128, 128],
                acquisition_method=row['acquisition_method'],
                preprocessing_applied=row['preprocessing_applied'] if row['preprocessing_applied'] else {},
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            microstructures.append(microstructure)
        
        return microstructures
    
    def update(self, microstructure: Microstructure) -> bool:
        """Update a microstructure."""
        
        query = """
        UPDATE microstructures 
        SET name = ?, volume_data_path = ?, voxel_size = ?, dimensions = ?,
            acquisition_method = ?, preprocessing_applied = ?
        WHERE id = ?
        """
        
        params = (
            microstructure.name,
            microstructure.volume_data_path,
            microstructure.voxel_size,
            microstructure.dimensions,
            microstructure.acquisition_method,
            microstructure.preprocessing_applied,
            microstructure.id
        )
        
        rows_affected = self.db.execute_update(query, params)
        
        if rows_affected > 0:
            logger.info(f"Updated microstructure {microstructure.id}")
            return True
        
        return False
    
    def delete(self, microstructure_id: int) -> bool:
        """Delete a microstructure."""
        
        query = "DELETE FROM microstructures WHERE id = ?"
        rows_affected = self.db.execute_update(query, (microstructure_id,))
        
        if rows_affected > 0:
            logger.info(f"Deleted microstructure {microstructure_id}")
            return True
        
        return False


class ParametersRepository(BaseRepository):
    """Repository for process parameters operations."""
    
    def create(self, parameters: ProcessParameters) -> int:
        """Create new process parameters."""
        
        query = """
        INSERT INTO process_parameters 
        (experiment_id, microstructure_id, laser_power, scan_speed, layer_thickness,
         hatch_spacing, powder_bed_temp, atmosphere, scan_strategy, additional_params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            parameters.experiment_id,
            parameters.microstructure_id,
            parameters.laser_power,
            parameters.scan_speed,
            parameters.layer_thickness,
            parameters.hatch_spacing,
            parameters.powder_bed_temp,
            parameters.atmosphere,
            parameters.scan_strategy,
            parameters.additional_params
        )
        
        parameters_id = self.db.execute_insert(query, params)
        parameters.id = parameters_id
        
        logger.info(f"Created process parameters {parameters_id}")
        return parameters_id
    
    def get_by_id(self, parameters_id: int) -> Optional[ProcessParameters]:
        """Get process parameters by ID."""
        
        query = "SELECT * FROM process_parameters WHERE id = ?"
        results = self.db.execute_query(query, (parameters_id,))
        
        if results:
            row = results[0]
            return ProcessParameters(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                laser_power=row['laser_power'],
                scan_speed=row['scan_speed'],
                layer_thickness=row['layer_thickness'],
                hatch_spacing=row['hatch_spacing'],
                powder_bed_temp=row['powder_bed_temp'],
                atmosphere=row['atmosphere'],
                scan_strategy=row['scan_strategy'],
                additional_params=row['additional_params'] if row['additional_params'] else {},
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
        
        return None
    
    def get_by_experiment(self, experiment_id: int) -> List[ProcessParameters]:
        """Get process parameters by experiment ID."""
        
        query = "SELECT * FROM process_parameters WHERE experiment_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        parameters_list = []
        for row in results:
            parameters = ProcessParameters(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                laser_power=row['laser_power'],
                scan_speed=row['scan_speed'],
                layer_thickness=row['layer_thickness'],
                hatch_spacing=row['hatch_spacing'],
                powder_bed_temp=row['powder_bed_temp'],
                atmosphere=row['atmosphere'],
                scan_strategy=row['scan_strategy'],
                additional_params=row['additional_params'] if row['additional_params'] else {},
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            parameters_list.append(parameters)
        
        return parameters_list
    
    def get_by_energy_density_range(self, min_energy: float, max_energy: float) -> List[ProcessParameters]:
        """Get parameters by energy density range."""
        
        # We need to calculate energy density in the query
        # This is a simplified approach - in practice you might want to store calculated values
        
        query = """
        SELECT * FROM process_parameters 
        WHERE (laser_power / (scan_speed * hatch_spacing * layer_thickness / 1000)) 
        BETWEEN ? AND ?
        ORDER BY created_at DESC
        """
        
        results = self.db.execute_query(query, (min_energy, max_energy))
        
        parameters_list = []
        for row in results:
            parameters = ProcessParameters(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                laser_power=row['laser_power'],
                scan_speed=row['scan_speed'],
                layer_thickness=row['layer_thickness'],
                hatch_spacing=row['hatch_spacing'],
                powder_bed_temp=row['powder_bed_temp'],
                atmosphere=row['atmosphere'],
                scan_strategy=row['scan_strategy'],
                additional_params=row['additional_params'] if row['additional_params'] else {},
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            parameters_list.append(parameters)
        
        return parameters_list


class PropertiesRepository(BaseRepository):
    """Repository for material properties operations."""
    
    def create(self, properties: MaterialProperties) -> int:
        """Create new material properties."""
        
        query = """
        INSERT INTO material_properties 
        (experiment_id, microstructure_id, tensile_strength, yield_strength, elongation,
         hardness, density, elastic_modulus, thermal_conductivity, test_method, test_conditions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            properties.experiment_id,
            properties.microstructure_id,
            properties.tensile_strength,
            properties.yield_strength,
            properties.elongation,
            properties.hardness,
            properties.density,
            properties.elastic_modulus,
            properties.thermal_conductivity,
            properties.test_method,
            properties.test_conditions
        )
        
        properties_id = self.db.execute_insert(query, params)
        properties.id = properties_id
        
        logger.info(f"Created material properties {properties_id}")
        return properties_id
    
    def get_by_id(self, properties_id: int) -> Optional[MaterialProperties]:
        """Get material properties by ID."""
        
        query = "SELECT * FROM material_properties WHERE id = ?"
        results = self.db.execute_query(query, (properties_id,))
        
        if results:
            row = results[0]
            return MaterialProperties(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                tensile_strength=row['tensile_strength'],
                yield_strength=row['yield_strength'],
                elongation=row['elongation'],
                hardness=row['hardness'],
                density=row['density'],
                elastic_modulus=row['elastic_modulus'],
                thermal_conductivity=row['thermal_conductivity'],
                test_method=row['test_method'],
                test_conditions=row['test_conditions'] if row['test_conditions'] else {},
                measured_at=datetime.fromisoformat(row['measured_at']) if row['measured_at'] else None
            )
        
        return None
    
    def get_by_experiment(self, experiment_id: int) -> List[MaterialProperties]:
        """Get material properties by experiment ID."""
        
        query = "SELECT * FROM material_properties WHERE experiment_id = ? ORDER BY measured_at DESC"
        results = self.db.execute_query(query, (experiment_id,))
        
        properties_list = []
        for row in results:
            properties = MaterialProperties(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                tensile_strength=row['tensile_strength'],
                yield_strength=row['yield_strength'],
                elongation=row['elongation'],
                hardness=row['hardness'],
                density=row['density'],
                elastic_modulus=row['elastic_modulus'],
                thermal_conductivity=row['thermal_conductivity'],
                test_method=row['test_method'],
                test_conditions=row['test_conditions'] if row['test_conditions'] else {},
                measured_at=datetime.fromisoformat(row['measured_at']) if row['measured_at'] else None
            )
            properties_list.append(properties)
        
        return properties_list
    
    def get_by_strength_range(self, min_strength: float, max_strength: float) -> List[MaterialProperties]:
        """Get properties by tensile strength range."""
        
        query = """
        SELECT * FROM material_properties 
        WHERE tensile_strength BETWEEN ? AND ?
        ORDER BY tensile_strength DESC
        """
        
        results = self.db.execute_query(query, (min_strength, max_strength))
        
        properties_list = []
        for row in results:
            properties = MaterialProperties(
                id=row['id'],
                experiment_id=row['experiment_id'],
                microstructure_id=row['microstructure_id'],
                tensile_strength=row['tensile_strength'],
                yield_strength=row['yield_strength'],
                elongation=row['elongation'],
                hardness=row['hardness'],
                density=row['density'],
                elastic_modulus=row['elastic_modulus'],
                thermal_conductivity=row['thermal_conductivity'],
                test_method=row['test_method'],
                test_conditions=row['test_conditions'] if row['test_conditions'] else {},
                measured_at=datetime.fromisoformat(row['measured_at']) if row['measured_at'] else None
            )
            properties_list.append(properties)
        
        return properties_list


class AnalysisRepository(BaseRepository):
    """Repository for analysis results operations."""
    
    def create(self, analysis: AnalysisResult) -> int:
        """Create new analysis result."""
        
        query = """
        INSERT INTO analysis_results 
        (microstructure_id, analysis_type, features, quality_metrics, recommendations,
         warnings, confidence_scores, analysis_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            analysis.microstructure_id,
            analysis.analysis_type,
            analysis.features,
            analysis.quality_metrics,
            analysis.recommendations,
            analysis.warnings,
            analysis.confidence_scores,
            analysis.analysis_version
        )
        
        analysis_id = self.db.execute_insert(query, params)
        analysis.id = analysis_id
        
        logger.info(f"Created analysis result {analysis_id} for microstructure {analysis.microstructure_id}")
        return analysis_id
    
    def get_by_id(self, analysis_id: int) -> Optional[AnalysisResult]:
        """Get analysis result by ID."""
        
        query = "SELECT * FROM analysis_results WHERE id = ?"
        results = self.db.execute_query(query, (analysis_id,))
        
        if results:
            row = results[0]
            return AnalysisResult(
                id=row['id'],
                microstructure_id=row['microstructure_id'],
                analysis_type=row['analysis_type'],
                features=row['features'] if row['features'] else {},
                quality_metrics=row['quality_metrics'] if row['quality_metrics'] else {},
                recommendations=row['recommendations'] if row['recommendations'] else [],
                warnings=row['warnings'] if row['warnings'] else [],
                confidence_scores=row['confidence_scores'] if row['confidence_scores'] else {},
                analysis_version=row['analysis_version'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
        
        return None
    
    def get_by_microstructure(self, microstructure_id: int) -> List[AnalysisResult]:
        """Get analysis results by microstructure ID."""
        
        query = "SELECT * FROM analysis_results WHERE microstructure_id = ? ORDER BY created_at DESC"
        results = self.db.execute_query(query, (microstructure_id,))
        
        analyses = []
        for row in results:
            analysis = AnalysisResult(
                id=row['id'],
                microstructure_id=row['microstructure_id'],
                analysis_type=row['analysis_type'],
                features=row['features'] if row['features'] else {},
                quality_metrics=row['quality_metrics'] if row['quality_metrics'] else {},
                recommendations=row['recommendations'] if row['recommendations'] else [],
                warnings=row['warnings'] if row['warnings'] else [],
                confidence_scores=row['confidence_scores'] if row['confidence_scores'] else {},
                analysis_version=row['analysis_version'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            analyses.append(analysis)
        
        return analyses
    
    def get_latest_by_microstructure(self, microstructure_id: int, analysis_type: str = None) -> Optional[AnalysisResult]:
        """Get latest analysis result for a microstructure."""
        
        if analysis_type:
            query = """
            SELECT * FROM analysis_results 
            WHERE microstructure_id = ? AND analysis_type = ?
            ORDER BY created_at DESC LIMIT 1
            """
            results = self.db.execute_query(query, (microstructure_id, analysis_type))
        else:
            query = """
            SELECT * FROM analysis_results 
            WHERE microstructure_id = ?
            ORDER BY created_at DESC LIMIT 1
            """
            results = self.db.execute_query(query, (microstructure_id,))
        
        if results:
            row = results[0]
            return AnalysisResult(
                id=row['id'],
                microstructure_id=row['microstructure_id'],
                analysis_type=row['analysis_type'],
                features=row['features'] if row['features'] else {},
                quality_metrics=row['quality_metrics'] if row['quality_metrics'] else {},
                recommendations=row['recommendations'] if row['recommendations'] else [],
                warnings=row['warnings'] if row['warnings'] else [],
                confidence_scores=row['confidence_scores'] if row['confidence_scores'] else {},
                analysis_version=row['analysis_version'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
        
        return None
    
    def get_by_quality_range(self, min_quality: float, max_quality: float) -> List[AnalysisResult]:
        """Get analysis results by quality score range."""
        
        # This is a bit complex because quality_metrics is JSON
        # In SQLite, we can use JSON functions
        
        query = """
        SELECT * FROM analysis_results 
        WHERE json_extract(quality_metrics, '$.overall_quality') BETWEEN ? AND ?
        ORDER BY json_extract(quality_metrics, '$.overall_quality') DESC
        """
        
        results = self.db.execute_query(query, (min_quality, max_quality))
        
        analyses = []
        for row in results:
            analysis = AnalysisResult(
                id=row['id'],
                microstructure_id=row['microstructure_id'],
                analysis_type=row['analysis_type'],
                features=row['features'] if row['features'] else {},
                quality_metrics=row['quality_metrics'] if row['quality_metrics'] else {},
                recommendations=row['recommendations'] if row['recommendations'] else [],
                warnings=row['warnings'] if row['warnings'] else [],
                confidence_scores=row['confidence_scores'] if row['confidence_scores'] else {},
                analysis_version=row['analysis_version'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            analyses.append(analysis)
        
        return analyses