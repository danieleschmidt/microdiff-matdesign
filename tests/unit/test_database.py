"""Unit tests for database functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from microdiff_matdesign.database import (
    DatabaseManager, 
    Experiment, 
    Microstructure, 
    ProcessParameters,
    MaterialProperties,
    AnalysisResult,
    ExperimentRepository,
    MicrostructureRepository,
    ParametersRepository,
    PropertiesRepository,
    AnalysisRepository
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    db_manager = DatabaseManager(db_path)
    yield db_manager
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def sample_experiment():
    """Create sample experiment for testing."""
    return Experiment(
        name="Test Experiment",
        description="A test experiment for unit testing",
        alloy="Ti-6Al-4V",
        process="laser_powder_bed_fusion",
        metadata={"test": True, "version": "1.0"}
    )


@pytest.fixture
def sample_microstructure():
    """Create sample microstructure for testing."""
    return Microstructure(
        experiment_id=1,
        name="Test Microstructure",
        voxel_size=0.5,
        dimensions=[128, 128, 128],
        acquisition_method="micro_ct",
        preprocessing_applied={"denoising": True, "normalization": True}
    )


@pytest.fixture
def sample_parameters():
    """Create sample process parameters for testing."""
    return ProcessParameters(
        experiment_id=1,
        laser_power=200.0,
        scan_speed=800.0,
        layer_thickness=30.0,
        hatch_spacing=120.0,
        powder_bed_temp=80.0,
        atmosphere="argon",
        scan_strategy="alternating",
        additional_params={"rotation": 67}
    )


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_database_initialization(self, temp_db):
        """Test database initialization."""
        assert temp_db.database_path.exists()
        
        # Check that tables exist
        stats = temp_db.get_database_stats()
        assert 'experiments_count' in stats
        assert stats['experiments_count'] == 0
    
    def test_database_stats(self, temp_db):
        """Test database statistics."""
        stats = temp_db.get_database_stats()
        
        assert 'database_size_bytes' in stats
        assert 'database_size_mb' in stats
        assert stats['database_size_bytes'] > 0
        assert stats['database_size_mb'] > 0
    
    def test_backup_restore(self, temp_db):
        """Test database backup and restore."""
        # Create backup
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            backup_path = tmp.name
        
        try:
            success = temp_db.backup_database(backup_path)
            assert success
            assert Path(backup_path).exists()
            
            # Test restore (this would require a new database instance in practice)
            success = temp_db.restore_database(backup_path)
            assert success
        
        finally:
            if Path(backup_path).exists():
                os.unlink(backup_path)
    
    def test_vacuum_database(self, temp_db):
        """Test database vacuum operation."""
        success = temp_db.vacuum_database()
        assert success


class TestExperimentRepository:
    """Test experiment repository functionality."""
    
    def test_create_experiment(self, temp_db, sample_experiment):
        """Test creating an experiment."""
        repo = ExperimentRepository(temp_db)
        
        experiment_id = repo.create(sample_experiment)
        assert experiment_id > 0
        assert sample_experiment.id == experiment_id
    
    def test_get_experiment_by_id(self, temp_db, sample_experiment):
        """Test retrieving experiment by ID."""
        repo = ExperimentRepository(temp_db)
        
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Retrieve experiment
        retrieved = repo.get_by_id(experiment_id)
        assert retrieved is not None
        assert retrieved.name == sample_experiment.name
        assert retrieved.alloy == sample_experiment.alloy
        assert retrieved.metadata == sample_experiment.metadata
    
    def test_get_experiments_by_alloy_process(self, temp_db, sample_experiment):
        """Test retrieving experiments by alloy and process."""
        repo = ExperimentRepository(temp_db)
        
        # Create multiple experiments
        repo.create(sample_experiment)
        
        experiment2 = Experiment(
            name="Test Experiment 2",
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion"
        )
        repo.create(experiment2)
        
        experiment3 = Experiment(
            name="Test Experiment 3",
            alloy="Inconel 718",
            process="laser_powder_bed_fusion"
        )
        repo.create(experiment3)
        
        # Retrieve by alloy and process
        ti_experiments = repo.get_by_alloy_process("Ti-6Al-4V", "laser_powder_bed_fusion")
        assert len(ti_experiments) == 2
        
        inconel_experiments = repo.get_by_alloy_process("Inconel 718", "laser_powder_bed_fusion")
        assert len(inconel_experiments) == 1
    
    def test_update_experiment(self, temp_db, sample_experiment):
        """Test updating an experiment."""
        repo = ExperimentRepository(temp_db)
        
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Update experiment
        sample_experiment.description = "Updated description"
        sample_experiment.metadata["updated"] = True
        
        success = repo.update(sample_experiment)
        assert success
        
        # Verify update
        updated = repo.get_by_id(experiment_id)
        assert updated.description == "Updated description"
        assert updated.metadata["updated"] is True
    
    def test_delete_experiment(self, temp_db, sample_experiment):
        """Test deleting an experiment."""
        repo = ExperimentRepository(temp_db)
        
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Delete experiment
        success = repo.delete(experiment_id)
        assert success
        
        # Verify deletion
        deleted = repo.get_by_id(experiment_id)
        assert deleted is None
    
    def test_list_all_experiments(self, temp_db, sample_experiment):
        """Test listing all experiments."""
        repo = ExperimentRepository(temp_db)
        
        # Create multiple experiments
        for i in range(5):
            exp = Experiment(
                name=f"Experiment {i}",
                alloy="Ti-6Al-4V",
                process="laser_powder_bed_fusion"
            )
            repo.create(exp)
        
        # List all
        experiments = repo.list_all()
        assert len(experiments) == 5
        
        # Test limit
        limited = repo.list_all(limit=3)
        assert len(limited) == 3


class TestMicrostructureRepository:
    """Test microstructure repository functionality."""
    
    def test_create_microstructure(self, temp_db, sample_experiment, sample_microstructure):
        """Test creating a microstructure."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        
        # Create experiment first
        exp_id = exp_repo.create(sample_experiment)
        sample_microstructure.experiment_id = exp_id
        
        # Create microstructure
        micro_id = micro_repo.create(sample_microstructure)
        assert micro_id > 0
        assert sample_microstructure.id == micro_id
    
    def test_get_microstructure_by_id(self, temp_db, sample_experiment, sample_microstructure):
        """Test retrieving microstructure by ID."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        
        # Create experiment and microstructure
        exp_id = exp_repo.create(sample_experiment)
        sample_microstructure.experiment_id = exp_id
        micro_id = micro_repo.create(sample_microstructure)
        
        # Retrieve microstructure
        retrieved = micro_repo.get_by_id(micro_id)
        assert retrieved is not None
        assert retrieved.name == sample_microstructure.name
        assert retrieved.voxel_size == sample_microstructure.voxel_size
        assert retrieved.dimensions == sample_microstructure.dimensions
    
    def test_get_microstructures_by_experiment(self, temp_db, sample_experiment, sample_microstructure):
        """Test retrieving microstructures by experiment."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        
        # Create experiment
        exp_id = exp_repo.create(sample_experiment)
        
        # Create multiple microstructures
        for i in range(3):
            micro = Microstructure(
                experiment_id=exp_id,
                name=f"Microstructure {i}",
                voxel_size=0.5 + i * 0.1,
                dimensions=[128, 128, 128]
            )
            micro_repo.create(micro)
        
        # Retrieve by experiment
        microstructures = micro_repo.get_by_experiment(exp_id)
        assert len(microstructures) == 3


class TestProcessParametersRepository:
    """Test process parameters repository functionality."""
    
    def test_create_parameters(self, temp_db, sample_experiment, sample_parameters):
        """Test creating process parameters."""
        exp_repo = ExperimentRepository(temp_db)
        param_repo = ParametersRepository(temp_db)
        
        # Create experiment first
        exp_id = exp_repo.create(sample_experiment)
        sample_parameters.experiment_id = exp_id
        
        # Create parameters
        param_id = param_repo.create(sample_parameters)
        assert param_id > 0
        assert sample_parameters.id == param_id
    
    def test_get_parameters_by_id(self, temp_db, sample_experiment, sample_parameters):
        """Test retrieving parameters by ID."""
        exp_repo = ExperimentRepository(temp_db)
        param_repo = ParametersRepository(temp_db)
        
        # Create experiment and parameters
        exp_id = exp_repo.create(sample_experiment)
        sample_parameters.experiment_id = exp_id
        param_id = param_repo.create(sample_parameters)
        
        # Retrieve parameters
        retrieved = param_repo.get_by_id(param_id)
        assert retrieved is not None
        assert retrieved.laser_power == sample_parameters.laser_power
        assert retrieved.scan_speed == sample_parameters.scan_speed
        assert retrieved.additional_params == sample_parameters.additional_params
    
    def test_get_parameters_by_energy_density_range(self, temp_db, sample_experiment):
        """Test retrieving parameters by energy density range."""
        exp_repo = ExperimentRepository(temp_db)
        param_repo = ParametersRepository(temp_db)
        
        # Create experiment
        exp_id = exp_repo.create(sample_experiment)
        
        # Create parameters with different energy densities
        param_sets = [
            (200, 800, 30, 120),  # ~69 J/mm³
            (300, 600, 40, 100),  # ~125 J/mm³
            (150, 1000, 25, 150)  # ~40 J/mm³
        ]
        
        for power, speed, thickness, spacing in param_sets:
            params = ProcessParameters(
                experiment_id=exp_id,
                laser_power=power,
                scan_speed=speed,
                layer_thickness=thickness,
                hatch_spacing=spacing
            )
            param_repo.create(params)
        
        # Test range query
        mid_range_params = param_repo.get_by_energy_density_range(60, 80)
        assert len(mid_range_params) == 1
        
        high_range_params = param_repo.get_by_energy_density_range(100, 150)
        assert len(high_range_params) == 1


class TestMaterialPropertiesRepository:
    """Test material properties repository functionality."""
    
    def test_create_properties(self, temp_db, sample_experiment):
        """Test creating material properties."""
        exp_repo = ExperimentRepository(temp_db)
        prop_repo = PropertiesRepository(temp_db)
        
        # Create experiment
        exp_id = exp_repo.create(sample_experiment)
        
        # Create properties
        properties = MaterialProperties(
            experiment_id=exp_id,
            tensile_strength=950.0,
            yield_strength=850.0,
            elongation=12.5,
            hardness=320.0,
            density=4.42,
            test_method="ASTM_E8",
            test_conditions={"temperature": 25, "humidity": 45}
        )
        
        prop_id = prop_repo.create(properties)
        assert prop_id > 0
        assert properties.id == prop_id
    
    def test_get_properties_by_strength_range(self, temp_db, sample_experiment):
        """Test retrieving properties by strength range."""
        exp_repo = ExperimentRepository(temp_db)
        prop_repo = PropertiesRepository(temp_db)
        
        # Create experiment
        exp_id = exp_repo.create(sample_experiment)
        
        # Create properties with different strengths
        strengths = [800, 950, 1100, 1250]
        for strength in strengths:
            properties = MaterialProperties(
                experiment_id=exp_id,
                tensile_strength=strength,
                yield_strength=strength * 0.9
            )
            prop_repo.create(properties)
        
        # Test range query
        mid_strength = prop_repo.get_by_strength_range(900, 1150)
        assert len(mid_strength) == 2


class TestAnalysisResultRepository:
    """Test analysis result repository functionality."""
    
    def test_create_analysis_result(self, temp_db, sample_experiment, sample_microstructure):
        """Test creating analysis result."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        analysis_repo = AnalysisRepository(temp_db)
        
        # Create experiment and microstructure
        exp_id = exp_repo.create(sample_experiment)
        sample_microstructure.experiment_id = exp_id
        micro_id = micro_repo.create(sample_microstructure)
        
        # Create analysis result
        analysis = AnalysisResult(
            microstructure_id=micro_id,
            analysis_type="comprehensive",
            features={
                "porosity": 2.5,
                "grain_size": 45.2,
                "surface_roughness": 8.3
            },
            quality_metrics={
                "overall_quality": 0.85,
                "density_quality": 0.92,
                "defect_severity": 0.78
            },
            recommendations=[
                "Consider reducing scan speed for better density",
                "Optimize powder bed temperature"
            ],
            warnings=["Minor porosity detected"],
            confidence_scores={"overall": 0.9, "feature_extraction": 0.95},
            analysis_version="1.0"
        )
        
        analysis_id = analysis_repo.create(analysis)
        assert analysis_id > 0
        assert analysis.id == analysis_id
    
    def test_get_analysis_by_quality_range(self, temp_db, sample_experiment, sample_microstructure):
        """Test retrieving analysis by quality range."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        analysis_repo = AnalysisRepository(temp_db)
        
        # Create experiment and microstructure
        exp_id = exp_repo.create(sample_experiment)
        sample_microstructure.experiment_id = exp_id
        micro_id = micro_repo.create(sample_microstructure)
        
        # Create analysis results with different quality scores
        quality_scores = [0.6, 0.75, 0.85, 0.95]
        for quality in quality_scores:
            analysis = AnalysisResult(
                microstructure_id=micro_id,
                analysis_type="quality_assessment",
                quality_metrics={"overall_quality": quality}
            )
            analysis_repo.create(analysis)
        
        # Test range query
        high_quality = analysis_repo.get_by_quality_range(0.8, 1.0)
        assert len(high_quality) == 2
    
    def test_get_latest_analysis(self, temp_db, sample_experiment, sample_microstructure):
        """Test retrieving latest analysis for microstructure."""
        exp_repo = ExperimentRepository(temp_db)
        micro_repo = MicrostructureRepository(temp_db)
        analysis_repo = AnalysisRepository(temp_db)
        
        # Create experiment and microstructure
        exp_id = exp_repo.create(sample_experiment)
        sample_microstructure.experiment_id = exp_id
        micro_id = micro_repo.create(sample_microstructure)
        
        # Create multiple analysis results
        for i in range(3):
            analysis = AnalysisResult(
                microstructure_id=micro_id,
                analysis_type="comprehensive",
                quality_metrics={"overall_quality": 0.7 + i * 0.1}
            )
            analysis_repo.create(analysis)
        
        # Get latest
        latest = analysis_repo.get_latest_by_microstructure(micro_id)
        assert latest is not None
        assert latest.quality_metrics["overall_quality"] == 0.9  # Last one created


class TestModelIntegration:
    """Test model functionality integration."""
    
    def test_experiment_to_dict_from_dict(self, sample_experiment):
        """Test experiment serialization."""
        # Convert to dict
        exp_dict = sample_experiment.to_dict()
        assert exp_dict['name'] == sample_experiment.name
        assert exp_dict['metadata'] == sample_experiment.metadata
        
        # Convert back from dict
        exp_restored = Experiment.from_dict(exp_dict)
        assert exp_restored.name == sample_experiment.name
        assert exp_restored.metadata == sample_experiment.metadata
    
    def test_process_parameters_calculations(self, sample_parameters):
        """Test process parameter calculations."""
        # Test energy density calculation
        energy_density = sample_parameters.calculate_energy_density()
        expected = 200.0 / (800.0 * 120.0 * 30.0 / 1000)
        assert abs(energy_density - expected) < 0.01
        
        # Test line energy
        line_energy = sample_parameters.calculate_line_energy()
        expected_line = 200.0 / 800.0
        assert abs(line_energy - expected_line) < 0.01
        
        # Test area energy
        area_energy = sample_parameters.calculate_area_energy()
        expected_area = 200.0 / (800.0 * 120.0)
        assert abs(area_energy - expected_area) < 0.01
    
    def test_process_parameters_validation(self, sample_parameters):
        """Test process parameter validation."""
        warnings = sample_parameters.validate_parameters()
        assert isinstance(warnings, list)
        
        # Test with extreme values
        extreme_params = ProcessParameters(
            laser_power=50,  # Too low
            scan_speed=3000,  # Too high
            layer_thickness=150,  # Too high
            hatch_spacing=25,  # Too low
            powder_bed_temp=80
        )
        
        warnings = extreme_params.validate_parameters()
        assert len(warnings) > 0
        assert any("Laser power" in warning for warning in warnings)
        assert any("Scan speed" in warning for warning in warnings)
    
    def test_material_properties_quality_index(self):
        """Test material properties quality index calculation."""
        properties = MaterialProperties(
            tensile_strength=1000.0,
            elongation=10.0,
            density=4.40
        )
        
        quality_index = properties.calculate_quality_index()
        assert 0.0 <= quality_index <= 1.5  # Should be in reasonable range
        
        # Test with no properties
        empty_properties = MaterialProperties()
        quality_index = empty_properties.calculate_quality_index()
        assert quality_index == 0.0
    
    def test_analysis_result_methods(self):
        """Test analysis result helper methods."""
        analysis = AnalysisResult(
            quality_metrics={"overall_quality": 0.85},
            warnings=["Critical crack detected", "Minor porosity", "Severe delamination"],
            recommendations=["Rec 1", "Rec 2", "Rec 3", "Rec 4", "Rec 5"]
        )
        
        # Test overall quality
        assert analysis.get_overall_quality_score() == 0.85
        
        # Test critical warnings
        critical = analysis.get_critical_warnings()
        assert len(critical) == 2  # "Critical" and "Severe"
        
        # Test top recommendations
        top_recs = analysis.get_top_recommendations(3)
        assert len(top_recs) == 3
        
        # Test summary report
        summary = analysis.summary_report()
        assert summary['overall_quality'] == 0.85
        assert summary['critical_warnings'] == 2
        assert summary['total_recommendations'] == 5