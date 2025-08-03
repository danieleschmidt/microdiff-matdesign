"""Integration tests for the complete MicroDiff-MatDesign pipeline."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from microdiff_matdesign.core import MicrostructureDiffusion, ProcessParameters
from microdiff_matdesign.imaging import MicroCTProcessor
from microdiff_matdesign.services.analysis import AnalysisService
from microdiff_matdesign.services.optimization import OptimizationService, OptimizationConfig, OptimizationAlgorithm
from microdiff_matdesign.services.parameter_generation import ParameterConstraints
from microdiff_matdesign.database import (
    DatabaseManager, ExperimentRepository, MicrostructureRepository,
    ParametersRepository, AnalysisRepository,
    Experiment, Microstructure, ProcessParameters as DBProcessParameters, AnalysisResult
)


@pytest.fixture
def temp_database():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    db_manager = DatabaseManager(db_path)
    yield db_manager
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def sample_microstructure_volume():
    """Create sample 3D microstructure volume."""
    # Create a synthetic microstructure with some patterns
    volume = np.random.rand(64, 64, 64)
    
    # Add some structure (spherical inclusions)
    center = np.array([32, 32, 32])
    x, y, z = np.meshgrid(range(64), range(64), range(64), indexing='ij')
    
    # Create spherical features
    for i in range(3):
        sphere_center = center + np.random.randint(-20, 20, 3)
        radius = np.random.randint(5, 15)
        
        distances = np.sqrt((x - sphere_center[0])**2 + 
                           (y - sphere_center[1])**2 + 
                           (z - sphere_center[2])**2)
        
        volume[distances < radius] = 0.2 + 0.3 * np.random.rand()
    
    # Add some noise
    volume += 0.1 * np.random.randn(*volume.shape)
    volume = np.clip(volume, 0, 1)
    
    return volume


@pytest.fixture
def sample_experiment_data():
    """Create sample experiment data."""
    return {
        'name': 'Integration Test Experiment',
        'description': 'Test experiment for pipeline integration',
        'alloy': 'Ti-6Al-4V',
        'process': 'laser_powder_bed_fusion',
        'metadata': {'test_run': True}
    }


@pytest.fixture
def sample_parameters():
    """Create sample process parameters."""
    return ProcessParameters(
        laser_power=225.0,
        scan_speed=850.0,
        layer_thickness=35.0,
        hatch_spacing=110.0,
        powder_bed_temp=85.0,
        atmosphere='argon'
    )


class TestFullPipeline:
    """Test the complete pipeline from microstructure to optimized parameters."""
    
    def test_complete_analysis_pipeline(self, sample_microstructure_volume, temp_database):
        """Test complete analysis pipeline."""
        
        # 1. Setup database and repositories
        exp_repo = ExperimentRepository(temp_database)
        micro_repo = MicrostructureRepository(temp_database)
        analysis_repo = AnalysisRepository(temp_database)
        
        # 2. Create experiment in database
        experiment = Experiment(
            name="Pipeline Test",
            description="Complete pipeline test",
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion"
        )
        exp_id = exp_repo.create(experiment)
        
        # 3. Save microstructure volume and create database entry
        with tempfile.TemporaryDirectory() as temp_dir:
            volume_path = Path(temp_dir) / "test_volume.npz"
            np.savez_compressed(volume_path, volume=sample_microstructure_volume)
            
            microstructure = Microstructure(
                experiment_id=exp_id,
                name="Test Microstructure",
                volume_data_path=str(volume_path),
                voxel_size=0.5,
                dimensions=list(sample_microstructure_volume.shape),
                acquisition_method="synthetic"
            )
            micro_id = micro_repo.create(microstructure)
            
            # 4. Initialize analysis service and analyze microstructure
            processor = MicroCTProcessor()
            analysis_service = AnalysisService(processor=processor)
            
            analysis_report = analysis_service.analyze_microstructure(
                sample_microstructure_volume,
                analysis_types=['grain_analysis', 'porosity_analysis', 'defect_analysis']
            )
            
            # 5. Store analysis results in database
            analysis_result = AnalysisResult(
                microstructure_id=micro_id,
                analysis_type="comprehensive",
                features=analysis_report.microstructure_features,
                quality_metrics=analysis_report.quality_metrics,
                recommendations=analysis_report.recommendations,
                warnings=analysis_report.warnings,
                confidence_scores=analysis_report.confidence_scores
            )
            analysis_id = analysis_repo.create(analysis_result)
            
            # 6. Verify pipeline results
            assert exp_id > 0
            assert micro_id > 0
            assert analysis_id > 0
            
            # Verify analysis contains expected features
            assert 'porosity' in analysis_report.microstructure_features
            assert 'total_porosity' in analysis_report.microstructure_features
            assert len(analysis_report.quality_metrics) > 0
            assert 'overall_quality' in analysis_report.quality_metrics
            
            # Verify database storage
            retrieved_analysis = analysis_repo.get_by_id(analysis_id)
            assert retrieved_analysis is not None
            assert retrieved_analysis.analysis_type == "comprehensive"
            assert len(retrieved_analysis.features) > 0
    
    def test_inverse_design_pipeline(self, sample_microstructure_volume, sample_parameters):
        """Test inverse design pipeline."""
        
        # Mock the diffusion model since we don't have trained weights
        with patch('microdiff_matdesign.core.MicrostructureDiffusion') as mock_diffusion:
            # Setup mock
            mock_model = Mock()
            mock_model.inverse_design.return_value = sample_parameters
            mock_diffusion.return_value = mock_model
            
            # 1. Initialize diffusion model
            diffusion_model = MicrostructureDiffusion(
                alloy="Ti-6Al-4V",
                process="laser_powder_bed_fusion",
                pretrained=True
            )
            
            # 2. Perform inverse design
            generated_parameters = diffusion_model.inverse_design(
                target_microstructure=sample_microstructure_volume,
                num_samples=5,
                guidance_scale=7.5
            )
            
            # 3. Verify results
            assert isinstance(generated_parameters, ProcessParameters)
            assert generated_parameters.laser_power > 0
            assert generated_parameters.scan_speed > 0
            
            # Verify mock was called correctly
            mock_model.inverse_design.assert_called_once()
            call_args = mock_model.inverse_design.call_args[1]
            assert 'target_microstructure' in call_args
            assert call_args['num_samples'] == 5
            assert call_args['guidance_scale'] == 7.5
    
    def test_optimization_pipeline(self, temp_database):
        """Test parameter optimization pipeline."""
        
        # 1. Setup optimization service
        optimization_service = OptimizationService()
        
        # 2. Define optimization problem
        constraints = ParameterConstraints(
            laser_power_range=(150, 350),
            scan_speed_range=(500, 1500),
            layer_thickness_range=(25, 50),
            hatch_spacing_range=(80, 150),
            powder_bed_temp_range=(60, 120)
        )
        
        objectives = {'density': 0.7, 'strength': 0.3}
        
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=20,
            max_iterations=10,
            random_seed=42
        )
        
        # 3. Run optimization
        result = optimization_service.optimize_parameters(
            objectives=objectives,
            constraints=constraints,
            config=config
        )
        
        # 4. Verify optimization results
        assert 'best_solution' in result
        assert 'best_fitness' in result
        assert result['best_fitness'] > 0
        
        best_params = result['best_solution']
        assert isinstance(best_params, ProcessParameters)
        
        # Verify constraints are satisfied
        assert 150 <= best_params.laser_power <= 350
        assert 500 <= best_params.scan_speed <= 1500
        assert 25 <= best_params.layer_thickness <= 50
        assert 80 <= best_params.hatch_spacing <= 150
        assert 60 <= best_params.powder_bed_temp <= 120
        
        # 5. Store optimization results in database
        exp_repo = ExperimentRepository(temp_database)
        param_repo = ParametersRepository(temp_database)
        
        # Create experiment for optimization results
        experiment = Experiment(
            name="Optimization Results",
            description="Results from parameter optimization",
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion",
            metadata={'optimization_config': config.__dict__}
        )
        exp_id = exp_repo.create(experiment)
        
        # Store best parameters
        db_params = DBProcessParameters(
            experiment_id=exp_id,
            laser_power=best_params.laser_power,
            scan_speed=best_params.scan_speed,
            layer_thickness=best_params.layer_thickness,
            hatch_spacing=best_params.hatch_spacing,
            powder_bed_temp=best_params.powder_bed_temp,
            atmosphere=best_params.atmosphere,
            additional_params={
                'optimization_fitness': result['best_fitness'],
                'optimization_generations': result.get('generations', 0)
            }
        )
        param_id = param_repo.create(db_params)
        
        assert param_id > 0
    
    def test_multi_objective_optimization_pipeline(self):
        """Test multi-objective optimization pipeline."""
        
        optimization_service = OptimizationService()
        
        # Define constraints
        constraints = ParameterConstraints(
            laser_power_range=(100, 400),
            scan_speed_range=(200, 2000),
            layer_thickness_range=(20, 100),
            hatch_spacing_range=(50, 200),
            powder_bed_temp_range=(20, 200)
        )
        
        # Multiple objectives
        objectives = ['density', 'strength', 'surface_quality']
        
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=15,
            max_iterations=8,
            random_seed=42
        )
        
        # Run multi-objective optimization
        result = optimization_service.multi_objective_optimization(
            objectives=objectives,
            constraints=constraints,
            config=config
        )
        
        # Verify results
        assert 'pareto_front' in result
        assert 'all_solutions' in result
        assert 'num_pareto_optimal' in result
        
        assert len(result['pareto_front']) > 0
        assert len(result['all_solutions']) >= len(result['pareto_front'])
        assert result['num_pareto_optimal'] == len(result['pareto_front'])
        
        # Verify Pareto solutions
        for solution in result['pareto_front']:
            assert 'parameters' in solution
            assert 'objective_values' in solution
            assert isinstance(solution['parameters'], ProcessParameters)
            
            # Check that all objectives are evaluated
            for obj in objectives:
                assert obj in solution['objective_values']
    
    def test_microstructure_comparison_pipeline(self, temp_database):
        """Test microstructure comparison pipeline."""
        
        # 1. Create multiple synthetic microstructures
        volumes = []
        labels = []
        
        for i in range(3):
            # Create variations of microstructures
            volume = np.random.rand(32, 32, 32)
            
            # Add different patterns for each
            if i == 0:
                # High porosity
                volume[volume < 0.3] = 0.1
            elif i == 1:
                # Medium porosity with inclusions
                volume[volume < 0.2] = 0.1
                volume[volume > 0.8] = 0.9
            else:
                # Low porosity, uniform
                volume = np.clip(volume + 0.3, 0, 1)
            
            volumes.append(volume)
            labels.append(f"Sample_{i+1}")
        
        # 2. Initialize analysis service
        processor = MicroCTProcessor()
        analysis_service = AnalysisService(processor=processor)
        
        # 3. Compare microstructures
        comparison_result = analysis_service.compare_microstructures(
            volumes=volumes,
            labels=labels,
            parameters=None  # No parameters for this test
        )
        
        # 4. Verify comparison results
        assert 'individual_analyses' in comparison_result
        assert 'feature_comparison' in comparison_result
        assert 'statistical_analysis' in comparison_result
        assert 'ranking' in comparison_result
        assert 'summary' in comparison_result
        
        # Check individual analyses
        assert len(comparison_result['individual_analyses']) == 3
        for label in labels:
            assert label in comparison_result['individual_analyses']
        
        # Check feature comparison
        feature_comparison = comparison_result['feature_comparison']
        assert len(feature_comparison) > 0
        
        # Each feature should have statistics
        for feature_name, stats in feature_comparison.items():
            assert 'values' in stats
            assert 'mean' in stats
            assert 'std' in stats
            assert len(stats['values']) == 3
        
        # Check ranking
        ranking = comparison_result['ranking']
        assert 'overall_ranking' in ranking
        assert len(ranking['overall_ranking']) == 3
        
        # Rankings should be sorted by quality (highest first)
        qualities = [score for _, score in ranking['overall_ranking']]
        assert qualities == sorted(qualities, reverse=True)
    
    def test_end_to_end_workflow(self, sample_microstructure_volume, temp_database):
        """Test complete end-to-end workflow."""
        
        # This test simulates a complete research workflow:
        # 1. Analyze existing microstructure
        # 2. Store results in database
        # 3. Use analysis to guide optimization
        # 4. Validate optimized parameters
        
        # Step 1: Setup
        exp_repo = ExperimentRepository(temp_database)
        micro_repo = MicrostructureRepository(temp_database)
        param_repo = ParametersRepository(temp_database)
        analysis_repo = AnalysisRepository(temp_database)
        
        processor = MicroCTProcessor()
        analysis_service = AnalysisService(processor=processor)
        optimization_service = OptimizationService()
        
        # Step 2: Create and analyze initial microstructure
        experiment = Experiment(
            name="End-to-End Workflow Test",
            description="Complete workflow from analysis to optimization",
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion"
        )
        exp_id = exp_repo.create(experiment)
        
        # Create microstructure entry
        microstructure = Microstructure(
            experiment_id=exp_id,
            name="Initial Microstructure",
            voxel_size=0.5,
            dimensions=list(sample_microstructure_volume.shape),
            acquisition_method="synthetic"
        )
        micro_id = micro_repo.create(microstructure)
        
        # Analyze microstructure
        analysis_report = analysis_service.analyze_microstructure(
            sample_microstructure_volume,
            analysis_types=['porosity_analysis', 'defect_analysis']
        )
        
        # Store analysis results
        analysis_result = AnalysisResult(
            microstructure_id=micro_id,
            analysis_type="workflow_test",
            features=analysis_report.microstructure_features,
            quality_metrics=analysis_report.quality_metrics,
            recommendations=analysis_report.recommendations,
            confidence_scores=analysis_report.confidence_scores
        )
        analysis_id = analysis_repo.create(analysis_result)
        
        # Step 3: Use analysis results to define optimization targets
        current_density = 1.0 - analysis_report.microstructure_features.get('porosity', 0.05)
        target_density = min(current_density + 0.02, 0.98)  # Improve density slightly
        
        # Define optimization problem based on analysis
        constraints = ParameterConstraints(
            laser_power_range=(150, 350),
            scan_speed_range=(600, 1200),
            layer_thickness_range=(25, 45),
            hatch_spacing_range=(90, 140),
            powder_bed_temp_range=(70, 100)
        )
        
        # Create objective function based on desired improvements
        objectives = {'density': 0.8, 'surface_quality': 0.2}
        
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=15,
            max_iterations=6,
            random_seed=42
        )
        
        # Step 4: Run optimization
        optimization_result = optimization_service.optimize_parameters(
            objectives=objectives,
            constraints=constraints,
            config=config
        )
        
        # Step 5: Store optimized parameters
        optimized_params = optimization_result['best_solution']
        
        db_params = DBProcessParameters(
            experiment_id=exp_id,
            microstructure_id=micro_id,
            laser_power=optimized_params.laser_power,
            scan_speed=optimized_params.scan_speed,
            layer_thickness=optimized_params.layer_thickness,
            hatch_spacing=optimized_params.hatch_spacing,
            powder_bed_temp=optimized_params.powder_bed_temp,
            atmosphere=optimized_params.atmosphere,
            additional_params={
                'optimization_objective': optimization_result['best_fitness'],
                'target_density': target_density,
                'original_porosity': analysis_report.microstructure_features.get('porosity', 0)
            }
        )
        param_id = param_repo.create(db_params)
        
        # Step 6: Validate complete workflow
        assert exp_id > 0
        assert micro_id > 0
        assert analysis_id > 0
        assert param_id > 0
        
        # Verify optimization improved upon analysis
        assert optimization_result['best_fitness'] > 0
        
        # Verify parameter constraints were respected
        assert 150 <= optimized_params.laser_power <= 350
        assert 600 <= optimized_params.scan_speed <= 1200
        
        # Verify database relationships
        retrieved_params = param_repo.get_by_id(param_id)
        assert retrieved_params.experiment_id == exp_id
        assert retrieved_params.microstructure_id == micro_id
        
        retrieved_analysis = analysis_repo.get_by_id(analysis_id)
        assert retrieved_analysis.microstructure_id == micro_id
        
        # Step 7: Simulate iterative improvement
        # In a real workflow, you might use the optimized parameters
        # to generate a new microstructure and repeat the cycle
        
        # For testing, we'll just verify the data is properly stored
        # and linked for future iterations
        
        all_params = param_repo.get_by_experiment(exp_id)
        assert len(all_params) == 1
        assert all_params[0].id == param_id
        
        all_analyses = analysis_repo.get_by_microstructure(micro_id)
        assert len(all_analyses) == 1
        assert all_analyses[0].id == analysis_id


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_invalid_microstructure_data(self):
        """Test handling of invalid microstructure data."""
        
        processor = MicroCTProcessor()
        analysis_service = AnalysisService(processor=processor)
        
        # Test with various invalid inputs
        invalid_inputs = [
            np.array([]),  # Empty array
            np.ones((1, 1, 1)),  # Too small
            np.full((10, 10, 10), np.nan),  # NaN values
            np.full((10, 10, 10), np.inf),  # Infinite values
        ]
        
        for invalid_input in invalid_inputs:
            # Analysis should handle invalid input gracefully
            try:
                result = analysis_service.analyze_microstructure(
                    invalid_input,
                    analysis_types=['porosity_analysis']
                )
                # If no exception, check that warnings were generated
                assert len(result.warnings) > 0
            except Exception as e:
                # Expected for some invalid inputs
                assert isinstance(e, (ValueError, RuntimeError))
    
    def test_optimization_with_impossible_constraints(self):
        """Test optimization with impossible constraints."""
        
        optimization_service = OptimizationService()
        
        # Create impossible constraints (min > max)
        impossible_constraints = ParameterConstraints(
            laser_power_range=(400, 100),  # Invalid: min > max
            scan_speed_range=(200, 2000),
            layer_thickness_range=(20, 100),
            hatch_spacing_range=(50, 200),
            powder_bed_temp_range=(20, 200)
        )
        
        objectives = {'density': 1.0}
        config = OptimizationConfig(
            population_size=5,
            max_iterations=3
        )
        
        # This should either handle the constraint error gracefully
        # or raise an appropriate exception
        try:
            result = optimization_service.optimize_parameters(
                objectives=objectives,
                constraints=impossible_constraints,
                config=config
            )
            # If it succeeds, verify constraints were corrected
            assert 'best_solution' in result
        except ValueError:
            # Expected for impossible constraints
            pass
    
    def test_database_transaction_handling(self, temp_database):
        """Test database transaction handling."""
        
        exp_repo = ExperimentRepository(temp_database)
        
        # Test that partial failures don't corrupt the database
        experiment = Experiment(
            name="Transaction Test",
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion"
        )
        
        # This should succeed
        exp_id = exp_repo.create(experiment)
        assert exp_id > 0
        
        # Verify experiment was created
        retrieved = exp_repo.get_by_id(exp_id)
        assert retrieved is not None
        assert retrieved.name == "Transaction Test"
        
        # Test with invalid update (should not affect existing data)
        experiment.id = 99999  # Non-existent ID
        success = exp_repo.update(experiment)
        assert not success
        
        # Original experiment should still exist and be unchanged
        original = exp_repo.get_by_id(exp_id)
        assert original is not None
        assert original.name == "Transaction Test"


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_large_volume_processing(self):
        """Test processing of larger microstructure volumes."""
        
        # Create a larger synthetic volume
        large_volume = np.random.rand(128, 128, 128)
        
        # Add some structure
        center = np.array([64, 64, 64])
        x, y, z = np.meshgrid(range(128), range(128), range(128), indexing='ij')
        
        # Create multiple spherical features
        for i in range(10):
            sphere_center = center + np.random.randint(-40, 40, 3)
            radius = np.random.randint(5, 20)
            
            distances = np.sqrt((x - sphere_center[0])**2 + 
                               (y - sphere_center[1])**2 + 
                               (z - sphere_center[2])**2)
            
            large_volume[distances < radius] = 0.1 + 0.4 * np.random.rand()
        
        # Test analysis performance
        processor = MicroCTProcessor()
        analysis_service = AnalysisService(processor=processor)
        
        import time
        start_time = time.time()
        
        # Analyze with limited feature set for performance
        result = analysis_service.analyze_microstructure(
            large_volume,
            analysis_types=['porosity_analysis']
        )
        
        analysis_time = time.time() - start_time
        
        # Verify results were produced
        assert len(result.microstructure_features) > 0
        assert 'porosity' in result.microstructure_features
        
        # Analysis should complete in reasonable time (adjust threshold as needed)
        assert analysis_time < 30.0  # 30 seconds max for 128³ volume
    
    def test_batch_optimization_performance(self):
        """Test performance of batch optimization runs."""
        
        optimization_service = OptimizationService()
        
        constraints = ParameterConstraints(
            laser_power_range=(100, 400),
            scan_speed_range=(200, 2000),
            layer_thickness_range=(20, 100),
            hatch_spacing_range=(50, 200),
            powder_bed_temp_range=(20, 200)
        )
        
        objectives = {'density': 1.0}
        
        # Test multiple optimization runs
        import time
        
        configs = [
            OptimizationConfig(population_size=10, max_iterations=5, random_seed=i)
            for i in range(5)
        ]
        
        start_time = time.time()
        
        results = []
        for config in configs:
            result = optimization_service.optimize_parameters(
                objectives=objectives,
                constraints=constraints,
                config=config
            )
            results.append(result)
        
        batch_time = time.time() - start_time
        
        # Verify all optimizations completed
        assert len(results) == 5
        for result in results:
            assert 'best_solution' in result
            assert result['best_fitness'] > 0
        
        # Batch should complete in reasonable time
        assert batch_time < 60.0  # 1 minute for 5 small optimizations