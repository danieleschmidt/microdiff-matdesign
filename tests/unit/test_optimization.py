"""Unit tests for optimization service functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from microdiff_matdesign.services.optimization import (
    OptimizationService, OptimizationConfig, OptimizationAlgorithm,
    GeneticAlgorithmOptimizer, ParticleSwarmOptimizer, BayesianOptimizer
)
from microdiff_matdesign.services.parameter_generation import (
    ParameterConstraints, OptimizationObjective
)
from microdiff_matdesign.core import ProcessParameters


@pytest.fixture
def sample_constraints():
    """Create sample parameter constraints."""
    return ParameterConstraints(
        laser_power_range=(100, 400),
        scan_speed_range=(200, 2000),
        layer_thickness_range=(20, 100),
        hatch_spacing_range=(50, 200),
        powder_bed_temp_range=(20, 200)
    )


@pytest.fixture
def sample_parameters():
    """Create sample process parameters."""
    return ProcessParameters(
        laser_power=200.0,
        scan_speed=800.0,
        layer_thickness=30.0,
        hatch_spacing=120.0,
        powder_bed_temp=80.0
    )


@pytest.fixture
def simple_objective_function():
    """Create simple objective function for testing."""
    def objective(params: ProcessParameters) -> float:
        # Simple quadratic function with maximum at specific values
        target_power = 250
        target_speed = 1000
        
        power_error = (params.laser_power - target_power) ** 2
        speed_error = (params.scan_speed - target_speed) ** 2
        
        # Return fitness (higher is better)
        return 1.0 / (1.0 + 0.001 * (power_error + speed_error))
    
    return objective


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM
        assert config.population_size == 50
        assert config.max_iterations == 100
        assert config.convergence_tolerance == 1e-6
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.random_seed is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
            population_size=30,
            max_iterations=50,
            random_seed=42
        )
        
        assert config.algorithm == OptimizationAlgorithm.PARTICLE_SWARM
        assert config.population_size == 30
        assert config.max_iterations == 50
        assert config.random_seed == 42


class TestGeneticAlgorithmOptimizer:
    """Test genetic algorithm optimizer."""
    
    def test_parameter_vector_conversion(self, sample_constraints, sample_parameters):
        """Test parameter to vector conversion."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        # Test parameter to vector
        vector = optimizer._parameter_to_vector(sample_parameters)
        assert len(vector) == 5
        assert vector[0] == sample_parameters.laser_power
        assert vector[1] == sample_parameters.scan_speed
        
        # Test vector to parameter
        params_restored = optimizer._vector_to_parameter(vector)
        assert params_restored.laser_power == sample_parameters.laser_power
        assert params_restored.scan_speed == sample_parameters.scan_speed
    
    def test_constraint_application(self, sample_constraints):
        """Test constraint application."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        # Test with values outside constraints
        vector = np.array([50, 3000, 150, 25, 250])  # All outside bounds
        constrained = optimizer._apply_constraints(vector, sample_constraints)
        
        assert 100 <= constrained[0] <= 400  # laser_power
        assert 200 <= constrained[1] <= 2000  # scan_speed
        assert 20 <= constrained[2] <= 100  # layer_thickness
        assert 50 <= constrained[3] <= 200  # hatch_spacing
        assert 20 <= constrained[4] <= 200  # powder_bed_temp
    
    def test_random_individual_generation(self, sample_constraints):
        """Test random individual generation."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        individual = optimizer._generate_random_individual(sample_constraints)
        
        assert len(individual) == 5
        assert 100 <= individual[0] <= 400  # laser_power
        assert 200 <= individual[1] <= 2000  # scan_speed
        assert 20 <= individual[2] <= 100  # layer_thickness
        assert 50 <= individual[3] <= 200  # hatch_spacing
        assert 20 <= individual[4] <= 200  # powder_bed_temp
    
    def test_tournament_selection(self, sample_constraints):
        """Test tournament selection."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        fitness_scores = [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.7, 0.15]
        selected_indices = optimizer._tournament_selection(fitness_scores)
        
        assert len(selected_indices) == len(fitness_scores)
        assert all(0 <= idx < len(fitness_scores) for idx in selected_indices)
        
        # Higher fitness should be more likely to be selected
        # This is probabilistic, so we just check that selection occurred
        assert isinstance(selected_indices[0], (int, np.integer))
    
    def test_crossover(self, sample_constraints):
        """Test crossover operation."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        parent1 = np.array([200, 800, 30, 120, 80])
        parent2 = np.array([300, 1000, 40, 100, 100])
        
        child1, child2 = optimizer._crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        
        # Children should have values from both parents
        for i in range(len(parent1)):
            assert child1[i] in [parent1[i], parent2[i]]
            assert child2[i] in [parent1[i], parent2[i]]
    
    def test_mutation(self, sample_constraints):
        """Test mutation operation."""
        config = OptimizationConfig(population_size=10, max_iterations=5)
        optimizer = GeneticAlgorithmOptimizer(config)
        
        individual = np.array([200, 800, 30, 120, 80])
        mutated = optimizer._mutate(individual, sample_constraints)
        
        assert len(mutated) == len(individual)
        
        # Mutated individual should be different (with high probability)
        # and within constraints
        assert 100 <= mutated[0] <= 400
        assert 200 <= mutated[1] <= 2000
        assert 20 <= mutated[2] <= 100
        assert 50 <= mutated[3] <= 200
        assert 20 <= mutated[4] <= 200
    
    def test_optimization_run(self, sample_constraints, simple_objective_function):
        """Test complete optimization run."""
        config = OptimizationConfig(
            population_size=20, 
            max_iterations=10,
            random_seed=42
        )
        optimizer = GeneticAlgorithmOptimizer(config)
        
        result = optimizer.optimize(simple_objective_function, sample_constraints)
        
        assert 'best_solution' in result
        assert 'best_fitness' in result
        assert 'generations' in result
        assert 'fitness_history' in result
        assert 'final_population' in result
        
        assert isinstance(result['best_solution'], ProcessParameters)
        assert result['best_fitness'] > 0
        assert result['generations'] <= config.max_iterations
        assert len(result['final_population']) == config.population_size


class TestParticleSwarmOptimizer:
    """Test particle swarm optimizer."""
    
    def test_optimization_run(self, sample_constraints, simple_objective_function):
        """Test complete PSO optimization run."""
        config = OptimizationConfig(
            population_size=20,
            max_iterations=10,
            random_seed=42
        )
        optimizer = ParticleSwarmOptimizer(config)
        
        result = optimizer.optimize(simple_objective_function, sample_constraints)
        
        assert 'best_solution' in result
        assert 'best_fitness' in result
        assert 'iterations' in result
        assert 'fitness_history' in result
        assert 'final_population' in result
        
        assert isinstance(result['best_solution'], ProcessParameters)
        assert result['best_fitness'] > 0
        assert result['iterations'] <= config.max_iterations
        assert len(result['final_population']) == config.population_size
    
    def test_convergence_behavior(self, sample_constraints, simple_objective_function):
        """Test PSO convergence behavior."""
        config = OptimizationConfig(
            population_size=10,
            max_iterations=20,
            convergence_tolerance=1e-3,
            random_seed=42
        )
        optimizer = ParticleSwarmOptimizer(config)
        
        result = optimizer.optimize(simple_objective_function, sample_constraints)
        
        # Check that fitness improves over time
        fitness_history = result['fitness_history']['best']
        assert len(fitness_history) > 1
        
        # Final fitness should be at least as good as initial
        assert fitness_history[-1] >= fitness_history[0]


class TestBayesianOptimizer:
    """Test Bayesian optimizer."""
    
    @pytest.mark.skip(reason="Requires scikit-learn - optional dependency")
    def test_optimization_run(self, sample_constraints, simple_objective_function):
        """Test complete Bayesian optimization run."""
        config = OptimizationConfig(
            population_size=20,
            max_iterations=15,
            random_seed=42
        )
        optimizer = BayesianOptimizer(config)
        
        result = optimizer.optimize(simple_objective_function, sample_constraints)
        
        assert 'best_solution' in result
        assert 'best_fitness' in result
        assert 'iterations' in result
        assert 'fitness_history' in result
        
        assert isinstance(result['best_solution'], ProcessParameters)
        assert result['best_fitness'] > 0
    
    def test_normal_pdf_cdf(self):
        """Test normal distribution functions."""
        config = OptimizationConfig()
        optimizer = BayesianOptimizer(config)
        
        # Test standard normal PDF
        pdf_val = optimizer._normal_pdf(0.0)
        expected_pdf = 1.0 / np.sqrt(2 * np.pi)
        assert abs(pdf_val - expected_pdf) < 1e-10
        
        # Test standard normal CDF
        cdf_val = optimizer._normal_cdf(0.0)
        assert abs(cdf_val - 0.5) < 1e-6  # Should be 0.5 at zero


class TestOptimizationService:
    """Test optimization service."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = OptimizationService()
        
        assert OptimizationAlgorithm.GENETIC_ALGORITHM in service.optimizers
        assert OptimizationAlgorithm.PARTICLE_SWARM in service.optimizers
        assert OptimizationAlgorithm.BAYESIAN_OPTIMIZATION in service.optimizers
    
    def test_default_property_predictors(self):
        """Test default property predictors."""
        service = OptimizationService()
        predictors = service._default_property_predictors()
        
        assert 'density' in predictors
        assert 'strength' in predictors
        assert 'surface_quality' in predictors
        assert 'build_speed' in predictors
        
        # Test density predictor
        params = ProcessParameters(laser_power=200, scan_speed=800, 
                                 layer_thickness=30, hatch_spacing=120)
        density = predictors['density'](params)
        assert 0.8 <= density <= 1.0
        
        # Test strength predictor
        strength = predictors['strength'](params)
        assert 700 <= strength <= 1300
    
    def test_objective_normalization(self):
        """Test objective value normalization."""
        service = OptimizationService()
        
        # Test density normalization
        normalized = service._normalize_objective_value('density', 0.95)
        assert 0.0 <= normalized <= 1.0
        
        # Test surface quality normalization (lower is better)
        normalized = service._normalize_objective_value('surface_quality', 10)
        assert 0.0 <= normalized <= 1.0
        
        # Test unknown objective (should clamp to 0-1)
        normalized = service._normalize_objective_value('unknown_objective', 1.5)
        assert normalized == 1.0
    
    def test_objective_function_creation(self, sample_parameters):
        """Test objective function creation."""
        service = OptimizationService()
        
        objectives = {'density': 1.0, 'strength': 0.5}
        predictors = service._default_property_predictors()
        
        objective_fn = service._create_objective_function(objectives, predictors)
        
        # Test that function is callable and returns reasonable value
        fitness = objective_fn(sample_parameters)
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 2.0  # Should be reasonable range
    
    def test_weight_combinations_generation(self):
        """Test weight combination generation."""
        service = OptimizationService()
        
        # Test with 3 objectives
        combinations = service._generate_weight_combinations(3, 10)
        
        assert len(combinations) == 10
        
        # Check that corner solutions are included
        corner_solutions = 0
        for combo in combinations:
            if combo.count(1.0) == 1 and combo.count(0.0) == 2:
                corner_solutions += 1
        
        assert corner_solutions == 3  # One for each objective
        
        # Check that all combinations sum to 1
        for combo in combinations:
            assert abs(sum(combo) - 1.0) < 1e-10
    
    def test_pareto_front_extraction(self):
        """Test Pareto front extraction."""
        service = OptimizationService()
        
        # Create mock solutions
        solutions = [
            {
                'parameters': ProcessParameters(laser_power=200),
                'objective_values': {'density': 0.9, 'strength': 900}
            },
            {
                'parameters': ProcessParameters(laser_power=250),
                'objective_values': {'density': 0.95, 'strength': 850}
            },
            {
                'parameters': ProcessParameters(laser_power=300),
                'objective_values': {'density': 0.85, 'strength': 950}
            },
            {
                'parameters': ProcessParameters(laser_power=350),
                'objective_values': {'density': 0.8, 'strength': 800}  # Dominated
            }
        ]
        
        objectives = ['density', 'strength']
        pareto_front = service._extract_pareto_front(solutions, objectives)
        
        # Solution 4 should be dominated (lower in both objectives)
        assert len(pareto_front) == 3
        
        # Check that dominated solution is not in Pareto front
        pareto_powers = [sol['parameters'].laser_power for sol in pareto_front]
        assert 350 not in pareto_powers
    
    def test_optimize_parameters(self, sample_constraints):
        """Test parameter optimization."""
        service = OptimizationService()
        
        objectives = {'density': 1.0}
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=10,
            max_iterations=5,
            random_seed=42
        )
        
        result = service.optimize_parameters(objectives, sample_constraints, config)
        
        assert 'best_solution' in result
        assert 'best_fitness' in result
        assert 'objectives' in result
        assert 'constraints' in result
        assert 'config' in result
        
        assert result['objectives'] == objectives
        assert isinstance(result['best_solution'], ProcessParameters)
    
    def test_multi_objective_optimization(self, sample_constraints):
        """Test multi-objective optimization."""
        service = OptimizationService()
        
        objectives = ['density', 'strength']
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=10,
            max_iterations=3,  # Small for testing
            random_seed=42
        )
        
        result = service.multi_objective_optimization(objectives, sample_constraints, config)
        
        assert 'pareto_front' in result
        assert 'all_solutions' in result
        assert 'objectives' in result
        assert 'num_pareto_optimal' in result
        
        assert result['objectives'] == objectives
        assert len(result['pareto_front']) <= len(result['all_solutions'])
        assert result['num_pareto_optimal'] == len(result['pareto_front'])
    
    def test_unsupported_algorithm(self, sample_constraints):
        """Test handling of unsupported algorithm."""
        service = OptimizationService()
        
        # Mock an unsupported algorithm
        config = OptimizationConfig()
        config.algorithm = "unsupported_algorithm"
        
        with pytest.raises(ValueError, match="Unsupported optimization algorithm"):
            service.optimize_parameters({}, sample_constraints, config)


class TestOptimizationIntegration:
    """Test integration between optimization components."""
    
    def test_full_optimization_workflow(self, sample_constraints):
        """Test complete optimization workflow."""
        service = OptimizationService()
        
        # Define objectives
        objectives = {'density': 0.6, 'strength': 0.4}
        
        # Configure optimization
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=15,
            max_iterations=8,
            random_seed=42
        )
        
        # Run optimization
        result = service.optimize_parameters(objectives, sample_constraints, config)
        
        # Validate results
        assert result['best_fitness'] > 0
        assert isinstance(result['best_solution'], ProcessParameters)
        
        # Check that solution respects constraints
        solution = result['best_solution']
        assert 100 <= solution.laser_power <= 400
        assert 200 <= solution.scan_speed <= 2000
        assert 20 <= solution.layer_thickness <= 100
        assert 50 <= solution.hatch_spacing <= 200
        assert 20 <= solution.powder_bed_temp <= 200
        
        # Check convergence information
        assert 'fitness_history' in result
        fitness_history = result['fitness_history']['best']
        assert len(fitness_history) > 0
        
        # Fitness should generally improve or stay the same
        assert fitness_history[-1] >= fitness_history[0]
    
    def test_algorithm_comparison(self, sample_constraints, simple_objective_function):
        """Test comparison between different algorithms."""
        
        algorithms = [
            OptimizationAlgorithm.GENETIC_ALGORITHM,
            OptimizationAlgorithm.PARTICLE_SWARM
        ]
        
        results = {}
        
        for algorithm in algorithms:
            config = OptimizationConfig(
                algorithm=algorithm,
                population_size=10,
                max_iterations=5,
                random_seed=42
            )
            
            optimizer_class = {
                OptimizationAlgorithm.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer,
                OptimizationAlgorithm.PARTICLE_SWARM: ParticleSwarmOptimizer,
            }[algorithm]
            
            optimizer = optimizer_class(config)
            result = optimizer.optimize(simple_objective_function, sample_constraints)
            results[algorithm] = result
        
        # All algorithms should produce valid results
        for algorithm, result in results.items():
            assert 'best_solution' in result
            assert 'best_fitness' in result
            assert result['best_fitness'] > 0
            assert isinstance(result['best_solution'], ProcessParameters)
    
    def test_optimization_reproducibility(self, sample_constraints, simple_objective_function):
        """Test optimization reproducibility with random seed."""
        
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
            population_size=10,
            max_iterations=5,
            random_seed=123
        )
        
        # Run optimization twice with same seed
        optimizer1 = GeneticAlgorithmOptimizer(config)
        result1 = optimizer1.optimize(simple_objective_function, sample_constraints)
        
        optimizer2 = GeneticAlgorithmOptimizer(config)
        result2 = optimizer2.optimize(simple_objective_function, sample_constraints)
        
        # Results should be identical with same seed
        assert result1['best_fitness'] == result2['best_fitness']
        
        # Parameters should be the same
        params1 = result1['best_solution']
        params2 = result2['best_solution']
        
        assert params1.laser_power == params2.laser_power
        assert params1.scan_speed == params2.scan_speed
        assert params1.layer_thickness == params2.layer_thickness
        assert params1.hatch_spacing == params2.hatch_spacing