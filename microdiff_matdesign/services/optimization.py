"""Optimization service for multi-objective parameter optimization."""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..core import ProcessParameters
from .parameter_generation import ParameterConstraints, OptimizationObjective


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    RANDOM_SEARCH = "random_search"
    GRADIENT_DESCENT = "gradient_descent"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithm."""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM
    population_size: int = 50
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    random_seed: Optional[int] = None


class BaseOptimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def optimize(self, objective_function: Callable, constraints: ParameterConstraints,
                initial_population: Optional[List[ProcessParameters]] = None) -> Dict[str, Any]:
        """Optimize the objective function subject to constraints."""
        pass
    
    def _parameter_to_vector(self, params: ProcessParameters) -> np.ndarray:
        """Convert ProcessParameters to optimization vector."""
        return np.array([
            params.laser_power,
            params.scan_speed,
            params.layer_thickness,
            params.hatch_spacing,
            params.powder_bed_temp
        ])
    
    def _vector_to_parameter(self, vector: np.ndarray) -> ProcessParameters:
        """Convert optimization vector to ProcessParameters."""
        return ProcessParameters(
            laser_power=float(vector[0]),
            scan_speed=float(vector[1]),
            layer_thickness=float(vector[2]),
            hatch_spacing=float(vector[3]),
            powder_bed_temp=float(vector[4]) if len(vector) > 4 else 80.0
        )
    
    def _apply_constraints(self, vector: np.ndarray, constraints: ParameterConstraints) -> np.ndarray:
        """Apply box constraints to parameter vector."""
        constraint_dict = constraints.to_dict()
        bounds = [
            constraint_dict['laser_power'],
            constraint_dict['scan_speed'],
            constraint_dict['layer_thickness'],
            constraint_dict['hatch_spacing'],
            constraint_dict['powder_bed_temp']
        ]
        
        constrained = vector.copy()
        for i, (min_val, max_val) in enumerate(bounds):
            if i < len(constrained):
                constrained[i] = np.clip(constrained[i], min_val, max_val)
        
        return constrained


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer for parameter optimization."""
    
    def optimize(self, objective_function: Callable, constraints: ParameterConstraints,
                initial_population: Optional[List[ProcessParameters]] = None) -> Dict[str, Any]:
        """Optimize using genetic algorithm."""
        
        # Initialize population
        if initial_population is not None:
            population = [self._parameter_to_vector(p) for p in initial_population]
            while len(population) < self.config.population_size:
                population.append(self._generate_random_individual(constraints))
        else:
            population = [self._generate_random_individual(constraints) 
                         for _ in range(self.config.population_size)]
        
        # Evolution tracking
        best_fitness_history = []
        mean_fitness_history = []
        
        for generation in range(self.config.max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                params = self._vector_to_parameter(individual)
                fitness = objective_function(params)
                fitness_scores.append(fitness)
            
            # Track statistics
            best_fitness = max(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)
            
            # Check convergence
            if len(best_fitness_history) > 10:
                recent_improvement = (best_fitness_history[-1] - best_fitness_history[-10]) / 10
                if recent_improvement < self.config.convergence_tolerance:
                    break
            
            # Selection
            selected_indices = self._tournament_selection(fitness_scores)
            selected_population = [population[i] for i in selected_indices]
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1, constraints)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2, constraints)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.population_size]
        
        # Find best solution
        final_fitness = [objective_function(self._vector_to_parameter(ind)) for ind in population]
        best_index = np.argmax(final_fitness)
        best_solution = self._vector_to_parameter(population[best_index])
        
        return {
            'best_solution': best_solution,
            'best_fitness': final_fitness[best_index],
            'generations': generation + 1,
            'fitness_history': {
                'best': best_fitness_history,
                'mean': mean_fitness_history
            },
            'final_population': [self._vector_to_parameter(ind) for ind in population]
        }
    
    def _generate_random_individual(self, constraints: ParameterConstraints) -> np.ndarray:
        """Generate random individual within constraints."""
        constraint_dict = constraints.to_dict()
        
        individual = []
        for param_name in ['laser_power', 'scan_speed', 'layer_thickness', 'hatch_spacing', 'powder_bed_temp']:
            min_val, max_val = constraint_dict[param_name]
            value = np.random.uniform(min_val, max_val)
            individual.append(value)
        
        return np.array(individual)
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
        """Tournament selection for genetic algorithm."""
        population_size = len(fitness_scores)
        selected_indices = []
        
        for _ in range(population_size):
            # Random tournament
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices.append(winner_index)
        
        return selected_indices
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover for genetic algorithm."""
        mask = np.random.random(len(parent1)) < 0.5
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, constraints: ParameterConstraints, 
               mutation_strength: float = 0.1) -> np.ndarray:
        """Gaussian mutation for genetic algorithm."""
        mutated = individual.copy()
        
        # Gaussian perturbation
        perturbation = np.random.normal(0, mutation_strength, len(individual))
        mutated += perturbation * individual  # Relative perturbation
        
        # Apply constraints
        mutated = self._apply_constraints(mutated, constraints)
        
        return mutated


class ParticleSwarmOptimizer(BaseOptimizer):
    """Particle Swarm Optimization for parameter optimization."""
    
    def optimize(self, objective_function: Callable, constraints: ParameterConstraints,
                initial_population: Optional[List[ProcessParameters]] = None) -> Dict[str, Any]:
        """Optimize using particle swarm optimization."""
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Initialize particles
        if initial_population is not None:
            positions = [self._parameter_to_vector(p) for p in initial_population]
            while len(positions) < self.config.population_size:
                positions.append(self._generate_random_individual(constraints))
        else:
            positions = [self._generate_random_individual(constraints) 
                        for _ in range(self.config.population_size)]
        
        # Initialize velocities
        velocities = [np.random.normal(0, 0.1, len(positions[0])) for _ in range(self.config.population_size)]
        
        # Initialize personal and global bests
        personal_bests = positions.copy()
        personal_best_fitness = [objective_function(self._vector_to_parameter(pos)) for pos in positions]
        
        global_best_index = np.argmax(personal_best_fitness)
        global_best = personal_bests[global_best_index].copy()
        global_best_fitness = personal_best_fitness[global_best_index]
        
        # Evolution tracking
        fitness_history = [global_best_fitness]
        
        for iteration in range(self.config.max_iterations):
            for i in range(self.config.population_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                
                cognitive_component = c1 * r1 * (personal_bests[i] - positions[i])
                social_component = c2 * r2 * (global_best - positions[i])
                
                velocities[i] = w * velocities[i] + cognitive_component + social_component
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = self._apply_constraints(positions[i], constraints)
                
                # Evaluate fitness
                params = self._vector_to_parameter(positions[i])
                fitness = objective_function(params)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_bests[i] = positions[i].copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best = positions[i].copy()
            
            fitness_history.append(global_best_fitness)
            
            # Check convergence
            if len(fitness_history) > 10:
                recent_improvement = (fitness_history[-1] - fitness_history[-10]) / 10
                if recent_improvement < self.config.convergence_tolerance:
                    break
        
        return {
            'best_solution': self._vector_to_parameter(global_best),
            'best_fitness': global_best_fitness,
            'iterations': iteration + 1,
            'fitness_history': {'best': fitness_history},
            'final_population': [self._vector_to_parameter(pos) for pos in positions]
        }


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using Gaussian Process surrogate models."""
    
    def optimize(self, objective_function: Callable, constraints: ParameterConstraints,
                initial_population: Optional[List[ProcessParameters]] = None) -> Dict[str, Any]:
        """Optimize using Bayesian optimization."""
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("Bayesian optimization requires scikit-learn and scipy")
        
        # Initialize with random samples
        if initial_population is not None:
            X = np.array([self._parameter_to_vector(p) for p in initial_population])
            while len(X) < min(10, self.config.population_size // 5):
                new_sample = self._generate_random_individual(constraints)
                X = np.vstack([X, new_sample])
        else:
            X = np.array([self._generate_random_individual(constraints) 
                         for _ in range(min(10, self.config.population_size // 5))])
        
        # Evaluate initial samples
        y = np.array([objective_function(self._vector_to_parameter(x)) for x in X])
        
        # Initialize Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        best_fitness_history = [np.max(y)]
        
        for iteration in range(self.config.max_iterations - len(X)):
            # Fit GP to current data
            gp.fit(X, y)
            
            # Acquisition function (Expected Improvement)
            def expected_improvement(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                
                best_f = np.max(y)
                xi = 0.01  # Exploration parameter
                
                with np.errstate(divide='warn'):
                    imp = mu - best_f - xi
                    Z = imp / sigma
                    ei = imp * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
                    ei[sigma == 0.0] = 0.0
                
                return -ei[0]  # Minimize negative EI
            
            # Optimize acquisition function
            constraint_dict = constraints.to_dict()
            bounds = [
                constraint_dict['laser_power'],
                constraint_dict['scan_speed'],
                constraint_dict['layer_thickness'],
                constraint_dict['hatch_spacing'],
                constraint_dict['powder_bed_temp']
            ]
            
            # Multi-start optimization of acquisition function
            best_ei = float('inf')
            best_x_next = None
            
            for _ in range(10):  # Multiple random starts
                x0 = self._generate_random_individual(constraints)
                
                result = minimize(
                    expected_improvement,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x_next = result.x
            
            # Evaluate new point
            if best_x_next is not None:
                x_next = best_x_next
            else:
                x_next = self._generate_random_individual(constraints)
            
            y_next = objective_function(self._vector_to_parameter(x_next))
            
            # Add to dataset
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            
            best_fitness_history.append(np.max(y))
            
            # Check convergence
            if len(best_fitness_history) > 10:
                recent_improvement = (best_fitness_history[-1] - best_fitness_history[-10]) / 10
                if recent_improvement < self.config.convergence_tolerance:
                    break
        
        # Find best solution
        best_index = np.argmax(y)
        best_solution = self._vector_to_parameter(X[best_index])
        
        return {
            'best_solution': best_solution,
            'best_fitness': y[best_index],
            'iterations': len(X),
            'fitness_history': {'best': best_fitness_history},
            'final_population': [self._vector_to_parameter(x) for x in X[-self.config.population_size:]]
        }
    
    def _normal_pdf(self, x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _normal_cdf(self, x):
        """Standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))


class OptimizationService:
    """Service for multi-objective parameter optimization."""
    
    def __init__(self):
        """Initialize optimization service."""
        self.optimizers = {
            OptimizationAlgorithm.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer,
            OptimizationAlgorithm.PARTICLE_SWARM: ParticleSwarmOptimizer,
            OptimizationAlgorithm.BAYESIAN_OPTIMIZATION: BayesianOptimizer,
        }
    
    def optimize_parameters(self, 
                          objectives: Dict[str, float],
                          constraints: ParameterConstraints,
                          config: OptimizationConfig,
                          property_predictors: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Optimize parameters for multiple objectives.
        
        Args:
            objectives: Dictionary of objective names and weights
            constraints: Parameter constraints
            config: Optimization configuration
            property_predictors: Property prediction functions
            
        Returns:
            Optimization results
        """
        
        if config.algorithm not in self.optimizers:
            raise ValueError(f"Unsupported optimization algorithm: {config.algorithm}")
        
        # Create optimizer
        optimizer = self.optimizers[config.algorithm](config)
        
        # Create objective function
        objective_fn = self._create_objective_function(objectives, property_predictors)
        
        # Run optimization
        result = optimizer.optimize(objective_fn, constraints)
        
        # Add additional information
        result['objectives'] = objectives
        result['constraints'] = constraints
        result['config'] = config
        
        return result
    
    def multi_objective_optimization(self,
                                   objectives: List[str],
                                   constraints: ParameterConstraints,
                                   config: OptimizationConfig,
                                   property_predictors: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Perform multi-objective optimization to find Pareto front.
        
        Args:
            objectives: List of objective names to optimize
            constraints: Parameter constraints
            config: Optimization configuration
            property_predictors: Property prediction functions
            
        Returns:
            Pareto front solutions and analysis
        """
        
        # Use weighted sum approach with multiple runs
        pareto_solutions = []
        
        # Generate different weight combinations
        num_weight_combinations = 20
        weight_combinations = self._generate_weight_combinations(len(objectives), num_weight_combinations)
        
        for weights in weight_combinations:
            # Create weighted objectives
            weighted_objectives = {obj: weight for obj, weight in zip(objectives, weights)}
            
            # Optimize
            result = self.optimize_parameters(
                weighted_objectives, constraints, config, property_predictors
            )
            
            if 'best_solution' in result:
                solution_info = {
                    'parameters': result['best_solution'],
                    'weights': weights,
                    'fitness': result['best_fitness']
                }
                
                # Evaluate individual objectives
                if property_predictors:
                    solution_info['objective_values'] = {}
                    for obj in objectives:
                        if obj in property_predictors:
                            value = property_predictors[obj](result['best_solution'])
                            solution_info['objective_values'][obj] = value
                
                pareto_solutions.append(solution_info)
        
        # Filter for Pareto optimal solutions
        pareto_front = self._extract_pareto_front(pareto_solutions, objectives)
        
        return {
            'pareto_front': pareto_front,
            'all_solutions': pareto_solutions,
            'objectives': objectives,
            'num_pareto_optimal': len(pareto_front)
        }
    
    def _create_objective_function(self, objectives: Dict[str, float],
                                 property_predictors: Optional[Dict[str, Callable]] = None) -> Callable:
        """Create weighted objective function."""
        
        if property_predictors is None:
            property_predictors = self._default_property_predictors()
        
        def objective_function(params: ProcessParameters) -> float:
            total_score = 0.0
            total_weight = 0.0
            
            for obj_name, weight in objectives.items():
                if obj_name in property_predictors:
                    value = property_predictors[obj_name](params)
                    
                    # Normalize to 0-1 scale (objective-specific)
                    normalized_value = self._normalize_objective_value(obj_name, value)
                    
                    total_score += weight * normalized_value
                    total_weight += abs(weight)
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
        
        return objective_function
    
    def _default_property_predictors(self) -> Dict[str, Callable]:
        """Default property prediction functions."""
        
        def predict_density(params: ProcessParameters) -> float:
            energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
            if energy_density < 40:
                return 0.85 + 0.01 * energy_density
            elif energy_density > 120:
                return 0.98 - 0.001 * (energy_density - 120)
            else:
                return 0.85 + 0.0175 * (energy_density - 40)
        
        def predict_strength(params: ProcessParameters) -> float:
            base_strength = 900
            energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
            
            if energy_density < 60:
                strength_factor = 0.8 + 0.003 * energy_density
            else:
                strength_factor = 1.0 + 0.001 * (energy_density - 60)
            
            layer_factor = 1.0 + 0.002 * (50 - params.layer_thickness)
            return base_strength * strength_factor * layer_factor
        
        def predict_surface_quality(params: ProcessParameters) -> float:
            base_roughness = 10.0
            layer_effect = params.layer_thickness / 30.0
            energy_density = params.laser_power / (params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000)
            
            if energy_density > 100:
                energy_effect = 1.0 + 0.01 * (energy_density - 100)
            else:
                energy_effect = 1.0
            
            return base_roughness * layer_effect * energy_effect
        
        def predict_build_speed(params: ProcessParameters) -> float:
            # Volume rate (mm³/s)
            return params.scan_speed * params.hatch_spacing * params.layer_thickness / 1000
        
        return {
            'density': predict_density,
            'strength': predict_strength,
            'surface_quality': predict_surface_quality,
            'build_speed': predict_build_speed
        }
    
    def _normalize_objective_value(self, obj_name: str, value: float) -> float:
        """Normalize objective value to 0-1 scale."""
        
        normalization_ranges = {
            'density': (0.8, 1.0),
            'strength': (800, 1200),
            'surface_quality': (5, 20),  # Lower is better, so we'll invert
            'build_speed': (0, 100),
            'energy_efficiency': (0, 1)
        }
        
        if obj_name in normalization_ranges:
            min_val, max_val = normalization_ranges[obj_name]
            
            if obj_name == 'surface_quality':
                # Lower is better for surface quality (roughness)
                normalized = 1.0 - (value - min_val) / (max_val - min_val)
            else:
                # Higher is better
                normalized = (value - min_val) / (max_val - min_val)
            
            return np.clip(normalized, 0.0, 1.0)
        else:
            # Default: assume value is already normalized
            return np.clip(value, 0.0, 1.0)
    
    def _generate_weight_combinations(self, num_objectives: int, num_combinations: int) -> List[List[float]]:
        """Generate diverse weight combinations for multi-objective optimization."""
        
        weight_combinations = []
        
        # Add corner solutions (single objective focus)
        for i in range(num_objectives):
            weights = [0.0] * num_objectives
            weights[i] = 1.0
            weight_combinations.append(weights)
        
        # Add random combinations
        for _ in range(num_combinations - num_objectives):
            weights = np.random.random(num_objectives)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            weight_combinations.append(weights.tolist())
        
        return weight_combinations
    
    def _extract_pareto_front(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """Extract Pareto optimal solutions from solution set."""
        
        if not solutions or 'objective_values' not in solutions[0]:
            return solutions
        
        pareto_front = []
        
        for i, solution in enumerate(solutions):
            is_dominated = False
            
            for j, other_solution in enumerate(solutions):
                if i == j:
                    continue
                
                # Check if solution i is dominated by solution j
                dominates = True
                at_least_one_better = False
                
                for obj in objectives:
                    if obj in solution['objective_values'] and obj in other_solution['objective_values']:
                        val_i = solution['objective_values'][obj]
                        val_j = other_solution['objective_values'][obj]
                        
                        # Assume higher is better (adjust normalization if needed)
                        if val_i > val_j:
                            dominates = False
                            break
                        elif val_j > val_i:
                            at_least_one_better = True
                
                if dominates and at_least_one_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        return pareto_front