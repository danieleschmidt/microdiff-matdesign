#!/usr/bin/env python3
"""Test Generation 5: Autonomous Self-Evolving Systems."""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pytest
import time
from typing import Dict, Any, List

# Import autonomous systems
from microdiff_matdesign.autonomous.self_evolving_ai import (
    SelfImprovingSystem,
    EvolvableNeuralModule,
    EvolutionaryOptimizer,
    MetaOptimizer,
    ArchitectureGenerator,
    AutonomousLearningScheduler,
    EvolutionConfig
)


class MockDataset(data.Dataset):
    """Mock dataset for testing autonomous systems."""
    
    def __init__(self, input_dim: int, output_dim: int, num_samples: int = 1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate synthetic data with some structure
        self.inputs = torch.randn(num_samples, input_dim)
        
        # Create target mapping with some noise
        weight_matrix = torch.randn(input_dim, output_dim) * 0.5
        self.targets = torch.matmul(self.inputs, weight_matrix) + torch.randn(num_samples, output_dim) * 0.1
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TestEvolvableNeuralModule:
    """Test evolvable neural module capabilities."""
    
    def test_evolvable_module_creation(self):
        """Test creation and basic functionality of evolvable modules."""
        print("\n🧬 Testing Evolvable Neural Module Creation...")
        
        input_dim = 32
        output_dim = 16
        
        # Create random module
        module = EvolvableNeuralModule(input_dim, output_dim)
        
        assert hasattr(module, 'genome'), "Module should have genome"
        assert hasattr(module, 'architecture'), "Module should have architecture"
        assert module.input_dim == input_dim
        assert module.output_dim == output_dim
        
        # Test forward pass
        x = torch.randn(4, input_dim)
        output = module(x)
        
        assert output.shape == (4, output_dim), f"Output shape mismatch: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        
        print(f"✅ Evolvable module: {input_dim}→{output_dim}, genome keys: {list(module.genome.keys())}")
    
    def test_mutation(self):
        """Test neural module mutation."""
        print("\n🔬 Testing Neural Module Mutation...")
        
        input_dim = 24
        output_dim = 12
        
        parent = EvolvableNeuralModule(input_dim, output_dim)
        original_genome = parent.genome.copy()
        
        # Create mutated offspring
        offspring = parent.mutate(mutation_rate=0.5)
        
        assert offspring.input_dim == parent.input_dim
        assert offspring.output_dim == parent.output_dim
        assert offspring.generation == parent.generation + 1
        
        # Check that some genes mutated
        mutations_found = 0
        for key in original_genome.keys():
            if offspring.genome[key] != original_genome[key]:
                mutations_found += 1
        
        assert mutations_found > 0, "No mutations occurred with 50% mutation rate"
        
        print(f"✅ Mutation: {mutations_found} genes mutated, generation {offspring.generation}")
    
    def test_crossover(self):
        """Test neural module crossover."""
        print("\n🧪 Testing Neural Module Crossover...")
        
        input_dim = 20
        output_dim = 10
        
        parent1 = EvolvableNeuralModule(input_dim, output_dim)
        parent2 = EvolvableNeuralModule(input_dim, output_dim)
        
        # Perform crossover
        child1, child2 = parent1.crossover(parent2)
        
        assert child1.input_dim == input_dim
        assert child2.input_dim == input_dim
        assert child1.output_dim == output_dim
        assert child2.output_dim == output_dim
        
        # Children should be different from parents
        assert child1.genome != parent1.genome or child1.genome != parent2.genome
        assert child2.genome != parent1.genome or child2.genome != parent2.genome
        
        print(f"✅ Crossover: Created children with generation {child1.generation}")
    
    def test_fitness_computation(self):
        """Test fitness computation."""
        print("\n📈 Testing Fitness Computation...")
        
        module = EvolvableNeuralModule(16, 8)
        
        # Test fitness computation
        performance = {
            'accuracy': 0.85,
            'efficiency': 0.70,
            'novelty': 0.60,
            'robustness': 0.75
        }
        
        fitness = module.compute_fitness(performance)
        
        assert 0.0 <= fitness <= 1.0, f"Fitness out of range: {fitness}"
        assert len(module.fitness_history) == 1
        
        # Test multiple fitness evaluations
        for i in range(5):
            perf = {k: v + 0.01 * i for k, v in performance.items()}
            module.compute_fitness(perf)
        
        assert len(module.fitness_history) == 6
        
        print(f"✅ Fitness: {fitness:.4f}, history length: {len(module.fitness_history)}")


class TestEvolutionaryOptimizer:
    """Test evolutionary optimization capabilities."""
    
    def test_evolutionary_optimizer_creation(self):
        """Test creation of evolutionary optimizer."""
        print("\n🌿 Testing Evolutionary Optimizer Creation...")
        
        input_dim = 28
        output_dim = 14
        config = EvolutionConfig(population_size=10, max_generations=5)
        
        optimizer = EvolutionaryOptimizer(input_dim, output_dim, config)
        
        assert len(optimizer.population) == config.population_size
        assert optimizer.generation == 0
        assert optimizer.best_individual is None
        
        # Check that all individuals are properly initialized
        for individual in optimizer.population:
            assert individual.input_dim == input_dim
            assert individual.output_dim == output_dim
            assert hasattr(individual, 'genome')
        
        print(f"✅ Evolutionary optimizer: Population size {len(optimizer.population)}")
    
    def test_evolution_process(self):
        """Test evolutionary optimization process."""
        print("\n🧬 Testing Evolution Process...")
        
        input_dim = 16
        output_dim = 8
        config = EvolutionConfig(population_size=8, max_generations=3)
        
        optimizer = EvolutionaryOptimizer(input_dim, output_dim, config)
        
        # Simple fitness evaluator
        def fitness_evaluator(individual):
            # Simulate performance evaluation
            complexity_penalty = individual.get_complexity() / 1000
            return {
                'accuracy': max(0.1, 0.9 - complexity_penalty),
                'efficiency': max(0.1, 0.8 - complexity_penalty * 0.5),
                'novelty': np.random.uniform(0.3, 0.7),
                'robustness': max(0.1, 0.7 - complexity_penalty * 0.3)
            }
        
        start_time = time.time()
        best_individual = optimizer.evolve(fitness_evaluator, max_generations=3)
        evolution_time = time.time() - start_time
        
        assert best_individual is not None
        assert len(best_individual.fitness_history) > 0
        assert len(optimizer.evolution_history) <= 3
        assert evolution_time < 10.0, f"Evolution too slow: {evolution_time:.3f}s"
        
        # Check evolution history
        assert len(optimizer.evolution_history) > 0
        first_gen = optimizer.evolution_history[0]
        assert 'best_fitness' in first_gen
        assert 'avg_fitness' in first_gen
        assert 'diversity' in first_gen
        
        print(f"✅ Evolution: {len(optimizer.evolution_history)} generations, best fitness: {first_gen['best_fitness']:.4f}")


class TestSelfImprovingSystem:
    """Test self-improving AI system."""
    
    def test_self_improving_system_creation(self):
        """Test creation of self-improving system."""
        print("\n🚀 Testing Self-Improving System Creation...")
        
        input_dim = 20
        output_dim = 10
        
        system = SelfImprovingSystem(input_dim, output_dim)
        
        assert hasattr(system, 'evolutionary_optimizer')
        assert hasattr(system, 'meta_optimizer')
        assert hasattr(system, 'architecture_generator')
        assert hasattr(system, 'learning_scheduler')
        assert system.current_model is None  # Initially no model
        
        print(f"✅ Self-improving system: {input_dim}→{output_dim} dimensions")
    
    def test_self_improvement_cycle(self):
        """Test autonomous self-improvement cycle."""
        print("\n🔄 Testing Self-Improvement Cycle...")
        
        input_dim = 12
        output_dim = 6
        
        system = SelfImprovingSystem(input_dim, output_dim)
        
        # Create mock datasets
        train_dataset = MockDataset(input_dim, output_dim, 200)
        val_dataset = MockDataset(input_dim, output_dim, 50)
        
        train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Run limited self-improvement
        start_time = time.time()
        result = system.self_improve(train_loader, val_loader, improvement_budget=2)
        improvement_time = time.time() - start_time
        
        assert 'improvement_log' in result
        assert 'final_performance' in result
        assert 'total_cycles' in result
        assert result['total_cycles'] == 2
        assert improvement_time < 30.0, f"Self-improvement too slow: {improvement_time:.3f}s"
        
        # System should have a model after improvement
        assert system.current_model is not None
        
        print(f"✅ Self-improvement: {len(result['improvement_log'])} cycles, {improvement_time:.3f}s")
    
    def test_performance_evaluation(self):
        """Test performance evaluation."""
        print("\n📊 Testing Performance Evaluation...")
        
        input_dim = 8
        output_dim = 4
        
        system = SelfImprovingSystem(input_dim, output_dim)
        
        # Create a test model
        test_model = EvolvableNeuralModule(input_dim, output_dim)
        system.current_model = test_model
        
        # Create validation data
        val_dataset = MockDataset(input_dim, output_dim, 100)
        val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Evaluate performance
        performance = system._evaluate_performance(val_loader)
        
        assert 'accuracy' in performance
        assert 'efficiency' in performance
        assert 'novelty' in performance
        assert 'robustness' in performance
        assert 'overall' in performance
        
        for metric_value in performance.values():
            assert 0.0 <= metric_value <= 1.0, f"Metric out of range: {metric_value}"
        
        print(f"✅ Performance evaluation: Overall={performance['overall']:.4f}")


class TestMetaOptimizer:
    """Test meta-optimization capabilities."""
    
    def test_meta_optimizer_creation(self):
        """Test meta-optimizer creation."""
        print("\n🎯 Testing Meta-Optimizer Creation...")
        
        input_dim = 16
        meta_optimizer = MetaOptimizer(input_dim)
        
        assert hasattr(meta_optimizer, 'strategy_generator')
        assert hasattr(meta_optimizer, 'strategy_decoder')
        
        print(f"✅ Meta-optimizer: {input_dim} input dimensions")
    
    def test_improvement_strategy_generation(self):
        """Test improvement strategy generation."""
        print("\n🧠 Testing Improvement Strategy Generation...")
        
        input_dim = 12
        meta_optimizer = MetaOptimizer(input_dim)
        
        # Mock performance data
        current_performance = {
            'accuracy': 0.75,
            'efficiency': 0.80,
            'novelty': 0.65,
            'robustness': 0.70
        }
        
        from collections import deque
        performance_history = deque([
            {'overall': 0.70}, {'overall': 0.72}, {'overall': 0.74}
        ], maxlen=1000)
        
        # Generate strategy
        strategy = meta_optimizer.generate_improvement_strategy(current_performance, performance_history)
        
        assert 'type' in strategy
        assert strategy['type'] in ['evolutionary_search', 'architecture_generation', 'meta_optimization']
        
        print(f"✅ Strategy generation: {strategy['type']} strategy")


class TestArchitectureGenerator:
    """Test architecture generation capabilities."""
    
    def test_architecture_generator_creation(self):
        """Test architecture generator creation."""
        print("\n🏗️ Testing Architecture Generator Creation...")
        
        input_dim = 20
        output_dim = 10
        
        generator = ArchitectureGenerator(input_dim, output_dim)
        
        assert hasattr(generator, 'encoder')
        assert hasattr(generator, 'decoder')
        assert generator.input_dim == input_dim
        assert generator.output_dim == output_dim
        
        print(f"✅ Architecture generator: {input_dim}→{output_dim}")
    
    def test_architecture_generation(self):
        """Test novel architecture generation."""
        print("\n🎨 Testing Architecture Generation...")
        
        input_dim = 16
        output_dim = 8
        
        generator = ArchitectureGenerator(input_dim, output_dim)
        
        # Generate architecture with different creativity levels
        conservative_arch = generator.generate_architecture(None, creativity_level=0.1)
        creative_arch = generator.generate_architecture(None, creativity_level=0.9)
        
        assert conservative_arch.input_dim == input_dim
        assert conservative_arch.output_dim == output_dim
        assert creative_arch.input_dim == input_dim
        assert creative_arch.output_dim == output_dim
        
        # Test that architectures are different
        assert conservative_arch.genome != creative_arch.genome
        
        print(f"✅ Architecture generation: Conservative vs Creative architectures")


class TestAutonomousLearningScheduler:
    """Test autonomous learning scheduler."""
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        print("\n⏰ Testing Autonomous Learning Scheduler...")
        
        scheduler = AutonomousLearningScheduler()
        
        assert hasattr(scheduler, 'active_tasks')
        assert hasattr(scheduler, 'task_history')
        assert not scheduler.running
        
        print(f"✅ Scheduler: Ready for autonomous learning")
    
    def test_scheduler_lifecycle(self):
        """Test scheduler start/stop lifecycle."""
        print("\n🔄 Testing Scheduler Lifecycle...")
        
        scheduler = AutonomousLearningScheduler()
        
        # Create minimal system for testing
        input_dim = 8
        output_dim = 4
        system = SelfImprovingSystem(input_dim, output_dim)
        
        # Create minimal datasets
        train_dataset = MockDataset(input_dim, output_dim, 50)
        val_dataset = MockDataset(input_dim, output_dim, 20)
        train_loader = data.DataLoader(train_dataset, batch_size=8)
        val_loader = data.DataLoader(val_dataset, batch_size=8)
        
        # Start autonomous learning (will run in background)
        scheduler.start_autonomous_learning(system, train_loader, val_loader)
        assert scheduler.running
        
        # Let it run briefly
        time.sleep(1.0)
        
        # Stop autonomous learning
        scheduler.stop_autonomous_learning()
        assert not scheduler.running
        
        print(f"✅ Scheduler lifecycle: Start/stop successful")


class TestAutonomousIntegration:
    """Test integration of autonomous systems."""
    
    def test_autonomous_system_integration(self):
        """Test complete autonomous system integration."""
        print("\n🌐 Testing Autonomous System Integration...")
        
        input_dim = 10
        output_dim = 5
        
        # Create integrated autonomous system
        system = SelfImprovingSystem(input_dim, output_dim)
        
        # Verify all components are integrated
        assert system.evolutionary_optimizer is not None
        assert system.meta_optimizer is not None
        assert system.architecture_generator is not None
        assert system.learning_scheduler is not None
        
        # Test component interactions
        test_model = EvolvableNeuralModule(input_dim, output_dim)
        
        # Test architecture generation
        new_arch = system.architecture_generator.generate_architecture(test_model, 0.5)
        assert new_arch.input_dim == input_dim
        assert new_arch.output_dim == output_dim
        
        # Test meta-optimization strategy
        performance = {'accuracy': 0.8, 'efficiency': 0.7, 'novelty': 0.6, 'robustness': 0.75}
        strategy = system.meta_optimizer.generate_improvement_strategy(performance, system.performance_history)
        assert 'type' in strategy
        
        print(f"✅ Autonomous integration: All components operational")
    
    def test_autonomous_performance_benchmarks(self):
        """Test performance benchmarks for autonomous systems."""
        print("\n📊 Testing Autonomous Performance Benchmarks...")
        
        input_dim = 12
        output_dim = 6
        
        # Test different autonomous components
        components = {
            'evolvable_module': EvolvableNeuralModule(input_dim, output_dim),
            'meta_optimizer': MetaOptimizer(input_dim),
            'arch_generator': ArchitectureGenerator(input_dim, output_dim),
            'self_improving': SelfImprovingSystem(input_dim, output_dim)
        }
        
        performance_results = {}
        
        for name, component in components.items():
            start_time = time.time()
            
            if name == 'evolvable_module':
                x = torch.randn(8, input_dim)
                output = component(x)
                
            elif name == 'meta_optimizer':
                perf = {'accuracy': 0.8, 'efficiency': 0.7, 'novelty': 0.6, 'robustness': 0.75}
                from collections import deque
                hist = deque([{'overall': 0.7}])
                strategy = component.generate_improvement_strategy(perf, hist)
                
            elif name == 'arch_generator':
                new_arch = component.generate_architecture(None, 0.5)
                
            else:  # self_improving
                # Just test creation time
                pass
            
            component_time = time.time() - start_time
            performance_results[name] = component_time
        
        print(f"✅ Performance benchmarks:")
        for name, comp_time in performance_results.items():
            print(f"   {name}: {comp_time:.4f}s")
            assert comp_time < 2.0, f"{name} too slow: {comp_time:.3f}s"


def run_generation5_tests():
    """Run all Generation 5 autonomous tests."""
    print("🤖 GENERATION 5: AUTONOMOUS SELF-EVOLVING SYSTEMS TESTING")
    print("=" * 70)
    
    # Test evolvable neural modules
    module_tests = TestEvolvableNeuralModule()
    module_tests.test_evolvable_module_creation()
    module_tests.test_mutation()
    module_tests.test_crossover()
    module_tests.test_fitness_computation()
    
    # Test evolutionary optimization
    evolution_tests = TestEvolutionaryOptimizer()
    evolution_tests.test_evolutionary_optimizer_creation()
    evolution_tests.test_evolution_process()
    
    # Test self-improving systems
    improvement_tests = TestSelfImprovingSystem()
    improvement_tests.test_self_improving_system_creation()
    improvement_tests.test_self_improvement_cycle()
    improvement_tests.test_performance_evaluation()
    
    # Test meta-optimization
    meta_tests = TestMetaOptimizer()
    meta_tests.test_meta_optimizer_creation()
    meta_tests.test_improvement_strategy_generation()
    
    # Test architecture generation
    arch_tests = TestArchitectureGenerator()
    arch_tests.test_architecture_generator_creation()
    arch_tests.test_architecture_generation()
    
    # Test autonomous learning
    scheduler_tests = TestAutonomousLearningScheduler()
    scheduler_tests.test_scheduler_creation()
    scheduler_tests.test_scheduler_lifecycle()
    
    # Test integration
    integration_tests = TestAutonomousIntegration()
    integration_tests.test_autonomous_system_integration()
    integration_tests.test_autonomous_performance_benchmarks()
    
    print("\n" + "=" * 70)
    print("🎉 GENERATION 5 AUTONOMOUS TESTS COMPLETED SUCCESSFULLY!")
    print("✅ Self-Evolving Neural Modules: Operational")
    print("✅ Evolutionary Optimization: Operational")
    print("✅ Self-Improving Systems: Operational")
    print("✅ Meta-Optimization: Operational")
    print("✅ Architecture Generation: Operational")
    print("✅ Autonomous Learning Scheduler: Operational")
    print("✅ Full Autonomous Integration: Successful")
    print("=" * 70)


if __name__ == "__main__":
    run_generation5_tests()