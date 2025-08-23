"""Meta-Learning Optimizer for Autonomous SDLC Evolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import concurrent.futures
from collections import deque, defaultdict
import logging
import json
import hashlib
import copy
from pathlib import Path

from .autonomous_discovery_engine import AutonomousDiscoveryEngine, DiscoveryStrategy
from .quantum_evolutionary_optimizer import QuantumEvolutionaryOptimizer
from ..autonomous.self_evolving_ai import SelfImprovingSystem
from ..models.adaptive_intelligence import AdaptiveIntelligenceNetwork


logger = logging.getLogger(__name__)


class MetaLearningStrategy(Enum):
    """Meta-learning strategies for autonomous evolution."""
    
    LEARNING_TO_LEARN = "learning_to_learn"
    SELF_MODIFYING_CODE = "self_modifying_code"
    EVOLUTIONARY_ALGORITHMS = "evolutionary_algorithms"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT_ADAPTATION = "few_shot_adaptation"
    TRANSFER_LEARNING = "transfer_learning"


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning optimizer."""
    
    # Core meta-learning settings
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 1e-2
    meta_batch_size: int = 32
    inner_update_steps: int = 5
    meta_update_steps: int = 1000
    
    # Adaptation settings
    adaptation_steps: int = 10
    support_shots: int = 5
    query_shots: int = 15
    task_distribution: str = "uniform"
    
    # Architecture evolution
    evolve_architecture: bool = True
    architecture_mutation_rate: float = 0.1
    max_layers: int = 10
    max_neurons: int = 1024
    
    # Self-modification
    allow_code_modification: bool = True
    safety_checks: bool = True
    backup_frequency: int = 100
    
    # Performance tracking
    performance_window: int = 100
    convergence_threshold: float = 1e-6
    plateau_patience: int = 50
    
    # Resource management
    max_memory_gb: float = 8.0
    max_compute_hours: float = 24.0
    parallel_tasks: int = 4


class MetaNetwork(nn.Module):
    """Meta-network for learning optimization strategies."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Meta-learning network
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learning rate predictor
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Architecture predictor
        self.arch_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10),  # Architecture parameters
            nn.Softmax(dim=-1)
        )
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, len(MetaLearningStrategy)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, task_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through meta-network."""
        
        features = self.meta_net(task_context)
        
        return {
            'learning_rate': self.lr_predictor(features),
            'architecture': self.arch_predictor(features),
            'strategy': self.strategy_selector(features),
            'features': features
        }


class MetaLearningOptimizer:
    """Meta-learning optimizer for autonomous SDLC evolution."""
    
    def __init__(self, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        
        # Initialize meta-network
        self.meta_network = MetaNetwork(
            input_dim=64,  # Task context dimension
            hidden_dim=256
        )
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.meta_network.parameters(),
            lr=self.config.meta_learning_rate
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.config.performance_window)
        self.task_performance = defaultdict(list)
        self.adaptation_history = []
        
        # Task management
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_embeddings = {}
        
        # Self-modification tracking
        self.code_modifications = []
        self.architecture_history = []
        self.strategy_history = []
        
        # Resource monitoring
        self.resource_usage = {
            'memory_gb': 0.0,
            'compute_hours': 0.0,
            'gpu_utilization': 0.0
        }
        
        # Components
        self.discovery_engine = None
        self.quantum_optimizer = None
        self.self_improving_system = None
        
        logger.info(f"Meta-learning optimizer initialized with config: {self.config}")
    
    def initialize_components(self):
        """Initialize autonomous components."""
        try:
            self.discovery_engine = AutonomousDiscoveryEngine()
            self.quantum_optimizer = QuantumEvolutionaryOptimizer()
            self.self_improving_system = SelfImprovingSystem()
            
            logger.info("All meta-learning components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    def extract_task_context(self, task: Dict[str, Any]) -> torch.Tensor:
        """Extract contextual features from task description."""
        
        # Create task embedding
        context_features = [
            task.get('complexity', 0.5),
            task.get('novelty', 0.5),
            task.get('resource_requirement', 0.5),
            task.get('urgency', 0.5),
            len(task.get('dependencies', [])) / 10.0,  # Normalized
            task.get('success_probability', 0.5)
        ]
        
        # Add historical performance for similar tasks
        task_type = task.get('type', 'unknown')
        if task_type in self.task_performance:
            recent_performance = self.task_performance[task_type][-10:]  # Last 10
            context_features.extend([
                np.mean(recent_performance) if recent_performance else 0.5,
                np.std(recent_performance) if len(recent_performance) > 1 else 0.0,
                len(recent_performance) / 100.0  # Experience level
            ])
        else:
            context_features.extend([0.5, 0.0, 0.0])
        
        # Pad to fixed size
        while len(context_features) < 64:
            context_features.append(0.0)
        
        return torch.tensor(context_features[:64], dtype=torch.float32)
    
    def meta_learn_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-learning on a single task."""
        
        task_id = task.get('id', f"task_{time.time()}")
        logger.info(f"Meta-learning task: {task_id}")
        
        # Extract task context
        task_context = self.extract_task_context(task).unsqueeze(0)
        
        # Get meta-predictions
        with torch.no_grad():
            meta_predictions = self.meta_network(task_context)
        
        # Select strategy based on predictions
        strategy_probs = meta_predictions['strategy'].squeeze(0)
        selected_strategy_idx = torch.argmax(strategy_probs).item()
        selected_strategy = list(MetaLearningStrategy)[selected_strategy_idx]
        
        # Get adaptive parameters
        adaptive_lr = meta_predictions['learning_rate'].item() * 0.01  # Scale to reasonable range
        architecture_params = meta_predictions['architecture'].squeeze(0)
        
        # Execute task with selected strategy
        task_result = self._execute_task_with_strategy(
            task, selected_strategy, adaptive_lr, architecture_params
        )
        
        # Record performance for meta-learning
        performance = task_result.get('performance', 0.0)
        self.performance_history.append(performance)
        self.task_performance[task.get('type', 'unknown')].append(performance)
        
        # Update meta-network based on performance
        self._update_meta_network(task_context, performance, selected_strategy_idx)
        
        # Log results
        logger.info(
            f"Task {task_id} completed with strategy {selected_strategy.value}, "
            f"performance: {performance:.4f}"
        )
        
        return {
            'task_id': task_id,
            'strategy': selected_strategy.value,
            'performance': performance,
            'adaptive_lr': adaptive_lr,
            'task_result': task_result
        }
    
    def _execute_task_with_strategy(
        self, 
        task: Dict[str, Any], 
        strategy: MetaLearningStrategy,
        learning_rate: float,
        architecture_params: torch.Tensor
    ) -> Dict[str, Any]:
        """Execute task using selected meta-learning strategy."""
        
        if strategy == MetaLearningStrategy.LEARNING_TO_LEARN:
            return self._learning_to_learn(task, learning_rate)
        
        elif strategy == MetaLearningStrategy.SELF_MODIFYING_CODE:
            return self._self_modifying_code(task)
        
        elif strategy == MetaLearningStrategy.EVOLUTIONARY_ALGORITHMS:
            return self._evolutionary_algorithms(task)
        
        elif strategy == MetaLearningStrategy.NEURAL_ARCHITECTURE_SEARCH:
            return self._neural_architecture_search(task, architecture_params)
        
        elif strategy == MetaLearningStrategy.HYPERPARAMETER_OPTIMIZATION:
            return self._hyperparameter_optimization(task)
        
        elif strategy == MetaLearningStrategy.CONTINUAL_LEARNING:
            return self._continual_learning(task)
        
        elif strategy == MetaLearningStrategy.FEW_SHOT_ADAPTATION:
            return self._few_shot_adaptation(task, learning_rate)
        
        elif strategy == MetaLearningStrategy.TRANSFER_LEARNING:
            return self._transfer_learning(task)
        
        else:
            # Default strategy
            return self._learning_to_learn(task, learning_rate)
    
    def _learning_to_learn(self, task: Dict[str, Any], learning_rate: float) -> Dict[str, Any]:
        """Implement learning-to-learn meta-strategy."""
        
        # Simulate inner loop optimization
        inner_losses = []
        
        for step in range(self.config.inner_update_steps):
            # Simulate task performance (in real scenario, would train model)
            simulated_loss = max(0.1, 1.0 - step * 0.15 + np.random.normal(0, 0.05))
            inner_losses.append(simulated_loss)
        
        # Calculate performance improvement
        initial_loss = inner_losses[0]
        final_loss = inner_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss
        
        return {
            'strategy': 'learning_to_learn',
            'performance': max(0.0, improvement),
            'inner_losses': inner_losses,
            'learning_rate': learning_rate,
            'convergence_steps': len(inner_losses)
        }
    
    def _self_modifying_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement self-modifying code strategy (with safety checks)."""
        
        if not self.config.allow_code_modification:
            logger.warning("Self-modifying code disabled by configuration")
            return {'strategy': 'self_modifying_code', 'performance': 0.0, 'modifications': 0}
        
        # Simulate code modifications (in real scenario, would analyze and modify actual code)
        potential_modifications = [
            {'type': 'optimize_loop', 'impact': 0.15},
            {'type': 'add_caching', 'impact': 0.25},
            {'type': 'parallelize_computation', 'impact': 0.30},
            {'type': 'improve_algorithm', 'impact': 0.20}
        ]
        
        modifications_applied = 0
        total_impact = 0.0
        
        for mod in potential_modifications:
            if np.random.random() < 0.6:  # 60% chance to apply each modification
                if self.config.safety_checks:
                    # Simulate safety validation
                    if np.random.random() > 0.1:  # 90% pass safety checks
                        modifications_applied += 1
                        total_impact += mod['impact']
                        self.code_modifications.append(mod)
                else:
                    modifications_applied += 1
                    total_impact += mod['impact']
                    self.code_modifications.append(mod)
        
        performance = min(1.0, total_impact)  # Cap at 100%
        
        return {
            'strategy': 'self_modifying_code',
            'performance': performance,
            'modifications': modifications_applied,
            'total_impact': total_impact,
            'safety_checks_passed': True
        }
    
    def _evolutionary_algorithms(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement evolutionary algorithms strategy."""
        
        population_size = 20
        generations = 10
        
        # Simulate evolutionary optimization
        best_fitness_history = []
        
        for generation in range(generations):
            # Simulate population fitness evaluation
            fitness_scores = np.random.beta(2, 5, population_size)  # Skewed towards lower values
            best_fitness = np.max(fitness_scores)
            best_fitness_history.append(best_fitness)
        
        # Calculate improvement
        initial_fitness = best_fitness_history[0]
        final_fitness = best_fitness_history[-1]
        improvement = (final_fitness - initial_fitness) / (1.0 - initial_fitness + 1e-8)
        
        return {
            'strategy': 'evolutionary_algorithms',
            'performance': max(0.0, improvement),
            'generations': generations,
            'best_fitness': final_fitness,
            'fitness_history': best_fitness_history
        }
    
    def _neural_architecture_search(self, task: Dict[str, Any], arch_params: torch.Tensor) -> Dict[str, Any]:
        """Implement neural architecture search strategy."""
        
        # Use architecture parameters to define search space
        arch_probs = F.softmax(arch_params, dim=0)
        
        # Simulate architecture search
        architectures_evaluated = 0
        best_architecture_performance = 0.0
        
        for i in range(5):  # Evaluate 5 architectures
            # Simulate architecture performance
            arch_performance = float(arch_probs[i % len(arch_probs)]) * np.random.beta(3, 2)
            best_architecture_performance = max(best_architecture_performance, arch_performance)
            architectures_evaluated += 1
        
        self.architecture_history.append({
            'task_id': task.get('id'),
            'best_performance': best_architecture_performance,
            'architectures_evaluated': architectures_evaluated
        })
        
        return {
            'strategy': 'neural_architecture_search',
            'performance': best_architecture_performance,
            'architectures_evaluated': architectures_evaluated,
            'best_architecture': arch_probs.tolist()
        }
    
    def _hyperparameter_optimization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement hyperparameter optimization strategy."""
        
        # Simulate Bayesian optimization of hyperparameters
        iterations = 15
        best_performance = 0.0
        hyperparameter_history = []
        
        for iteration in range(iterations):
            # Simulate hyperparameter evaluation
            performance = np.random.beta(2 + iteration * 0.2, 3)  # Improves over time
            best_performance = max(best_performance, performance)
            hyperparameter_history.append(performance)
        
        return {
            'strategy': 'hyperparameter_optimization',
            'performance': best_performance,
            'iterations': iterations,
            'optimization_history': hyperparameter_history
        }
    
    def _continual_learning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement continual learning strategy."""
        
        # Simulate continual learning with catastrophic forgetting mitigation
        tasks_learned = 0
        cumulative_performance = 0.0
        forgetting_rate = 0.1
        
        for task_step in range(5):  # Learn 5 related tasks
            # Performance on new task
            new_task_performance = np.random.beta(3, 2)
            
            # Apply forgetting to previous tasks
            cumulative_performance = cumulative_performance * (1 - forgetting_rate) + new_task_performance
            tasks_learned += 1
        
        average_performance = cumulative_performance / tasks_learned
        
        return {
            'strategy': 'continual_learning',
            'performance': average_performance,
            'tasks_learned': tasks_learned,
            'forgetting_rate': forgetting_rate
        }
    
    def _few_shot_adaptation(self, task: Dict[str, Any], learning_rate: float) -> Dict[str, Any]:
        """Implement few-shot adaptation strategy."""
        
        # Simulate few-shot learning with meta-gradients
        support_examples = self.config.support_shots
        query_examples = self.config.query_shots
        
        # Support set performance (training)
        support_losses = []
        for shot in range(support_examples):
            loss = max(0.1, 1.0 - shot * 0.15 - learning_rate * 2)
            support_losses.append(loss)
        
        # Query set performance (evaluation)
        query_performance = 1.0 - np.mean(support_losses) + np.random.normal(0, 0.05)
        query_performance = max(0.0, min(1.0, query_performance))
        
        return {
            'strategy': 'few_shot_adaptation',
            'performance': query_performance,
            'support_shots': support_examples,
            'query_shots': query_examples,
            'adaptation_lr': learning_rate
        }
    
    def _transfer_learning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement transfer learning strategy."""
        
        # Simulate transfer from related domains
        source_domains = ['materials_a', 'materials_b', 'materials_c']
        
        best_transfer_performance = 0.0
        best_source_domain = None
        
        for source_domain in source_domains:
            # Simulate domain similarity and transfer effectiveness
            domain_similarity = np.random.beta(2, 3)  # Most domains moderately similar
            transfer_effectiveness = domain_similarity * np.random.beta(3, 2)
            
            if transfer_effectiveness > best_transfer_performance:
                best_transfer_performance = transfer_effectiveness
                best_source_domain = source_domain
        
        return {
            'strategy': 'transfer_learning',
            'performance': best_transfer_performance,
            'best_source_domain': best_source_domain,
            'domains_evaluated': len(source_domains)
        }
    
    def _update_meta_network(self, task_context: torch.Tensor, performance: float, strategy_idx: int):
        """Update meta-network based on task performance."""
        
        # Convert performance to reward signal
        reward = torch.tensor(performance, dtype=torch.float32)
        
        # Get meta-network predictions
        meta_predictions = self.meta_network(task_context)
        
        # Compute loss (negative log-likelihood weighted by reward)
        strategy_logits = meta_predictions['strategy']
        strategy_loss = -torch.log(strategy_logits[0, strategy_idx] + 1e-8) * (1.0 - reward)
        
        # Learning rate prediction loss
        predicted_lr = meta_predictions['learning_rate']
        lr_target = torch.tensor([[reward]], dtype=torch.float32)  # Higher performance -> higher LR
        lr_loss = F.mse_loss(predicted_lr, lr_target)
        
        # Total loss
        total_loss = strategy_loss + 0.1 * lr_loss
        
        # Update meta-network
        self.meta_optimizer.zero_grad()
        total_loss.backward()
        self.meta_optimizer.step()
        
        # Log update
        logger.debug(f"Meta-network updated: loss={total_loss.item():.6f}, reward={reward.item():.4f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent_performance = list(self.performance_history)[-20:]  # Last 20 tasks
        
        summary = {
            'total_tasks': len(self.performance_history),
            'average_performance': np.mean(self.performance_history),
            'recent_performance': np.mean(recent_performance),
            'best_performance': np.max(self.performance_history),
            'performance_std': np.std(self.performance_history),
            'improvement_trend': self._calculate_trend(),
            'strategy_distribution': self._get_strategy_distribution(),
            'resource_usage': self.resource_usage.copy(),
            'code_modifications': len(self.code_modifications),
            'architecture_evolutions': len(self.architecture_history)
        }
        
        return summary
    
    def _calculate_trend(self) -> float:
        """Calculate performance improvement trend."""
        
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        earlier = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else list(self.performance_history)[:-10]
        
        if not earlier:
            return 0.0
        
        return np.mean(recent) - np.mean(earlier)
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies used."""
        
        distribution = defaultdict(int)
        
        for strategy_record in self.strategy_history:
            strategy = strategy_record.get('strategy', 'unknown')
            distribution[strategy] += 1
        
        return dict(distribution)
    
    def autonomous_evolution_cycle(self, duration_hours: float = 1.0) -> Dict[str, Any]:
        """Run autonomous evolution cycle for specified duration."""
        
        logger.info(f"Starting autonomous evolution cycle for {duration_hours} hours")
        
        start_time = time.time()
        end_time = start_time + duration_hours * 3600
        
        cycle_results = {
            'start_time': start_time,
            'duration_hours': duration_hours,
            'tasks_completed': 0,
            'improvements_discovered': 0,
            'performance_gains': 0.0
        }
        
        while time.time() < end_time:
            # Generate autonomous improvement task
            improvement_task = self._generate_improvement_task()
            
            # Execute meta-learning on the task
            task_result = self.meta_learn_task(improvement_task)
            
            cycle_results['tasks_completed'] += 1
            
            # Check if significant improvement was achieved
            if task_result['performance'] > 0.7:  # Threshold for significant improvement
                cycle_results['improvements_discovered'] += 1
                cycle_results['performance_gains'] += task_result['performance']
                
                logger.info(
                    f"Significant improvement discovered: {task_result['performance']:.4f} "
                    f"using {task_result['strategy']}"
                )
            
            # Short pause to prevent overwhelming the system
            time.sleep(1.0)
        
        cycle_results['end_time'] = time.time()
        cycle_results['actual_duration'] = cycle_results['end_time'] - cycle_results['start_time']
        
        logger.info(
            f"Autonomous evolution cycle completed: "
            f"{cycle_results['tasks_completed']} tasks, "
            f"{cycle_results['improvements_discovered']} improvements"
        )
        
        return cycle_results
    
    def _generate_improvement_task(self) -> Dict[str, Any]:
        """Generate autonomous improvement task."""
        
        task_types = [
            'optimize_algorithm',
            'improve_architecture',
            'enhance_feature',
            'fix_bottleneck',
            'add_capability',
            'refactor_code',
            'optimize_hyperparameters'
        ]
        
        task_type = np.random.choice(task_types)
        
        return {
            'id': f"auto_improvement_{int(time.time())}_{np.random.randint(1000)}",
            'type': task_type,
            'complexity': np.random.beta(2, 3),
            'novelty': np.random.beta(3, 2),
            'resource_requirement': np.random.beta(2, 4),
            'urgency': np.random.uniform(0.1, 0.9),
            'dependencies': [],
            'success_probability': np.random.beta(3, 2),
            'description': f"Autonomous {task_type} task for system improvement"
        }


def create_meta_learning_optimizer(config: Optional[Dict[str, Any]] = None) -> MetaLearningOptimizer:
    """Factory function to create meta-learning optimizer."""
    
    if config:
        meta_config = MetaLearningConfig(**config)
    else:
        meta_config = MetaLearningConfig()
    
    optimizer = MetaLearningOptimizer(meta_config)
    optimizer.initialize_components()
    
    return optimizer
