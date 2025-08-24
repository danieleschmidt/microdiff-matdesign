"""Adaptive Learning System for Continuous Self-Improvement."""

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
import copy
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

from .cognitive_architecture import CognitiveArchitecture, CognitiveState, ProcessingMode
from ..evolution.meta_learning_optimizer import MetaLearningOptimizer, MetaLearningConfig
from .self_evolving_ai import SelfImprovingSystem, EvolutionConfig


logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for adaptive system."""
    
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    SELF_SUPERVISED = "self_supervised"
    META_LEARNING = "meta_learning"
    CONTINUAL = "continual"
    TRANSFER = "transfer"
    CURRICULUM = "curriculum"


class AdaptationStrategy(Enum):
    """Strategies for adapting to new situations."""
    
    CONSERVATIVE = "conservative"    # Small, safe adaptations
    MODERATE = "moderate"           # Balanced adaptation
    AGGRESSIVE = "aggressive"       # Large, bold adaptations
    DYNAMIC = "dynamic"             # Contextually determined
    EXPERIMENTAL = "experimental"   # Highly exploratory


@dataclass
class LearningContext:
    """Context for learning operations."""
    
    # Task information
    task_type: str = "unknown"
    task_complexity: float = 0.5
    task_novelty: float = 0.5
    
    # Environment information
    environment_stability: float = 0.5
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    time_pressure: float = 0.5
    
    # Learning history
    previous_performance: List[float] = field(default_factory=list)
    learning_trajectory: List[Dict] = field(default_factory=list)
    adaptation_history: List[Dict] = field(default_factory=list)
    
    # Meta-information
    confidence_in_context: float = 0.5
    context_similarity_to_past: float = 0.5
    expected_learning_difficulty: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'task_type': self.task_type,
            'task_complexity': self.task_complexity,
            'task_novelty': self.task_novelty,
            'environment_stability': self.environment_stability,
            'resource_constraints': self.resource_constraints.copy(),
            'time_pressure': self.time_pressure,
            'previous_performance': self.previous_performance.copy(),
            'confidence_in_context': self.confidence_in_context,
            'context_similarity_to_past': self.context_similarity_to_past,
            'expected_learning_difficulty': self.expected_learning_difficulty
        }


class LearningModule(ABC):
    """Abstract base class for learning modules."""
    
    def __init__(self, module_name: str, learning_mode: LearningMode):
        self.module_name = module_name
        self.learning_mode = learning_mode
        self.performance_history = deque(maxlen=1000)
        self.adaptation_count = 0
        self.last_adaptation_time = time.time()
        self.is_active = True
        
    @abstractmethod
    def learn(self, data: Any, context: LearningContext) -> Dict[str, Any]:
        """Perform learning operation."""
        pass
    
    @abstractmethod
    def adapt(self, feedback: Dict[str, Any], context: LearningContext) -> bool:
        """Adapt based on feedback."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this module."""
        if not self.performance_history:
            return {'performance': 0.0, 'stability': 0.0, 'adaptations': self.adaptation_count}
        
        recent_performance = list(self.performance_history)[-50:]  # Last 50 entries
        
        return {
            'average_performance': np.mean(self.performance_history),
            'recent_performance': np.mean(recent_performance),
            'performance_std': np.std(self.performance_history),
            'adaptation_count': self.adaptation_count,
            'adaptation_frequency': self.adaptation_count / max(1, (time.time() - self.last_adaptation_time) / 3600),  # per hour
            'performance_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> float:
        """Calculate performance trend (positive = improving, negative = declining)."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        earlier = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else list(self.performance_history)[:-10]
        
        if not earlier:
            return 0.0
        
        return np.mean(recent) - np.mean(earlier)


class SelfSupervisedLearningModule(LearningModule):
    """Self-supervised learning module for autonomous improvement."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__("self_supervised", LearningMode.SELF_SUPERVISED)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Self-supervised networks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Optimizer for self-supervised learning
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.contrastive_head.parameters()),
            lr=1e-4
        )
        
        # Memory bank for contrastive learning
        self.memory_bank = deque(maxlen=1000)
        self.positive_pairs = deque(maxlen=500)
        
    def learn(self, data: torch.Tensor, context: LearningContext) -> Dict[str, Any]:
        """Perform self-supervised learning."""
        
        # Create augmented versions of data for self-supervision
        augmented_data = self._augment_data(data)
        
        # Encode original and augmented data
        encoded_original = self.encoder(data)
        encoded_augmented = self.encoder(augmented_data)
        
        # Reconstruction loss
        reconstructed = self.decoder(encoded_original)
        reconstruction_loss = F.mse_loss(reconstructed, data)
        
        # Contrastive loss
        contrastive_original = self.contrastive_head(encoded_original)
        contrastive_augmented = self.contrastive_head(encoded_augmented)
        contrastive_loss = self._compute_contrastive_loss(contrastive_original, contrastive_augmented)
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * contrastive_loss
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Record performance
        performance = 1.0 / (1.0 + float(total_loss))  # Convert loss to performance metric
        self.performance_history.append(performance)
        
        # Update memory bank
        self.memory_bank.append(encoded_original.detach())
        self.positive_pairs.append((contrastive_original.detach(), contrastive_augmented.detach()))
        
        return {
            'reconstruction_loss': float(reconstruction_loss),
            'contrastive_loss': float(contrastive_loss),
            'total_loss': float(total_loss),
            'performance': performance,
            'encoded_representation': encoded_original.detach()
        }
    
    def adapt(self, feedback: Dict[str, Any], context: LearningContext) -> bool:
        """Adapt learning strategy based on feedback."""
        
        performance = feedback.get('performance', 0.5)
        target_performance = feedback.get('target_performance', 0.8)
        
        # Determine if adaptation is needed
        performance_gap = target_performance - performance
        
        if performance_gap > 0.1:  # Significant performance gap
            # Adjust learning rate
            new_lr = self.optimizer.param_groups[0]['lr'] * 1.1  # Increase learning rate
            new_lr = min(new_lr, 1e-2)  # Cap at reasonable maximum
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.adaptation_count += 1
            self.last_adaptation_time = time.time()
            
            logger.info(f"Self-supervised module adapted: new_lr={new_lr:.6f}")
            return True
        
        elif performance_gap < -0.05:  # Performance is too high, might be overfitting
            # Decrease learning rate
            new_lr = self.optimizer.param_groups[0]['lr'] * 0.9
            new_lr = max(new_lr, 1e-6)  # Floor at reasonable minimum
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.adaptation_count += 1
            self.last_adaptation_time = time.time()
            
            logger.info(f"Self-supervised module adapted (reduce overfitting): new_lr={new_lr:.6f}")
            return True
        
        return False  # No adaptation needed
    
    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        """Create augmented version of data for self-supervised learning."""
        
        # Add noise augmentation
        noise = torch.randn_like(data) * 0.1
        augmented = data + noise
        
        # Add dropout-like masking
        mask = torch.rand_like(data) > 0.1  # Keep 90% of features
        augmented = augmented * mask.float()
        
        return augmented
    
    def _compute_contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Compute contrastive loss for self-supervised learning."""
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        
        # Compute similarity
        similarity = torch.mm(anchor, positive.t()) / temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(anchor.size(0), device=anchor.device)
        
        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class ContinualLearningModule(LearningModule):
    """Continual learning module to prevent catastrophic forgetting."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_tasks: int = 10):
        super().__init__("continual", LearningMode.CONTINUAL)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        
        # Task-specific networks
        self.task_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_tasks)
        ])
        
        # Shared network
        self.shared_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks),
            nn.Softmax(dim=-1)
        )
        
        # Memory buffer for rehearsal
        self.memory_buffer = deque(maxlen=1000)
        self.task_memories = defaultdict(lambda: deque(maxlen=100))
        
        # Optimizers
        self.task_optimizer = torch.optim.AdamW(self.task_networks.parameters(), lr=1e-3)
        self.shared_optimizer = torch.optim.AdamW(self.shared_network.parameters(), lr=5e-4)
        self.classifier_optimizer = torch.optim.AdamW(self.task_classifier.parameters(), lr=1e-3)
        
        self.current_task = 0
        
    def learn(self, data: torch.Tensor, context: LearningContext) -> Dict[str, Any]:
        """Perform continual learning with catastrophic forgetting mitigation."""
        
        # Identify task
        task_probs = self.task_classifier(data)
        predicted_task = torch.argmax(task_probs, dim=1)
        
        # Learn on current task
        task_features = self.task_networks[self.current_task](data)
        shared_features = self.shared_network(task_features)
        
        # Store in memory for rehearsal
        self.memory_buffer.append((data.detach(), self.current_task))
        self.task_memories[self.current_task].append(data.detach())
        
        # Rehearsal learning on previous tasks
        rehearsal_loss = 0.0
        if len(self.memory_buffer) > 10:
            # Sample from memory buffer
            rehearsal_samples = list(self.memory_buffer)[-10:]  # Last 10 samples
            
            for rehearsal_data, task_id in rehearsal_samples:
                if task_id != self.current_task:  # Don't rehearse current task
                    rehearsal_features = self.task_networks[task_id](rehearsal_data)
                    rehearsal_shared = self.shared_network(rehearsal_features)
                    
                    # Compute rehearsal loss (preserve previous knowledge)
                    target_features = rehearsal_shared.detach()  # Target is current representation
                    current_features = self.shared_network(self.task_networks[task_id](rehearsal_data))
                    rehearsal_loss += F.mse_loss(current_features, target_features)
        
        # Task classification loss
        task_labels = torch.full((data.size(0),), self.current_task, dtype=torch.long, device=data.device)
        classification_loss = F.cross_entropy(task_probs, task_labels)
        
        # Total loss
        total_loss = rehearsal_loss + 0.1 * classification_loss
        
        # Update networks
        if total_loss > 0:
            self.task_optimizer.zero_grad()
            self.shared_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            
            total_loss.backward()
            
            self.task_optimizer.step()
            self.shared_optimizer.step()
            self.classifier_optimizer.step()
        
        # Calculate performance
        performance = 1.0 / (1.0 + float(total_loss)) if total_loss > 0 else 0.8
        self.performance_history.append(performance)
        
        return {
            'rehearsal_loss': float(rehearsal_loss) if rehearsal_loss else 0.0,
            'classification_loss': float(classification_loss),
            'total_loss': float(total_loss) if total_loss > 0 else 0.0,
            'performance': performance,
            'predicted_task': predicted_task.tolist(),
            'current_task': self.current_task
        }
    
    def adapt(self, feedback: Dict[str, Any], context: LearningContext) -> bool:
        """Adapt continual learning strategy."""
        
        task_switch_signal = feedback.get('task_switch', False)
        performance = feedback.get('performance', 0.5)
        
        # Check if we should switch to a new task
        if task_switch_signal or performance > 0.9:  # High performance suggests mastery
            old_task = self.current_task
            self.current_task = (self.current_task + 1) % self.num_tasks
            
            self.adaptation_count += 1
            self.last_adaptation_time = time.time()
            
            logger.info(f"Continual learning task switch: {old_task} -> {self.current_task}")
            return True
        
        # Adjust learning rates based on forgetting
        if len(self.performance_history) > 20:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            earlier_performance = np.mean(list(self.performance_history)[-20:-10])
            
            if recent_performance < earlier_performance - 0.1:  # Significant forgetting
                # Increase rehearsal by adjusting learning rates
                for param_group in self.shared_optimizer.param_groups:
                    param_group['lr'] *= 0.8  # Reduce shared network learning rate
                
                self.adaptation_count += 1
                logger.info(f"Continual learning adapted to reduce forgetting")
                return True
        
        return False


class AdaptiveLearningSystem:
    """Main adaptive learning system coordinating all learning modules."""
    
    def __init__(self, input_dim: int = 512, config: Optional[Dict[str, Any]] = None):
        self.input_dim = input_dim
        self.config = config or {}
        
        # Initialize learning modules
        self.learning_modules = {
            'self_supervised': SelfSupervisedLearningModule(input_dim),
            'continual': ContinualLearningModule(input_dim)
        }
        
        # Meta-learning optimizer
        self.meta_optimizer = MetaLearningOptimizer()
        
        # Cognitive architecture for decision making
        self.cognitive_architecture = CognitiveArchitecture()
        
        # Adaptation strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(64, 128),  # Context features
            nn.ReLU(),
            nn.Linear(128, len(AdaptationStrategy)),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking
        self.system_performance = deque(maxlen=1000)
        self.adaptation_history = []
        self.learning_contexts = deque(maxlen=500)
        
        # Resource monitoring
        self.resource_usage = {
            'memory_mb': 0.0,
            'compute_time': 0.0,
            'adaptation_frequency': 0.0
        }
        
        # Thread pool for parallel learning
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        logger.info(f"Adaptive learning system initialized with {len(self.learning_modules)} modules")
    
    def adaptive_learn(self, data: torch.Tensor, context: Optional[LearningContext] = None) -> Dict[str, Any]:
        """Perform adaptive learning across all modules."""
        
        start_time = time.time()
        
        # Create learning context if not provided
        if context is None:
            context = self._create_learning_context(data)
        
        # Select adaptation strategy
        adaptation_strategy = self._select_adaptation_strategy(context)
        
        # Learn with all active modules
        learning_results = {}
        futures = {}
        
        for module_name, module in self.learning_modules.items():
            if module.is_active:
                # Submit learning task to thread pool
                future = self.executor.submit(module.learn, data, context)
                futures[module_name] = future
        
        # Collect results
        for module_name, future in futures.items():
            try:
                result = future.result(timeout=30)  # 30 second timeout
                learning_results[module_name] = result
            except concurrent.futures.TimeoutError:
                logger.warning(f"Learning module {module_name} timed out")
                learning_results[module_name] = {'error': 'timeout'}
            except Exception as e:
                logger.error(f"Learning module {module_name} failed: {e}")
                learning_results[module_name] = {'error': str(e)}
        
        # Integrate learning results
        integrated_performance = self._integrate_learning_results(learning_results)
        
        # Record system performance
        self.system_performance.append(integrated_performance)
        
        # Record context
        self.learning_contexts.append(context)
        
        # Update resource usage
        learning_time = time.time() - start_time
        self.resource_usage['compute_time'] += learning_time
        
        # Determine if adaptation is needed
        adaptation_needed = self._should_adapt(learning_results, context)
        
        result = {
            'learning_results': learning_results,
            'integrated_performance': integrated_performance,
            'adaptation_strategy': adaptation_strategy.value,
            'learning_time': learning_time,
            'context': context.to_dict(),
            'adaptation_needed': adaptation_needed
        }
        
        # Perform adaptation if needed
        if adaptation_needed:
            adaptation_result = self._perform_adaptation(learning_results, context, adaptation_strategy)
            result['adaptation_result'] = adaptation_result
        
        return result
    
    def _create_learning_context(self, data: torch.Tensor) -> LearningContext:
        """Create learning context from data and system state."""
        
        # Analyze data characteristics
        data_complexity = float(torch.std(data))  # Use std as complexity measure
        data_novelty = self._estimate_novelty(data)
        
        # Analyze system state
        recent_performance = list(self.system_performance)[-10:] if self.system_performance else [0.5]
        
        context = LearningContext(
            task_type="materials_design",
            task_complexity=min(1.0, data_complexity),
            task_novelty=data_novelty,
            environment_stability=0.8,  # Assume stable for materials design
            time_pressure=0.3,  # Low time pressure for research
            previous_performance=recent_performance,
            confidence_in_context=0.7,
            context_similarity_to_past=self._estimate_context_similarity(),
            expected_learning_difficulty=0.5
        )
        
        return context
    
    def _estimate_novelty(self, data: torch.Tensor) -> float:
        """Estimate novelty of input data."""
        
        if not self.learning_contexts:
            return 0.5  # Medium novelty for first sample
        
        # Compare with recent data (simplified)
        # In real implementation, would use more sophisticated novelty detection
        recent_contexts = list(self.learning_contexts)[-5:]
        
        # For now, use random novelty (would be replaced with actual comparison)
        return np.random.beta(2, 3)  # Skewed towards lower novelty
    
    def _estimate_context_similarity(self) -> float:
        """Estimate similarity to past contexts."""
        
        if len(self.learning_contexts) < 2:
            return 0.5
        
        # Simplified similarity estimation
        return np.random.beta(3, 2)  # Skewed towards higher similarity
    
    def _select_adaptation_strategy(self, context: LearningContext) -> AdaptationStrategy:
        """Select adaptation strategy based on context."""
        
        # Create context feature vector
        context_features = [
            context.task_complexity,
            context.task_novelty,
            context.environment_stability,
            context.time_pressure,
            np.mean(context.previous_performance) if context.previous_performance else 0.5,
            context.confidence_in_context,
            context.context_similarity_to_past,
            context.expected_learning_difficulty
        ]
        
        # Pad to 64 dimensions
        while len(context_features) < 64:
            context_features.append(0.0)
        
        context_tensor = torch.tensor(context_features[:64], dtype=torch.float32).unsqueeze(0)
        
        # Get strategy probabilities
        with torch.no_grad():
            strategy_probs = self.strategy_selector(context_tensor).squeeze(0)
        
        # Select strategy
        selected_idx = torch.argmax(strategy_probs).item()
        selected_strategy = list(AdaptationStrategy)[selected_idx]
        
        return selected_strategy
    
    def _integrate_learning_results(self, learning_results: Dict[str, Any]) -> float:
        """Integrate performance across all learning modules."""
        
        performances = []
        
        for module_name, result in learning_results.items():
            if isinstance(result, dict) and 'performance' in result:
                performances.append(result['performance'])
        
        if not performances:
            return 0.5  # Default performance
        
        # Weighted average (could be more sophisticated)
        weights = [1.0] * len(performances)  # Equal weights for now
        
        integrated = np.average(performances, weights=weights)
        return float(integrated)
    
    def _should_adapt(self, learning_results: Dict[str, Any], context: LearningContext) -> bool:
        """Determine if system adaptation is needed."""
        
        # Check individual module performance
        for module_name, result in learning_results.items():
            if isinstance(result, dict) and 'performance' in result:
                if result['performance'] < 0.4:  # Poor performance threshold
                    return True
        
        # Check system-level performance trends
        if len(self.system_performance) >= 10:
            recent = list(self.system_performance)[-5:]
            earlier = list(self.system_performance)[-10:-5]
            
            if np.mean(recent) < np.mean(earlier) - 0.1:  # Significant degradation
                return True
        
        # Check novelty - adapt for highly novel situations
        if context.task_novelty > 0.8:
            return True
        
        return False
    
    def _perform_adaptation(self, learning_results: Dict[str, Any], context: LearningContext, strategy: AdaptationStrategy) -> Dict[str, Any]:
        """Perform system adaptation."""
        
        adaptation_results = {}
        adaptations_made = 0
        
        # Strategy-specific adaptation parameters
        if strategy == AdaptationStrategy.CONSERVATIVE:
            adaptation_intensity = 0.1
        elif strategy == AdaptationStrategy.MODERATE:
            adaptation_intensity = 0.3
        elif strategy == AdaptationStrategy.AGGRESSIVE:
            adaptation_intensity = 0.7
        elif strategy == AdaptationStrategy.EXPERIMENTAL:
            adaptation_intensity = 0.9
        else:  # DYNAMIC
            adaptation_intensity = context.task_novelty  # Use novelty as intensity
        
        # Adapt each module based on its performance
        for module_name, module in self.learning_modules.items():
            if module.is_active and module_name in learning_results:
                result = learning_results[module_name]
                
                if isinstance(result, dict) and 'performance' in result:
                    # Create adaptation feedback
                    feedback = {
                        'performance': result['performance'],
                        'target_performance': 0.8,
                        'adaptation_intensity': adaptation_intensity,
                        'context': context.to_dict()
                    }
                    
                    # Attempt adaptation
                    adapted = module.adapt(feedback, context)
                    
                    adaptation_results[module_name] = {
                        'adapted': adapted,
                        'performance': result['performance'],
                        'adaptation_intensity': adaptation_intensity
                    }
                    
                    if adapted:
                        adaptations_made += 1
        
        # Record adaptation
        adaptation_record = {
            'timestamp': time.time(),
            'strategy': strategy.value,
            'adaptations_made': adaptations_made,
            'context': context.to_dict(),
            'results': adaptation_results
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Update resource usage
        self.resource_usage['adaptation_frequency'] = len(self.adaptation_history) / max(1, (time.time() - self.adaptation_history[0]['timestamp']) / 3600) if self.adaptation_history else 0
        
        logger.info(f"System adaptation completed: {adaptations_made} modules adapted using {strategy.value} strategy")
        
        return {
            'strategy': strategy.value,
            'adaptations_made': adaptations_made,
            'module_results': adaptation_results,
            'adaptation_intensity': adaptation_intensity
        }
    
    def continuous_learning_cycle(self, duration_hours: float = 2.0) -> Dict[str, Any]:
        """Run continuous learning cycle for specified duration."""
        
        logger.info(f"Starting continuous learning cycle for {duration_hours} hours")
        
        start_time = time.time()
        end_time = start_time + duration_hours * 3600
        
        cycle_results = {
            'start_time': start_time,
            'duration_hours': duration_hours,
            'learning_iterations': 0,
            'adaptations_made': 0,
            'average_performance': 0.0,
            'performance_improvement': 0.0
        }
        
        initial_performance = np.mean(list(self.system_performance)[-10:]) if self.system_performance else 0.5
        total_performance = 0.0
        
        while time.time() < end_time:
            # Generate synthetic data for continuous learning
            synthetic_data = self._generate_synthetic_learning_data()
            
            # Perform adaptive learning
            learning_result = self.adaptive_learn(synthetic_data)
            
            cycle_results['learning_iterations'] += 1
            total_performance += learning_result['integrated_performance']
            
            if learning_result.get('adaptation_needed', False):
                cycle_results['adaptations_made'] += 1
            
            # Brief pause between learning iterations
            time.sleep(2.0)
        
        # Calculate final metrics
        cycle_results['end_time'] = time.time()
        cycle_results['average_performance'] = total_performance / max(1, cycle_results['learning_iterations'])
        
        final_performance = np.mean(list(self.system_performance)[-10:]) if self.system_performance else 0.5
        cycle_results['performance_improvement'] = final_performance - initial_performance
        
        logger.info(
            f"Continuous learning cycle completed: {cycle_results['learning_iterations']} iterations, "
            f"{cycle_results['adaptations_made']} adaptations, "
            f"performance improvement: {cycle_results['performance_improvement']:.4f}"
        )
        
        return cycle_results
    
    def _generate_synthetic_learning_data(self) -> torch.Tensor:
        """Generate synthetic data for continuous learning."""
        
        # Generate data with varying complexity and novelty
        base_data = torch.randn(self.input_dim)
        
        # Add complexity through correlations
        complexity_factor = np.random.beta(2, 3)  # Varying complexity
        correlation_matrix = torch.randn(self.input_dim, self.input_dim) * complexity_factor
        
        synthetic_data = base_data + torch.mv(correlation_matrix, torch.randn(self.input_dim)) * 0.1
        
        # Add some noise for realism
        noise = torch.randn_like(synthetic_data) * 0.05
        synthetic_data += noise
        
        return synthetic_data
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of adaptive learning system."""
        
        summary = {
            'system_performance': {
                'current_performance': list(self.system_performance)[-1] if self.system_performance else 0.0,
                'average_performance': np.mean(self.system_performance) if self.system_performance else 0.0,
                'performance_trend': self._calculate_system_trend(),
                'total_learning_iterations': len(self.system_performance)
            },
            'learning_modules': {},
            'adaptation_statistics': {
                'total_adaptations': len(self.adaptation_history),
                'adaptation_frequency': self.resource_usage['adaptation_frequency'],
                'recent_adaptations': len([a for a in self.adaptation_history if time.time() - a['timestamp'] < 3600])  # Last hour
            },
            'resource_usage': self.resource_usage.copy(),
            'context_analysis': self._analyze_learning_contexts()
        }
        
        # Get module-specific summaries
        for module_name, module in self.learning_modules.items():
            summary['learning_modules'][module_name] = module.get_performance_metrics()
        
        return summary
    
    def _calculate_system_trend(self) -> float:
        """Calculate overall system performance trend."""
        
        if len(self.system_performance) < 10:
            return 0.0
        
        recent = list(self.system_performance)[-10:]
        earlier = list(self.system_performance)[-20:-10] if len(self.system_performance) >= 20 else list(self.system_performance)[:-10]
        
        if not earlier:
            return 0.0
        
        return np.mean(recent) - np.mean(earlier)
    
    def _analyze_learning_contexts(self) -> Dict[str, Any]:
        """Analyze learning contexts for insights."""
        
        if not self.learning_contexts:
            return {'status': 'no_data'}
        
        contexts = list(self.learning_contexts)
        
        return {
            'average_complexity': np.mean([c.task_complexity for c in contexts]),
            'average_novelty': np.mean([c.task_novelty for c in contexts]),
            'average_confidence': np.mean([c.confidence_in_context for c in contexts]),
            'context_diversity': np.std([c.task_complexity for c in contexts]),
            'total_contexts': len(contexts)
        }
    
    def shutdown(self):
        """Shutdown adaptive learning system gracefully."""
        
        logger.info("Shutting down adaptive learning system")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Deactivate all learning modules
        for module in self.learning_modules.values():
            module.is_active = False
        
        # Shutdown cognitive architecture
        self.cognitive_architecture.shutdown()
        
        logger.info("Adaptive learning system shutdown complete")


def create_adaptive_learning_system(input_dim: int = 512, config: Optional[Dict[str, Any]] = None) -> AdaptiveLearningSystem:
    """Factory function to create adaptive learning system."""
    
    return AdaptiveLearningSystem(input_dim, config)
