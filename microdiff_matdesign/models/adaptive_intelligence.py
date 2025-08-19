"""Adaptive Intelligence Models for Dynamic Materials Discovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math
from collections import deque


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive intelligence systems."""
    
    adaptation_rate: float = 0.01
    memory_capacity: int = 10000
    meta_learning_steps: int = 5
    adaptation_threshold: float = 0.1
    plasticity_decay: float = 0.99
    exploration_bonus: float = 0.1
    few_shot_examples: int = 10


class NeuralPlasticityModule(nn.Module):
    """Neural plasticity module that adapts network weights based on experience."""
    
    def __init__(self, layer_dim: int, adaptation_rate: float = 0.01):
        super().__init__()
        
        self.layer_dim = layer_dim
        self.adaptation_rate = adaptation_rate
        
        # Base weights that adapt slowly
        self.base_weights = nn.Parameter(torch.randn(layer_dim, layer_dim) * 0.1)
        
        # Fast adaptation weights
        self.fast_weights = nn.Parameter(torch.zeros(layer_dim, layer_dim))
        
        # Plasticity gates - control how much each connection can adapt
        self.plasticity_gates = nn.Parameter(torch.ones(layer_dim, layer_dim) * 0.5)
        
        # Adaptation controller
        self.adaptation_controller = nn.Sequential(
            nn.Linear(layer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, layer_dim * layer_dim),
            nn.Tanh()
        )
        
        # Hebbian learning detector
        self.hebbian_detector = HebbianLearningModule(layer_dim)
        
    def forward(self, x: torch.Tensor, adaptation_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with adaptive weights."""
        
        # Compute effective weights
        plasticity_mask = torch.sigmoid(self.plasticity_gates)
        effective_weights = self.base_weights + plasticity_mask * self.fast_weights
        
        # Apply transformation
        output = F.linear(x, effective_weights.T)
        
        # Adapt weights based on Hebbian learning
        if self.training and adaptation_signal is not None:
            self._adapt_weights(x, output, adaptation_signal)
        
        return output
    
    def _adapt_weights(self, input_activation: torch.Tensor, 
                     output_activation: torch.Tensor, 
                     adaptation_signal: torch.Tensor):
        """Adapt weights based on input-output correlations and adaptation signal."""
        
        # Hebbian learning rule: strengthen connections that fire together
        hebbian_update = self.hebbian_detector(input_activation, output_activation)
        
        # Modulate adaptation by signal strength
        signal_strength = adaptation_signal.mean()
        
        # Update fast weights
        adaptation_magnitude = self.adaptation_rate * signal_strength
        self.fast_weights.data += adaptation_magnitude * hebbian_update
        
        # Decay fast weights to prevent instability
        self.fast_weights.data *= 0.999


class HebbianLearningModule(nn.Module):
    """Implements Hebbian learning rules for synaptic plasticity."""
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.dim = dim
        
        # Learning rate modulation
        self.learning_rate_modulator = nn.Parameter(torch.ones(dim, dim) * 0.01)
        
    def forward(self, pre_synaptic: torch.Tensor, post_synaptic: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian weight updates."""
        
        # Standard Hebbian rule: Δw = η * pre * post
        batch_size = pre_synaptic.shape[0]
        
        # Average across batch for stability
        avg_pre = pre_synaptic.mean(dim=0, keepdim=True)  # [1, dim]
        avg_post = post_synaptic.mean(dim=0, keepdim=True)  # [1, dim]
        
        # Outer product for weight update
        hebbian_update = torch.outer(avg_pre.squeeze(), avg_post.squeeze())
        
        # Modulate by learned learning rates
        modulated_update = self.learning_rate_modulator * hebbian_update
        
        return modulated_update


class MetaLearningController(nn.Module):
    """Meta-learning controller that learns how to adapt quickly to new materials."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for processing sequential experiences
        self.experience_lstm = nn.LSTM(
            input_dim * 2 + 1,  # input + target + reward
            hidden_dim,
            batch_first=True
        )
        
        # Meta-parameter generator
        self.meta_param_generator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
        # Adaptation strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 different adaptation strategies
            nn.Softmax(dim=-1)
        )
        
        # Few-shot learning capability
        self.few_shot_learner = FewShotLearner(input_dim, hidden_dim)
        
    def forward(self, experience_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Process experience sequence and generate meta-parameters."""
        
        # Prepare sequence for LSTM
        sequence_inputs = []
        for exp in experience_sequence:
            # Combine input, target, and reward
            seq_input = torch.cat([
                exp['input'],
                exp['target'],
                exp['reward'].unsqueeze(-1)
            ])
            sequence_inputs.append(seq_input)
        
        # Stack into sequence
        sequence_tensor = torch.stack(sequence_inputs).unsqueeze(0)  # [1, seq_len, input_dim]
        
        # Process through LSTM
        lstm_output, (hidden, cell) = self.experience_lstm(sequence_tensor)
        
        # Use final hidden state for meta-parameter generation
        final_hidden = hidden[-1]  # Last layer, last timestep
        
        # Generate meta-parameters
        meta_params = self.meta_param_generator(final_hidden)
        
        # Select adaptation strategy
        adaptation_strategy = self.strategy_selector(final_hidden)
        
        # Few-shot learning prediction
        few_shot_prediction = self.few_shot_learner(experience_sequence)
        
        return {
            'meta_parameters': meta_params,
            'adaptation_strategy': adaptation_strategy,
            'few_shot_prediction': few_shot_prediction,
            'lstm_hidden': final_hidden
        }


class FewShotLearner(nn.Module):
    """Few-shot learning module for rapid adaptation to new materials."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Prototype network for few-shot learning
        self.prototype_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Attention mechanism for prototype weighting
        self.prototype_attention = nn.MultiheadAttention(
            hidden_dim // 2, num_heads=8, batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, support_examples: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Learn from few support examples and make prediction."""
        
        if len(support_examples) == 0:
            return torch.zeros(self.input_dim)
        
        # Encode support examples
        support_encodings = []
        for example in support_examples:
            encoding = self.prototype_encoder(example['input'])
            support_encodings.append(encoding)
        
        # Stack encodings
        support_tensor = torch.stack(support_encodings)  # [num_examples, hidden_dim//2]
        
        # Compute prototype through attention
        query = support_tensor.mean(dim=0, keepdim=True)  # [1, hidden_dim//2]
        attended_prototype, _ = self.prototype_attention(
            query, support_tensor, support_tensor
        )
        
        # Generate prediction
        prediction = self.prediction_head(attended_prototype.squeeze(0))
        
        return prediction


class ContinualLearningModule(nn.Module):
    """Continual learning module that prevents catastrophic forgetting."""
    
    def __init__(self, model_dim: int, memory_capacity: int = 1000):
        super().__init__()
        
        self.model_dim = model_dim
        self.memory_capacity = memory_capacity
        
        # Episodic memory for important experiences
        self.episodic_memory = EpisodicMemory(memory_capacity, model_dim)
        
        # Importance weighting for different experiences
        self.importance_estimator = nn.Sequential(
            nn.Linear(model_dim * 2, 128),  # input + output
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Memory consolidation network
        self.consolidation_network = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, model_dim)
        )
        
        # Regularization strength controller
        self.regularization_controller = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, current_features: torch.Tensor, 
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process current experience and manage memory."""
        
        # Estimate importance of current experience
        importance_input = torch.cat([current_features, target])
        importance = self.importance_estimator(importance_input)
        
        # Store in episodic memory if important enough
        if importance.item() > 0.5:
            self.episodic_memory.store(current_features, target, importance)
        
        # Retrieve similar past experiences
        similar_experiences = self.episodic_memory.retrieve_similar(current_features, k=5)
        
        # Memory consolidation
        consolidated_memory = None
        if similar_experiences:
            memory_features = torch.stack([exp['features'] for exp in similar_experiences])
            consolidated_memory = self.consolidation_network(memory_features.mean(dim=0))
        
        # Compute regularization loss to prevent forgetting
        regularization_loss = self._compute_regularization_loss(current_features, similar_experiences)
        
        return {
            'importance': importance,
            'consolidated_memory': consolidated_memory,
            'regularization_loss': regularization_loss,
            'num_memories': len(self.episodic_memory)
        }
    
    def _compute_regularization_loss(self, current_features: torch.Tensor, 
                                   similar_experiences: List[Dict]) -> torch.Tensor:
        """Compute regularization loss to prevent catastrophic forgetting."""
        
        if not similar_experiences:
            return torch.tensor(0.0, device=current_features.device)
        
        # Elastic Weight Consolidation-style regularization
        total_loss = torch.tensor(0.0, device=current_features.device)
        
        for exp in similar_experiences:
            memory_features = exp['features']
            importance = exp['importance']
            
            # L2 distance weighted by importance
            distance = F.mse_loss(current_features, memory_features, reduction='sum')
            weighted_distance = importance * distance
            total_loss += weighted_distance
        
        return self.regularization_controller * total_loss / len(similar_experiences)


class EpisodicMemory:
    """Episodic memory for storing and retrieving important experiences."""
    
    def __init__(self, capacity: int, feature_dim: int):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.memories = deque(maxlen=capacity)
        
    def store(self, features: torch.Tensor, target: torch.Tensor, importance: torch.Tensor):
        """Store a new memory."""
        memory = {
            'features': features.detach().cpu(),
            'target': target.detach().cpu(),
            'importance': importance.detach().cpu(),
            'timestamp': len(self.memories)
        }
        self.memories.append(memory)
    
    def retrieve_similar(self, query_features: torch.Tensor, k: int = 5) -> List[Dict]:
        """Retrieve k most similar memories to query."""
        
        if len(self.memories) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for memory in self.memories:
            memory_features = memory['features'].to(query_features.device)
            similarity = F.cosine_similarity(
                query_features.unsqueeze(0), 
                memory_features.unsqueeze(0)
            ).item()
            similarities.append((similarity, memory))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similarities[:k]]
    
    def __len__(self):
        return len(self.memories)


class AdaptiveNeuralArchitecture(nn.Module):
    """Neural architecture that adapts its structure based on task complexity."""
    
    def __init__(self, input_dim: int, max_hidden_dim: int = 512):
        super().__init__()
        
        self.input_dim = input_dim
        self.max_hidden_dim = max_hidden_dim
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Modular layers of different capacities
        self.layer_modules = nn.ModuleDict({
            'small': nn.Linear(input_dim, max_hidden_dim // 4),
            'medium': nn.Linear(input_dim, max_hidden_dim // 2),
            'large': nn.Linear(input_dim, max_hidden_dim),
            'extra_large': nn.Linear(input_dim, max_hidden_dim * 2)
        })
        
        # Architecture selector
        self.architecture_selector = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # +1 for complexity score
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 architecture options
            nn.Softmax(dim=-1)
        )
        
        # Dynamic routing network
        self.router = DynamicRoutingNetwork(max_hidden_dim * 2)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adaptive forward pass with dynamic architecture selection."""
        
        # Estimate task complexity
        complexity = self.complexity_estimator(x)
        
        # Select architecture based on complexity
        selector_input = torch.cat([x, complexity], dim=-1)
        architecture_weights = self.architecture_selector(selector_input)
        
        # Compute outputs from all modules
        module_outputs = {}
        for name, module in self.layer_modules.items():
            module_outputs[name] = module(x)
        
        # Weighted combination of outputs
        weighted_outputs = []
        module_names = ['small', 'medium', 'large', 'extra_large']
        
        for i, name in enumerate(module_names):
            weight = architecture_weights[:, i:i+1]
            output = module_outputs[name]
            
            # Pad smaller outputs to match largest dimension
            if output.shape[-1] < self.max_hidden_dim * 2:
                padding = self.max_hidden_dim * 2 - output.shape[-1]
                output = F.pad(output, (0, padding))
            
            weighted_output = weight * output
            weighted_outputs.append(weighted_output)
        
        # Sum weighted outputs
        combined_output = sum(weighted_outputs)
        
        # Dynamic routing for specialized processing
        routed_output = self.router(combined_output)
        
        return {
            'output': routed_output,
            'complexity': complexity,
            'architecture_weights': architecture_weights,
            'selected_capacity': torch.sum(architecture_weights * torch.tensor([0.25, 0.5, 1.0, 2.0]))
        }


class DynamicRoutingNetwork(nn.Module):
    """Dynamic routing network for capsule-like processing."""
    
    def __init__(self, input_dim: int, num_capsules: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_capsules = num_capsules
        self.capsule_dim = input_dim // num_capsules
        
        # Transformation matrices for each capsule
        self.capsule_transforms = nn.ModuleList([
            nn.Linear(self.capsule_dim, self.capsule_dim)
            for _ in range(num_capsules)
        ])
        
        # Routing coefficients
        self.routing_iterations = 3
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamic routing between capsules."""
        
        batch_size = x.shape[0]
        
        # Split input into capsules
        capsule_inputs = x.view(batch_size, self.num_capsules, self.capsule_dim)
        
        # Transform each capsule
        transformed_capsules = []
        for i, transform in enumerate(self.capsule_transforms):
            transformed = transform(capsule_inputs[:, i])
            transformed_capsules.append(transformed)
        
        capsule_outputs = torch.stack(transformed_capsules, dim=1)
        
        # Dynamic routing
        routing_weights = torch.ones(batch_size, self.num_capsules, 1) / self.num_capsules
        routing_weights = routing_weights.to(x.device)
        
        for iteration in range(self.routing_iterations):
            # Weighted sum of capsule outputs
            weighted_sum = torch.sum(routing_weights * capsule_outputs, dim=1)
            
            # Squash function (capsule activation)
            squashed_output = self._squash(weighted_sum)
            
            if iteration < self.routing_iterations - 1:
                # Update routing weights based on agreement
                agreement = torch.sum(
                    capsule_outputs * squashed_output.unsqueeze(1), dim=-1, keepdim=True
                )
                routing_weights = F.softmax(agreement, dim=1)
        
        return squashed_output.view(batch_size, -1)
    
    def _squash(self, tensor: torch.Tensor) -> torch.Tensor:
        """Squash function for capsule activation."""
        squared_norm = torch.sum(tensor ** 2, dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        return scale * unit_vector


class AdaptiveIntelligenceSystem(nn.Module):
    """Complete adaptive intelligence system for materials discovery."""
    
    def __init__(self, material_dim: int, property_dim: int, 
                 config: Optional[AdaptiveConfig] = None):
        super().__init__()
        
        if config is None:
            config = AdaptiveConfig()
        
        self.config = config
        self.material_dim = material_dim
        self.property_dim = property_dim
        
        # Core adaptive components
        self.neural_plasticity = NeuralPlasticityModule(material_dim, config.adaptation_rate)
        self.meta_learner = MetaLearningController(material_dim)
        self.continual_learner = ContinualLearningModule(material_dim, config.memory_capacity)
        self.adaptive_architecture = AdaptiveNeuralArchitecture(material_dim)
        
        # Task-specific heads
        self.property_predictor = nn.Sequential(
            nn.Linear(material_dim, 256),
            nn.ReLU(),
            nn.Linear(256, property_dim)
        )
        
        self.inverse_designer = nn.Sequential(
            nn.Linear(property_dim, 256),
            nn.ReLU(),
            nn.Linear(256, material_dim)
        )
        
        # Adaptation tracker
        self.adaptation_history = deque(maxlen=1000)
        
    def forward(self, materials: torch.Tensor, 
                target_properties: Optional[torch.Tensor] = None,
                mode: str = 'forward') -> Dict[str, torch.Tensor]:
        """Adaptive forward pass with multiple operation modes."""
        
        # Adaptive architecture processing
        arch_result = self.adaptive_architecture(materials)
        adaptive_features = arch_result['output']
        
        # Neural plasticity adaptation
        adaptation_signal = self._compute_adaptation_signal(materials, target_properties)
        plastic_features = self.neural_plasticity(adaptive_features, adaptation_signal)
        
        # Task-specific processing
        if mode == 'forward':
            # Predict properties from materials
            predicted_properties = self.property_predictor(plastic_features)
            output = predicted_properties
            
        elif mode == 'inverse':
            # Design materials from target properties
            if target_properties is None:
                raise ValueError("Target properties required for inverse mode")
            designed_materials = self.inverse_designer(target_properties)
            output = designed_materials
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Continual learning update
        if target_properties is not None:
            continual_result = self.continual_learner(plastic_features, target_properties)
        else:
            continual_result = {'regularization_loss': torch.tensor(0.0)}
        
        # Track adaptation
        self._track_adaptation(adaptation_signal, arch_result['complexity'])
        
        return {
            'output': output,
            'adaptive_features': plastic_features,
            'architecture_info': arch_result,
            'continual_learning': continual_result,
            'adaptation_signal': adaptation_signal
        }
    
    def meta_adapt(self, few_shot_examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Meta-adaptation based on few-shot examples."""
        
        # Meta-learning from examples
        meta_result = self.meta_learner(few_shot_examples)
        
        # Apply meta-parameters to adapt the system
        meta_params = meta_result['meta_parameters']
        
        # Update plasticity based on meta-learning
        self.neural_plasticity.adaptation_rate = (
            self.config.adaptation_rate * meta_params.mean().item()
        )
        
        return meta_result
    
    def _compute_adaptation_signal(self, materials: torch.Tensor, 
                                 target_properties: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute adaptation signal based on prediction error."""
        
        if target_properties is None:
            # No supervision signal
            return torch.zeros(materials.shape[0], device=materials.device)
        
        # Predict properties
        with torch.no_grad():
            arch_result = self.adaptive_architecture(materials)
            predicted_properties = self.property_predictor(arch_result['output'])
        
        # Compute prediction error as adaptation signal
        error = F.mse_loss(predicted_properties, target_properties, reduction='none')
        adaptation_signal = error.mean(dim=-1)  # Average across property dimensions
        
        return adaptation_signal
    
    def _track_adaptation(self, adaptation_signal: torch.Tensor, complexity: torch.Tensor):
        """Track adaptation progress over time."""
        
        adaptation_info = {
            'signal_strength': adaptation_signal.mean().item(),
            'complexity': complexity.mean().item(),
            'timestamp': len(self.adaptation_history)
        }
        
        self.adaptation_history.append(adaptation_info)
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation progress."""
        
        if not self.adaptation_history:
            return {'status': 'no_adaptation_data'}
        
        recent_signals = [info['signal_strength'] for info in list(self.adaptation_history)[-100:]]
        recent_complexity = [info['complexity'] for info in list(self.adaptation_history)[-100:]]
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'recent_avg_signal': np.mean(recent_signals),
            'recent_avg_complexity': np.mean(recent_complexity),
            'adaptation_trend': np.polyfit(range(len(recent_signals)), recent_signals, 1)[0],
            'current_plasticity_rate': self.neural_plasticity.adaptation_rate
        }