"""Cognitive Architecture for Next-Generation Autonomous Systems."""

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
import networkx as nx
from abc import ABC, abstractmethod

from .self_evolving_ai import SelfImprovingSystem, EvolutionConfig
from ..evolution.meta_learning_optimizer import MetaLearningOptimizer, MetaLearningConfig
from ..models.consciousness_aware import ConsciousMaterialsExplorer
from ..models.adaptive_intelligence import AdaptiveIntelligenceNetwork


logger = logging.getLogger(__name__)


class CognitiveModule(Enum):
    """Types of cognitive modules in the architecture."""
    
    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    METACOGNITION = "metacognition"
    CREATIVITY = "creativity"
    CONSCIOUSNESS = "consciousness"


class ProcessingMode(Enum):
    """Processing modes for cognitive operations."""
    
    AUTOMATIC = "automatic"      # Fast, unconscious processing
    CONTROLLED = "controlled"    # Slow, conscious processing
    HYBRID = "hybrid"            # Combination of both
    QUANTUM = "quantum"          # Quantum-enhanced processing


@dataclass
class CognitiveState:
    """Current state of the cognitive architecture."""
    
    # Core states
    attention_focus: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Processing states
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    cognitive_load: float = 0.0
    arousal_level: float = 0.5
    
    # Meta-cognitive states
    confidence_level: float = 0.5
    uncertainty: float = 0.5
    learning_rate_adjustment: float = 1.0
    
    # Creative states
    creativity_level: float = 0.5
    exploration_bias: float = 0.5
    novelty_seeking: float = 0.5
    
    # Performance metrics
    task_performance: float = 0.0
    adaptation_rate: float = 0.0
    problem_solving_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cognitive state to dictionary."""
        return {
            'attention_focus': self.attention_focus.copy(),
            'working_memory_size': len(self.working_memory),
            'long_term_memory_size': len(self.long_term_memory),
            'processing_mode': self.processing_mode.value,
            'cognitive_load': self.cognitive_load,
            'arousal_level': self.arousal_level,
            'confidence_level': self.confidence_level,
            'uncertainty': self.uncertainty,
            'learning_rate_adjustment': self.learning_rate_adjustment,
            'creativity_level': self.creativity_level,
            'exploration_bias': self.exploration_bias,
            'novelty_seeking': self.novelty_seeking,
            'task_performance': self.task_performance,
            'adaptation_rate': self.adaptation_rate,
            'problem_solving_efficiency': self.problem_solving_efficiency
        }


class CognitiveProcessor(ABC):
    """Abstract base class for cognitive processors."""
    
    def __init__(self, module_type: CognitiveModule, capacity: int = 1000):
        self.module_type = module_type
        self.capacity = capacity
        self.processing_history = deque(maxlen=capacity)
        self.performance_metrics = defaultdict(list)
        self.is_active = True
        
    @abstractmethod
    def process(self, input_data: Any, context: CognitiveState) -> Tuple[Any, CognitiveState]:
        """Process input data and update cognitive state."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[f"{metric_name}_mean"] = np.mean(values[-100:])  # Last 100 values
                metrics[f"{metric_name}_std"] = np.std(values[-100:])
        return metrics


class PerceptionProcessor(CognitiveProcessor):
    """Processor for perceptual input and feature extraction."""
    
    def __init__(self, input_dim: int, feature_dim: int = 256):
        super().__init__(CognitiveModule.PERCEPTION)
        
        # Perceptual networks
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.attention_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.pattern_memory = {}
        self.novelty_detector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def process(self, input_data: torch.Tensor, context: CognitiveState) -> Tuple[Dict[str, torch.Tensor], CognitiveState]:
        """Process perceptual input."""
        
        # Extract features
        features = self.feature_extractor(input_data)
        
        # Compute attention weights
        attention_weights = self.attention_network(features)
        attended_features = features * attention_weights
        
        # Detect novelty
        novelty_score = self.novelty_detector(features)
        
        # Update cognitive state
        context.working_memory['current_features'] = features
        context.working_memory['attended_features'] = attended_features
        context.working_memory['novelty_score'] = float(novelty_score)
        
        # Adjust arousal based on novelty
        context.arousal_level = 0.7 * context.arousal_level + 0.3 * float(novelty_score)
        
        # Record performance
        self.performance_metrics['attention_strength'].append(float(attention_weights.mean()))
        self.performance_metrics['novelty_detection'].append(float(novelty_score))
        
        return {
            'features': features,
            'attended_features': attended_features,
            'attention_weights': attention_weights,
            'novelty_score': novelty_score
        }, context


class MemoryProcessor(CognitiveProcessor):
    """Processor for memory operations and knowledge management."""
    
    def __init__(self, memory_dim: int = 512, max_memories: int = 10000):
        super().__init__(CognitiveModule.MEMORY, max_memories)
        
        self.memory_dim = memory_dim
        self.episodic_memory = {}  # Event-specific memories
        self.semantic_memory = {}  # General knowledge
        self.working_memory_buffer = deque(maxlen=10)  # Short-term buffer
        
        # Memory networks
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )
        
        self.memory_retrieval = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )
        
        self.relevance_scorer = nn.Sequential(
            nn.Linear(memory_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.memory_consolidation = nn.GRU(memory_dim, memory_dim, batch_first=True)
    
    def process(self, input_data: Dict[str, Any], context: CognitiveState) -> Tuple[Dict[str, Any], CognitiveState]:
        """Process memory operations."""
        
        query = input_data.get('query')
        store_data = input_data.get('store')
        retrieve_request = input_data.get('retrieve', True)
        
        results = {}
        
        # Store new information
        if store_data is not None:
            memory_id = self._store_memory(store_data, context)
            results['stored_memory_id'] = memory_id
        
        # Retrieve relevant memories
        if retrieve_request and query is not None:
            retrieved_memories = self._retrieve_memories(query, context)
            results['retrieved_memories'] = retrieved_memories
            
            # Update working memory
            context.working_memory['retrieved_memories'] = retrieved_memories
        
        # Update memory metrics
        context.working_memory['memory_load'] = len(self.episodic_memory) / self.capacity
        
        return results, context
    
    def _store_memory(self, data: Any, context: CognitiveState) -> str:
        """Store information in memory."""
        
        memory_id = f"mem_{time.time()}_{len(self.episodic_memory)}"
        
        # Encode memory
        if isinstance(data, torch.Tensor):
            encoded_memory = self.memory_encoder(data)
        else:
            # Convert to tensor for encoding (simplified)
            data_tensor = torch.randn(self.memory_dim)  # Placeholder
            encoded_memory = self.memory_encoder(data_tensor)
        
        # Store in episodic memory
        self.episodic_memory[memory_id] = {
            'data': data,
            'encoded': encoded_memory,
            'timestamp': time.time(),
            'context': copy.deepcopy(context.to_dict()),
            'access_count': 0
        }
        
        # Add to working memory buffer
        self.working_memory_buffer.append(memory_id)
        
        return memory_id
    
    def _retrieve_memories(self, query: torch.Tensor, context: CognitiveState, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        
        if not self.episodic_memory:
            return []
        
        # Encode query
        query_encoded = self.memory_retrieval(query)
        
        # Compute relevance scores
        relevance_scores = []
        memory_ids = list(self.episodic_memory.keys())
        
        for memory_id in memory_ids:
            memory_encoded = self.episodic_memory[memory_id]['encoded']
            
            # Compute relevance
            combined = torch.cat([query_encoded, memory_encoded], dim=-1)
            relevance = self.relevance_scorer(combined)
            
            relevance_scores.append((float(relevance), memory_id))
        
        # Sort by relevance and return top_k
        relevance_scores.sort(key=lambda x: x[0], reverse=True)
        top_memories = relevance_scores[:top_k]
        
        retrieved = []
        for relevance, memory_id in top_memories:
            memory_data = self.episodic_memory[memory_id]
            memory_data['access_count'] += 1
            
            retrieved.append({
                'memory_id': memory_id,
                'relevance': relevance,
                'data': memory_data['data'],
                'timestamp': memory_data['timestamp'],
                'access_count': memory_data['access_count']
            })
        
        return retrieved


class ReasoningProcessor(CognitiveProcessor):
    """Processor for logical reasoning and inference."""
    
    def __init__(self, reasoning_dim: int = 512):
        super().__init__(CognitiveModule.REASONING)
        
        self.reasoning_dim = reasoning_dim
        
        # Reasoning networks
        self.logical_reasoner = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim)
        )
        
        self.causal_reasoner = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim)
        )
        
        self.analogical_reasoner = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim)
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(reasoning_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Knowledge graph for structured reasoning
        self.knowledge_graph = nx.DiGraph()
    
    def process(self, input_data: Dict[str, Any], context: CognitiveState) -> Tuple[Dict[str, Any], CognitiveState]:
        """Process reasoning operations."""
        
        premises = input_data.get('premises', [])
        reasoning_type = input_data.get('type', 'logical')
        
        if reasoning_type == 'logical':
            result = self._logical_reasoning(premises, context)
        elif reasoning_type == 'causal':
            result = self._causal_reasoning(premises, context)
        elif reasoning_type == 'analogical':
            result = self._analogical_reasoning(premises, context)
        else:
            result = self._logical_reasoning(premises, context)
        
        # Update cognitive state
        context.uncertainty = float(result.get('uncertainty', 0.5))
        context.confidence_level = 1.0 - context.uncertainty
        
        return result, context
    
    def _logical_reasoning(self, premises: List[torch.Tensor], context: CognitiveState) -> Dict[str, Any]:
        """Perform logical reasoning on premises."""
        
        if not premises:
            return {'conclusion': None, 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Combine premises
        combined_premises = torch.stack(premises).mean(dim=0)
        
        # Apply logical reasoning
        reasoning_output = self.logical_reasoner(combined_premises)
        uncertainty = self.uncertainty_estimator(reasoning_output)
        
        return {
            'conclusion': reasoning_output,
            'confidence': 1.0 - float(uncertainty),
            'uncertainty': float(uncertainty),
            'reasoning_type': 'logical'
        }
    
    def _causal_reasoning(self, premises: List[torch.Tensor], context: CognitiveState) -> Dict[str, Any]:
        """Perform causal reasoning."""
        
        if len(premises) < 2:
            return {'conclusion': None, 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Use first premise as cause, second as potential effect
        cause = premises[0]
        effect = premises[1]
        
        # Causal reasoning
        causal_link = self.causal_reasoner(torch.cat([cause, effect], dim=-1))
        uncertainty = self.uncertainty_estimator(causal_link)
        
        return {
            'conclusion': causal_link,
            'confidence': 1.0 - float(uncertainty),
            'uncertainty': float(uncertainty),
            'reasoning_type': 'causal'
        }
    
    def _analogical_reasoning(self, premises: List[torch.Tensor], context: CognitiveState) -> Dict[str, Any]:
        """Perform analogical reasoning."""
        
        if len(premises) < 2:
            return {'conclusion': None, 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Use analogical reasoning network
        source = premises[0]
        target = premises[1]
        
        analogy = self.analogical_reasoner(torch.cat([source, target], dim=-1))
        uncertainty = self.uncertainty_estimator(analogy)
        
        return {
            'conclusion': analogy,
            'confidence': 1.0 - float(uncertainty),
            'uncertainty': float(uncertainty),
            'reasoning_type': 'analogical'
        }


class MetaCognitiveProcessor(CognitiveProcessor):
    """Processor for meta-cognitive operations and self-awareness."""
    
    def __init__(self, meta_dim: int = 256):
        super().__init__(CognitiveModule.METACOGNITION)
        
        self.meta_dim = meta_dim
        
        # Meta-cognitive networks
        self.self_monitor = nn.Sequential(
            nn.Linear(meta_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, meta_dim)
        )
        
        self.strategy_selector = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 different strategies
            nn.Softmax(dim=-1)
        )
        
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Meta-learning components
        self.meta_learner = MetaLearningOptimizer()
        
        # Self-model (model of own cognitive processes)
        self.self_model = nn.GRU(meta_dim, meta_dim, batch_first=True)
        
        # Strategy tracking
        self.strategy_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
    
    def process(self, input_data: Dict[str, Any], context: CognitiveState) -> Tuple[Dict[str, Any], CognitiveState]:
        """Process meta-cognitive operations."""
        
        # Monitor current cognitive state
        state_vector = self._encode_cognitive_state(context)
        monitoring_output = self.self_monitor(state_vector)
        
        # Select cognitive strategy
        strategy_probs = self.strategy_selector(monitoring_output)
        selected_strategy = torch.argmax(strategy_probs).item()
        
        # Calibrate confidence
        calibrated_confidence = self.confidence_calibrator(monitoring_output)
        
        # Update cognitive state
        context.confidence_level = float(calibrated_confidence)
        context.processing_mode = self._select_processing_mode(strategy_probs)
        
        # Adjust learning rate based on performance
        if self.performance_history:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            context.learning_rate_adjustment = min(2.0, max(0.5, recent_performance * 2))
        
        # Record strategy usage
        self.strategy_history.append(selected_strategy)
        
        return {
            'monitoring_output': monitoring_output,
            'selected_strategy': selected_strategy,
            'strategy_probabilities': strategy_probs,
            'calibrated_confidence': float(calibrated_confidence)
        }, context
    
    def _encode_cognitive_state(self, context: CognitiveState) -> torch.Tensor:
        """Encode cognitive state into a vector."""
        
        state_features = [
            context.cognitive_load,
            context.arousal_level,
            context.confidence_level,
            context.uncertainty,
            context.creativity_level,
            context.exploration_bias,
            context.novelty_seeking,
            context.task_performance,
            context.adaptation_rate,
            context.problem_solving_efficiency,
            len(context.working_memory) / 100.0,  # Normalized
            len(context.attention_focus) / 10.0   # Normalized
        ]
        
        # Pad to meta_dim
        while len(state_features) < self.meta_dim:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:self.meta_dim], dtype=torch.float32)
    
    def _select_processing_mode(self, strategy_probs: torch.Tensor) -> ProcessingMode:
        """Select processing mode based on strategy probabilities."""
        
        # Simplified mapping from strategy to processing mode
        dominant_strategy = torch.argmax(strategy_probs).item()
        
        if dominant_strategy < 3:
            return ProcessingMode.AUTOMATIC
        elif dominant_strategy < 6:
            return ProcessingMode.CONTROLLED
        elif dominant_strategy < 8:
            return ProcessingMode.HYBRID
        else:
            return ProcessingMode.QUANTUM


class CognitiveArchitecture:
    """Next-generation cognitive architecture for autonomous systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize cognitive state
        self.cognitive_state = CognitiveState()
        
        # Initialize processors
        self.processors = {
            CognitiveModule.PERCEPTION: PerceptionProcessor(512, 256),
            CognitiveModule.MEMORY: MemoryProcessor(512, 10000),
            CognitiveModule.REASONING: ReasoningProcessor(512),
            CognitiveModule.METACOGNITION: MetaCognitiveProcessor(256)
        }
        
        # Processing pipeline
        self.processing_pipeline = [
            CognitiveModule.PERCEPTION,
            CognitiveModule.MEMORY,
            CognitiveModule.REASONING,
            CognitiveModule.METACOGNITION
        ]
        
        # Global cognitive controller
        self.global_controller = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(CognitiveModule)),
            nn.Softmax(dim=-1)
        )
        
        # Integration network
        self.integration_network = nn.GRU(512, 512, batch_first=True)
        
        # Performance tracking
        self.performance_tracker = defaultdict(list)
        self.processing_history = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        logger.info("Cognitive architecture initialized with advanced capabilities")
    
    def process_input(self, input_data: Any, processing_mode: Optional[ProcessingMode] = None) -> Dict[str, Any]:
        """Process input through cognitive architecture."""
        
        start_time = time.time()
        
        # Override processing mode if specified
        if processing_mode:
            self.cognitive_state.processing_mode = processing_mode
        
        # Process through pipeline
        current_data = input_data
        processing_results = {}
        
        for module_type in self.processing_pipeline:
            processor = self.processors[module_type]
            
            if processor.is_active:
                try:
                    # Process with current processor
                    if module_type == CognitiveModule.PERCEPTION:
                        result, self.cognitive_state = processor.process(current_data, self.cognitive_state)
                    else:
                        # For other modules, use dict format
                        input_dict = {'data': current_data} if not isinstance(current_data, dict) else current_data
                        result, self.cognitive_state = processor.process(input_dict, self.cognitive_state)
                    
                    processing_results[module_type.value] = result
                    
                    # Update data for next processor
                    if isinstance(result, dict) and 'features' in result:
                        current_data = result['features']
                    elif isinstance(result, dict) and 'conclusion' in result:
                        current_data = result['conclusion']
                    
                except Exception as e:
                    logger.error(f"Error in {module_type.value} processor: {e}")
                    processing_results[module_type.value] = {'error': str(e)}
        
        # Global integration
        integrated_result = self._integrate_results(processing_results)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.performance_tracker['processing_time'].append(processing_time)
        self.processing_history.append({
            'timestamp': start_time,
            'processing_time': processing_time,
            'cognitive_state': self.cognitive_state.to_dict(),
            'results': processing_results
        })
        
        # Update cognitive state performance
        self.cognitive_state.problem_solving_efficiency = min(1.0, 1.0 / (processing_time + 0.1))
        
        return {
            'results': processing_results,
            'integrated_result': integrated_result,
            'cognitive_state': self.cognitive_state.to_dict(),
            'processing_time': processing_time
        }
    
    def _integrate_results(self, processing_results: Dict[str, Any]) -> torch.Tensor:
        """Integrate results from all processors."""
        
        # Collect features from all processors
        feature_tensors = []
        
        for module, result in processing_results.items():
            if isinstance(result, dict):
                if 'features' in result:
                    feature_tensors.append(result['features'])
                elif 'conclusion' in result:
                    feature_tensors.append(result['conclusion'])
                elif 'monitoring_output' in result:
                    feature_tensors.append(result['monitoring_output'])
        
        if not feature_tensors:
            return torch.zeros(512)  # Default output
        
        # Stack and integrate features
        stacked_features = torch.stack(feature_tensors)
        
        # Use GRU for temporal integration
        integrated, _ = self.integration_network(stacked_features.unsqueeze(0))
        
        return integrated.squeeze(0).mean(dim=0)  # Average across sequence
    
    def autonomous_thinking_loop(self, duration_minutes: float = 5.0) -> Dict[str, Any]:
        """Run autonomous thinking and problem-solving loop."""
        
        logger.info(f"Starting autonomous thinking loop for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + duration_minutes * 60
        
        thinking_results = {
            'start_time': start_time,
            'duration_minutes': duration_minutes,
            'thoughts_generated': 0,
            'insights_discovered': 0,
            'problems_solved': 0,
            'creative_outputs': 0
        }
        
        while time.time() < end_time:
            # Generate autonomous thought
            thought = self._generate_autonomous_thought()
            
            # Process the thought
            thinking_result = self.process_input(thought, ProcessingMode.CONTROLLED)
            
            thinking_results['thoughts_generated'] += 1
            
            # Analyze result for insights
            if self._detect_insight(thinking_result):
                thinking_results['insights_discovered'] += 1
                logger.info(f"Insight discovered during autonomous thinking")
            
            # Check for creative output
            if self.cognitive_state.creativity_level > 0.7:
                thinking_results['creative_outputs'] += 1
            
            # Brief pause between thoughts
            time.sleep(0.5)
        
        thinking_results['end_time'] = time.time()
        
        logger.info(
            f"Autonomous thinking completed: {thinking_results['thoughts_generated']} thoughts, "
            f"{thinking_results['insights_discovered']} insights"
        )
        
        return thinking_results
    
    def _generate_autonomous_thought(self) -> torch.Tensor:
        """Generate autonomous thought for processing."""
        
        # Combine current state with random exploration
        thought_components = [
            self.cognitive_state.creativity_level,
            self.cognitive_state.exploration_bias,
            self.cognitive_state.novelty_seeking,
            np.random.random(),  # Random exploration
            np.random.random(),  # Random creativity
            self.cognitive_state.arousal_level
        ]
        
        # Add noise for exploration
        noise = np.random.normal(0, 0.1, 506)  # Fill to 512 dimensions
        
        thought_vector = np.concatenate([thought_components, noise])
        
        return torch.tensor(thought_vector, dtype=torch.float32)
    
    def _detect_insight(self, thinking_result: Dict[str, Any]) -> bool:
        """Detect if a thinking result contains an insight."""
        
        # Simple heuristic for insight detection
        cognitive_state = thinking_result.get('cognitive_state', {})
        
        # High confidence + high novelty + low uncertainty = potential insight
        confidence = cognitive_state.get('confidence_level', 0.0)
        novelty = self.cognitive_state.working_memory.get('novelty_score', 0.0)
        uncertainty = cognitive_state.get('uncertainty', 1.0)
        
        insight_score = confidence * novelty * (1.0 - uncertainty)
        
        return insight_score > 0.6  # Threshold for insight detection
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of cognitive architecture state."""
        
        summary = {
            'cognitive_state': self.cognitive_state.to_dict(),
            'processor_performance': {},
            'processing_statistics': {},
            'recent_performance': {}
        }
        
        # Processor performance
        for module, processor in self.processors.items():
            summary['processor_performance'][module.value] = processor.get_performance_metrics()
        
        # Processing statistics
        if self.performance_tracker['processing_time']:
            times = self.performance_tracker['processing_time']
            summary['processing_statistics'] = {
                'average_processing_time': np.mean(times),
                'processing_time_std': np.std(times),
                'total_processes': len(times),
                'throughput_per_second': len(times) / (sum(times) + 1e-8)
            }
        
        # Recent performance trends
        if len(self.processing_history) >= 10:
            recent_states = [entry['cognitive_state'] for entry in list(self.processing_history)[-10:]]
            
            summary['recent_performance'] = {
                'confidence_trend': np.mean([state['confidence_level'] for state in recent_states]),
                'creativity_trend': np.mean([state['creativity_level'] for state in recent_states]),
                'performance_trend': np.mean([state['task_performance'] for state in recent_states]),
                'cognitive_load_trend': np.mean([state['cognitive_load'] for state in recent_states])
            }
        
        return summary
    
    def shutdown(self):
        """Shutdown cognitive architecture gracefully."""
        
        logger.info("Shutting down cognitive architecture")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Deactivate all processors
        for processor in self.processors.values():
            processor.is_active = False
        
        logger.info("Cognitive architecture shutdown complete")


def create_cognitive_architecture(config: Optional[Dict[str, Any]] = None) -> CognitiveArchitecture:
    """Factory function to create cognitive architecture."""
    
    return CognitiveArchitecture(config)
