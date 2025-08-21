"""Autonomous Discovery Engine for Self-Directed Materials Research."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import concurrent.futures
from collections import deque
import logging
import json
import hashlib

from .quantum_evolutionary_optimizer import QuantumEvolutionaryOptimizer, QuantumGenome
from ..models.quantum_consciousness_bridge import HyperDimensionalMaterialsExplorer
from ..models.consciousness_aware import ConsciousMaterialsExplorer
from ..autonomous.self_evolving_ai import SelfImprovingSystem


logger = logging.getLogger(__name__)


class DiscoveryStrategy(Enum):
    """Discovery strategies for autonomous exploration."""
    
    QUANTUM_EVOLUTIONARY = "quantum_evolutionary"
    CONSCIOUSNESS_DRIVEN = "consciousness_driven"
    HYPERDIMENSIONAL = "hyperdimensional"
    HYBRID_MULTIVERSE = "hybrid_multiverse"
    SERENDIPITY_SEARCH = "serendipity_search"


@dataclass
class ExplorationMetrics:
    """Metrics for tracking exploration progress."""
    
    # Discovery metrics
    materials_explored: int = 0
    novel_materials_discovered: int = 0
    breakthrough_materials: int = 0
    property_targets_achieved: int = 0
    
    # Efficiency metrics
    exploration_time: float = 0.0
    convergence_rate: float = 0.0
    success_rate: float = 0.0
    
    # Quality metrics
    average_property_accuracy: float = 0.0
    best_property_accuracy: float = 0.0
    diversity_score: float = 0.0
    
    # AI metrics
    quantum_advantage: float = 0.0
    consciousness_complexity: float = 0.0
    evolutionary_fitness: float = 0.0
    
    # Research metrics
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    scientific_insights: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'materials_explored': self.materials_explored,
            'novel_materials_discovered': self.novel_materials_discovered,
            'breakthrough_materials': self.breakthrough_materials,
            'property_targets_achieved': self.property_targets_achieved,
            'exploration_time': self.exploration_time,
            'convergence_rate': self.convergence_rate,
            'success_rate': self.success_rate,
            'average_property_accuracy': self.average_property_accuracy,
            'best_property_accuracy': self.best_property_accuracy,
            'diversity_score': self.diversity_score,
            'quantum_advantage': self.quantum_advantage,
            'consciousness_complexity': self.consciousness_complexity,
            'evolutionary_fitness': self.evolutionary_fitness,
            'hypotheses_generated': self.hypotheses_generated,
            'hypotheses_validated': self.hypotheses_validated,
            'scientific_insights': self.scientific_insights
        }


@dataclass
class ResearchHypothesis:
    """Scientific hypothesis for materials discovery."""
    
    hypothesis_id: str
    description: str
    target_properties: Dict[str, float]
    predicted_materials: List[torch.Tensor]
    confidence_level: float
    
    # Evidence tracking
    supporting_evidence: List[Dict] = field(default_factory=list)
    contradicting_evidence: List[Dict] = field(default_factory=list)
    
    # Validation status
    validation_status: str = "proposed"  # proposed, testing, validated, refuted
    validation_score: float = 0.0
    
    # Research context
    discovery_strategy: DiscoveryStrategy = DiscoveryStrategy.CONSCIOUSNESS_DRIVEN
    generation_timestamp: float = field(default_factory=time.time)
    
    def get_evidence_balance(self) -> float:
        """Calculate balance between supporting and contradicting evidence."""
        support_weight = sum(e.get('weight', 1.0) for e in self.supporting_evidence)
        contradict_weight = sum(e.get('weight', 1.0) for e in self.contradicting_evidence)
        
        total_weight = support_weight + contradict_weight
        if total_weight == 0:
            return 0.5  # Neutral if no evidence
        
        return support_weight / total_weight


class AutonomousHypothesisGenerator(nn.Module):
    """Generates scientific hypotheses for materials discovery."""
    
    def __init__(self, material_dim: int, property_dim: int):
        super().__init__()
        
        self.material_dim = material_dim
        self.property_dim = property_dim
        
        # Hypothesis generation network
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(property_dim + 10, 256),  # +10 for context
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, material_dim + property_dim + 1)  # Materials + properties + confidence
        )
        
        # Context encoder for domain knowledge
        self.context_encoder = nn.Sequential(
            nn.Linear(material_dim * 3, 128),  # Historical materials
            nn.ReLU(),
            nn.Linear(128, 10)  # Context vector
        )
        
        # Novelty detector
        self.novelty_detector = nn.Sequential(
            nn.Linear(material_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, target_properties: torch.Tensor,
                historical_materials: Optional[List[torch.Tensor]] = None) -> ResearchHypothesis:
        """Generate research hypothesis for target properties."""
        
        # Encode context from historical materials
        if historical_materials and len(historical_materials) >= 3:
            context_input = torch.cat(historical_materials[:3])
            context_vector = self.context_encoder(context_input)
        else:
            context_vector = torch.zeros(10)
        
        # Generate hypothesis
        hypothesis_input = torch.cat([target_properties, context_vector])
        hypothesis_output = self.hypothesis_generator(hypothesis_input)
        
        # Split output
        predicted_material = hypothesis_output[:self.material_dim]
        refined_properties = hypothesis_output[self.material_dim:self.material_dim + self.property_dim]
        confidence = torch.sigmoid(hypothesis_output[-1]).item()
        
        # Assess novelty
        novelty_score = self.novelty_detector(predicted_material).item()
        
        # Create hypothesis
        hypothesis_id = self._generate_hypothesis_id(predicted_material, refined_properties)
        
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            description=f"Material with enhanced properties: {refined_properties.tolist()}",
            target_properties={f"prop_{i}": val.item() for i, val in enumerate(refined_properties)},
            predicted_materials=[predicted_material],
            confidence_level=confidence * novelty_score,  # Confidence weighted by novelty
            discovery_strategy=DiscoveryStrategy.CONSCIOUSNESS_DRIVEN
        )
        
        return hypothesis
    
    def _generate_hypothesis_id(self, material: torch.Tensor, properties: torch.Tensor) -> str:
        """Generate unique hypothesis ID."""
        material_hash = hashlib.md5(material.numpy().tobytes()).hexdigest()[:8]
        property_hash = hashlib.md5(properties.numpy().tobytes()).hexdigest()[:8]
        return f"hyp_{material_hash}_{property_hash}"


class SerendipityEngine(nn.Module):
    """Engine for serendipitous discovery through unexpected combinations."""
    
    def __init__(self, material_dim: int):
        super().__init__()
        
        self.material_dim = material_dim
        
        # Unexpected combination generator
        self.combination_generator = nn.Sequential(
            nn.Linear(material_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Adds randomness for serendipity
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, material_dim),
            nn.Tanh()
        )
        
        # Surprise detector
        self.surprise_detector = nn.Sequential(
            nn.Linear(material_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Serendipity memory
        self.serendipity_memory = deque(maxlen=1000)
        
    def forward(self, material1: torch.Tensor, material2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate serendipitous combination of two materials."""
        
        # Create unexpected combination
        combination_input = torch.cat([material1, material2])
        serendipitous_material = self.combination_generator(combination_input)
        
        # Measure surprise level
        surprise_level = self.surprise_detector(serendipitous_material)
        
        # Store in serendipity memory if surprising enough
        if surprise_level.item() > 0.7:
            self.serendipity_memory.append({
                'material': serendipitous_material.clone(),
                'parents': [material1.clone(), material2.clone()],
                'surprise_level': surprise_level.item(),
                'timestamp': time.time()
            })
        
        return {
            'serendipitous_material': serendipitous_material,
            'surprise_level': surprise_level,
            'novelty_score': self._calculate_novelty(serendipitous_material)
        }
    
    def _calculate_novelty(self, material: torch.Tensor) -> torch.Tensor:
        """Calculate novelty based on distance from memory."""
        
        if len(self.serendipity_memory) == 0:
            return torch.tensor(1.0)  # Maximum novelty if no history
        
        # Calculate minimum distance to existing materials
        min_distance = float('inf')
        for memory_item in self.serendipity_memory:
            distance = torch.norm(material - memory_item['material']).item()
            min_distance = min(min_distance, distance)
        
        # Convert distance to novelty score
        novelty = min(1.0, min_distance / 2.0)  # Normalize to [0,1]
        return torch.tensor(novelty)
    
    def get_surprising_discoveries(self) -> List[Dict]:
        """Get most surprising discoveries from memory."""
        
        if not self.serendipity_memory:
            return []
        
        # Sort by surprise level
        sorted_discoveries = sorted(
            self.serendipity_memory, 
            key=lambda x: x['surprise_level'], 
            reverse=True
        )
        
        return sorted_discoveries[:10]  # Top 10 most surprising


class AutonomousDiscoveryEngine:
    """Autonomous engine for materials discovery with multiple AI strategies."""
    
    def __init__(self, material_dim: int, property_dim: int,
                 discovery_config: Optional[Dict[str, Any]] = None):
        
        self.material_dim = material_dim
        self.property_dim = property_dim
        
        # Configuration
        self.config = discovery_config or {
            'max_exploration_time': 3600,  # 1 hour
            'breakthrough_threshold': 0.95,
            'novelty_threshold': 0.8,
            'hypothesis_validation_threshold': 0.7,
            'serendipity_frequency': 0.1,
            'parallel_strategies': True
        }
        
        # AI Discovery Components
        self.quantum_evolutionary_optimizer = QuantumEvolutionaryOptimizer(
            material_dim, property_dim, population_size=50
        )
        
        self.consciousness_explorer = ConsciousMaterialsExplorer(
            material_dim, property_dim
        )
        
        self.hyperdimensional_explorer = HyperDimensionalMaterialsExplorer(
            material_dim, property_dim
        )
        
        self.self_improving_system = SelfImprovingSystem(
            material_dim, property_dim
        )
        
        self.hypothesis_generator = AutonomousHypothesisGenerator(
            material_dim, property_dim
        )
        
        self.serendipity_engine = SerendipityEngine(material_dim)
        
        # Discovery state
        self.active_discoveries: List[Dict] = []
        self.research_hypotheses: List[ResearchHypothesis] = []
        self.discovered_materials: List[Dict] = []
        self.exploration_metrics = ExplorationMetrics()
        
        # Discovery memory
        self.discovery_history = deque(maxlen=10000)
        self.breakthrough_materials = deque(maxlen=100)
        
        # Threading for autonomous operation
        self.discovery_thread: Optional[threading.Thread] = None
        self.discovery_active = False
        self.discovery_lock = threading.Lock()
        
        logger.info("Autonomous Discovery Engine initialized")
    
    def start_autonomous_discovery(self, target_properties_list: List[torch.Tensor],
                                 discovery_duration: int = 3600) -> None:
        """Start autonomous discovery process."""
        
        if self.discovery_active:
            logger.warning("Discovery already active")
            return
        
        self.discovery_active = True
        
        def discovery_loop():
            """Main discovery loop running in background."""
            
            start_time = time.time()
            
            logger.info(f"🔬 Starting autonomous discovery for {discovery_duration}s")
            
            while (self.discovery_active and 
                   time.time() - start_time < discovery_duration):
                
                try:
                    # Run discovery cycle for each target
                    for target_properties in target_properties_list:
                        if not self.discovery_active:
                            break
                        
                        discovery_result = self.run_discovery_cycle(
                            target_properties, cycle_duration=60
                        )
                        
                        # Process results
                        self._process_discovery_results(discovery_result)
                        
                        # Brief pause between targets
                        time.sleep(5)
                    
                    # Longer pause between full cycles
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Error in discovery loop: {e}")
                    time.sleep(60)  # Wait before retrying
            
            logger.info("🏁 Autonomous discovery completed")
            self.discovery_active = False
        
        self.discovery_thread = threading.Thread(target=discovery_loop, daemon=True)
        self.discovery_thread.start()
    
    def stop_autonomous_discovery(self) -> None:
        """Stop autonomous discovery process."""
        
        self.discovery_active = False
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=10)
        
        logger.info("⏹️ Autonomous discovery stopped")
    
    def run_discovery_cycle(self, target_properties: torch.Tensor,
                          cycle_duration: int = 300) -> Dict[str, Any]:
        """Run one cycle of autonomous discovery."""
        
        cycle_start_time = time.time()
        cycle_results = {
            'target_properties': target_properties,
            'strategies_used': [],
            'materials_discovered': [],
            'hypotheses_generated': [],
            'breakthroughs': [],
            'metrics_update': {}
        }
        
        logger.info(f"🧪 Discovery cycle started for properties: {target_properties.tolist()}")
        
        # Strategy selection based on context
        strategies = self._select_discovery_strategies(target_properties)
        
        if self.config.get('parallel_strategies', True):
            # Parallel execution of strategies
            cycle_results = self._run_parallel_strategies(
                strategies, target_properties, cycle_duration
            )
        else:
            # Sequential execution
            cycle_results = self._run_sequential_strategies(
                strategies, target_properties, cycle_duration
            )
        
        # Generate research hypotheses
        new_hypothesis = self.hypothesis_generator(
            target_properties, 
            [m['material'] for m in self.discovered_materials[-10:]]
        )
        self.research_hypotheses.append(new_hypothesis)
        cycle_results['hypotheses_generated'].append(new_hypothesis)
        
        # Serendipity exploration
        if random.random() < self.config.get('serendipity_frequency', 0.1):
            serendipity_results = self._explore_serendipity(target_properties)
            cycle_results['materials_discovered'].extend(serendipity_results)
        
        # Validate existing hypotheses
        self._validate_hypotheses(cycle_results['materials_discovered'])
        
        # Update metrics
        cycle_time = time.time() - cycle_start_time
        self._update_exploration_metrics(cycle_results, cycle_time)
        
        return cycle_results
    
    def _select_discovery_strategies(self, target_properties: torch.Tensor) -> List[DiscoveryStrategy]:
        """Select optimal discovery strategies based on target properties."""
        
        strategies = []
        
        # Always include consciousness-driven exploration
        strategies.append(DiscoveryStrategy.CONSCIOUSNESS_DRIVEN)
        
        # Add quantum evolutionary if exploring complex property space
        if torch.norm(target_properties).item() > 1.0:
            strategies.append(DiscoveryStrategy.QUANTUM_EVOLUTIONARY)
        
        # Add hyperdimensional for high-dimensional targets
        if len(target_properties) > 5:
            strategies.append(DiscoveryStrategy.HYPERDIMENSIONAL)
        
        # Occasionally use serendipity
        if random.random() < 0.2:
            strategies.append(DiscoveryStrategy.SERENDIPITY_SEARCH)
        
        # Use hybrid multiverse for complex multi-objective problems
        if len(target_properties) > 3 and torch.var(target_properties).item() > 0.5:
            strategies.append(DiscoveryStrategy.HYBRID_MULTIVERSE)
        
        return strategies
    
    def _run_parallel_strategies(self, strategies: List[DiscoveryStrategy],
                               target_properties: torch.Tensor,
                               cycle_duration: int) -> Dict[str, Any]:
        """Run discovery strategies in parallel."""
        
        results = {
            'target_properties': target_properties,
            'strategies_used': strategies,
            'materials_discovered': [],
            'hypotheses_generated': [],
            'breakthroughs': [],
            'metrics_update': {}
        }
        
        # Execute strategies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            future_to_strategy = {
                executor.submit(
                    self._execute_discovery_strategy, 
                    strategy, 
                    target_properties, 
                    cycle_duration // len(strategies)
                ): strategy
                for strategy in strategies
            }
            
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    strategy_result = future.result()
                    results['materials_discovered'].extend(strategy_result.get('materials', []))
                    results['hypotheses_generated'].extend(strategy_result.get('hypotheses', []))
                    results['breakthroughs'].extend(strategy_result.get('breakthroughs', []))
                    
                    logger.info(f"✅ {strategy.value} completed: {len(strategy_result.get('materials', []))} materials")
                    
                except Exception as e:
                    logger.error(f"❌ {strategy.value} failed: {e}")
        
        return results
    
    def _run_sequential_strategies(self, strategies: List[DiscoveryStrategy],
                                 target_properties: torch.Tensor,
                                 cycle_duration: int) -> Dict[str, Any]:
        """Run discovery strategies sequentially."""
        
        results = {
            'target_properties': target_properties,
            'strategies_used': strategies,
            'materials_discovered': [],
            'hypotheses_generated': [],
            'breakthroughs': [],
            'metrics_update': {}
        }
        
        time_per_strategy = cycle_duration // max(len(strategies), 1)
        
        for strategy in strategies:
            try:
                strategy_result = self._execute_discovery_strategy(
                    strategy, target_properties, time_per_strategy
                )
                
                results['materials_discovered'].extend(strategy_result.get('materials', []))
                results['hypotheses_generated'].extend(strategy_result.get('hypotheses', []))
                results['breakthroughs'].extend(strategy_result.get('breakthroughs', []))
                
                logger.info(f"✅ {strategy.value} completed: {len(strategy_result.get('materials', []))} materials")
                
            except Exception as e:
                logger.error(f"❌ {strategy.value} failed: {e}")
        
        return results
    
    def _execute_discovery_strategy(self, strategy: DiscoveryStrategy,
                                  target_properties: torch.Tensor,
                                  time_limit: int) -> Dict[str, Any]:
        """Execute specific discovery strategy."""
        
        start_time = time.time()
        
        if strategy == DiscoveryStrategy.QUANTUM_EVOLUTIONARY:
            return self._quantum_evolutionary_discovery(target_properties, time_limit)
        
        elif strategy == DiscoveryStrategy.CONSCIOUSNESS_DRIVEN:
            return self._consciousness_driven_discovery(target_properties, time_limit)
        
        elif strategy == DiscoveryStrategy.HYPERDIMENSIONAL:
            return self._hyperdimensional_discovery(target_properties, time_limit)
        
        elif strategy == DiscoveryStrategy.HYBRID_MULTIVERSE:
            return self._hybrid_multiverse_discovery(target_properties, time_limit)
        
        elif strategy == DiscoveryStrategy.SERENDIPITY_SEARCH:
            return self._serendipity_search_discovery(target_properties, time_limit)
        
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _quantum_evolutionary_discovery(self, target_properties: torch.Tensor,
                                      time_limit: int) -> Dict[str, Any]:
        """Quantum evolutionary discovery strategy."""
        
        # Initialize with recent discoveries as seeds
        seed_materials = [m['material'] for m in self.discovered_materials[-10:]]
        if not seed_materials:
            seed_materials = None
        
        # Run evolution for limited generations based on time
        max_generations = min(20, time_limit // 10)  # 10 seconds per generation
        
        try:
            evolution_result = self.quantum_evolutionary_optimizer.evolve_complete(
                target_properties, 
                max_generations=max_generations,
                convergence_threshold=1e-4
            )
            
            discovered_materials = []
            if evolution_result['best_genome']:
                best_genome = evolution_result['best_genome']
                material_tensor = torch.tensor([
                    best_genome.classical_genes.get(f'material_param_{i}', 0.0)
                    for i in range(self.material_dim)
                ], dtype=torch.float32)
                
                discovered_materials.append({
                    'material': material_tensor,
                    'predicted_properties': target_properties,  # Simplified
                    'fitness': best_genome.fitness,
                    'quantum_advantage': best_genome.quantum_advantage,
                    'discovery_strategy': DiscoveryStrategy.QUANTUM_EVOLUTIONARY,
                    'discovery_timestamp': time.time()
                })
            
            return {
                'materials': discovered_materials,
                'hypotheses': [],
                'breakthroughs': self._identify_breakthroughs(discovered_materials),
                'evolution_result': evolution_result
            }
            
        except Exception as e:
            logger.error(f"Quantum evolutionary discovery error: {e}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _consciousness_driven_discovery(self, target_properties: torch.Tensor,
                                      time_limit: int) -> Dict[str, Any]:
        """Consciousness-driven discovery strategy."""
        
        try:
            exploration_budget = min(100, time_limit // 5)  # 5 seconds per exploration
            
            consciousness_result = self.consciousness_explorer.explore_materials_space(
                target_properties, exploration_budget=exploration_budget
            )
            
            discovered_materials = []
            if consciousness_result.get('best_materials'):
                for material_data in consciousness_result['best_materials'][:5]:  # Top 5
                    discovered_materials.append({
                        'material': material_data['material'],
                        'predicted_properties': material_data['properties'],
                        'total_score': material_data.get('total_score', 0.0),
                        'consciousness_complexity': material_data.get('consciousness_complexity', 0.0),
                        'discovery_strategy': DiscoveryStrategy.CONSCIOUSNESS_DRIVEN,
                        'discovery_timestamp': time.time()
                    })
            
            return {
                'materials': discovered_materials,
                'hypotheses': [],
                'breakthroughs': self._identify_breakthroughs(discovered_materials),
                'consciousness_result': consciousness_result
            }
            
        except Exception as e:
            logger.error(f"Consciousness-driven discovery error: {e}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _hyperdimensional_discovery(self, target_properties: torch.Tensor,
                                  time_limit: int) -> Dict[str, Any]:
        """Hyperdimensional discovery strategy."""
        
        try:
            exploration_universes = min(5, time_limit // 20)  # 20 seconds per universe
            
            hyperdim_result = self.hyperdimensional_explorer.explore_hyperdimensional_space(
                target_properties, 
                exploration_universes=exploration_universes,
                consciousness_depth=3
            )
            
            discovered_materials = []
            if hyperdim_result.get('optimal_materials'):
                for material_data in hyperdim_result['optimal_materials'][:5]:  # Top 5
                    discovered_materials.append({
                        'material': material_data['material'],
                        'predicted_properties': material_data['properties'],
                        'total_score': material_data['total_score'],
                        'universe_id': material_data.get('universe_id', 0),
                        'discovery_strategy': DiscoveryStrategy.HYPERDIMENSIONAL,
                        'discovery_timestamp': time.time()
                    })
            
            return {
                'materials': discovered_materials,
                'hypotheses': [],
                'breakthroughs': self._identify_breakthroughs(discovered_materials),
                'hyperdimensional_result': hyperdim_result
            }
            
        except Exception as e:
            logger.error(f"Hyperdimensional discovery error: {e}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _hybrid_multiverse_discovery(self, target_properties: torch.Tensor,
                                   time_limit: int) -> Dict[str, Any]:
        """Hybrid multiverse discovery combining multiple strategies."""
        
        try:
            # Combine quantum evolution and consciousness exploration
            half_time = time_limit // 2
            
            quantum_result = self._quantum_evolutionary_discovery(target_properties, half_time)
            consciousness_result = self._consciousness_driven_discovery(target_properties, half_time)
            
            # Merge results
            all_materials = quantum_result['materials'] + consciousness_result['materials']
            all_breakthroughs = quantum_result['breakthroughs'] + consciousness_result['breakthroughs']
            
            # Select best materials from both strategies
            best_materials = sorted(
                all_materials, 
                key=lambda x: x.get('total_score', x.get('fitness', 0.0)), 
                reverse=True
            )[:10]
            
            return {
                'materials': best_materials,
                'hypotheses': [],
                'breakthroughs': all_breakthroughs,
                'hybrid_components': {
                    'quantum_result': quantum_result,
                    'consciousness_result': consciousness_result
                }
            }
            
        except Exception as e:
            logger.error(f"Hybrid multiverse discovery error: {e}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _serendipity_search_discovery(self, target_properties: torch.Tensor,
                                    time_limit: int) -> Dict[str, Any]:
        """Serendipity-based discovery strategy."""
        
        try:
            discovered_materials = []
            
            # Get recent materials for serendipitous combinations
            recent_materials = [m['material'] for m in self.discovered_materials[-20:]]
            
            if len(recent_materials) >= 2:
                num_combinations = min(10, time_limit // 5)  # 5 seconds per combination
                
                for _ in range(num_combinations):
                    # Random combination of materials
                    mat1, mat2 = random.sample(recent_materials, 2)
                    
                    serendipity_result = self.serendipity_engine(mat1, mat2)
                    
                    # Only keep surprising discoveries
                    if serendipity_result['surprise_level'].item() > 0.6:
                        discovered_materials.append({
                            'material': serendipity_result['serendipitous_material'],
                            'predicted_properties': target_properties,  # Estimated
                            'surprise_level': serendipity_result['surprise_level'].item(),
                            'novelty_score': serendipity_result['novelty_score'].item(),
                            'parent_materials': [mat1, mat2],
                            'discovery_strategy': DiscoveryStrategy.SERENDIPITY_SEARCH,
                            'discovery_timestamp': time.time()
                        })
            
            return {
                'materials': discovered_materials,
                'hypotheses': [],
                'breakthroughs': self._identify_breakthroughs(discovered_materials),
                'surprising_discoveries': self.serendipity_engine.get_surprising_discoveries()
            }
            
        except Exception as e:
            logger.error(f"Serendipity search discovery error: {e}")
            return {'materials': [], 'hypotheses': [], 'breakthroughs': []}
    
    def _explore_serendipity(self, target_properties: torch.Tensor) -> List[Dict]:
        """Explore serendipitous combinations."""
        
        serendipitous_materials = []
        
        if len(self.discovered_materials) >= 2:
            # Try 5 random combinations
            for _ in range(5):
                mat1_data = random.choice(self.discovered_materials)
                mat2_data = random.choice(self.discovered_materials)
                
                if mat1_data != mat2_data:
                    serendipity_result = self.serendipity_engine(
                        mat1_data['material'], mat2_data['material']
                    )
                    
                    if serendipity_result['surprise_level'].item() > 0.5:
                        serendipitous_materials.append({
                            'material': serendipity_result['serendipitous_material'],
                            'predicted_properties': target_properties,
                            'surprise_level': serendipity_result['surprise_level'].item(),
                            'discovery_strategy': DiscoveryStrategy.SERENDIPITY_SEARCH,
                            'discovery_timestamp': time.time()
                        })
        
        return serendipitous_materials
    
    def _identify_breakthroughs(self, discovered_materials: List[Dict]) -> List[Dict]:
        """Identify breakthrough materials based on criteria."""
        
        breakthroughs = []
        breakthrough_threshold = self.config.get('breakthrough_threshold', 0.95)
        
        for material_data in discovered_materials:
            # Check if material meets breakthrough criteria
            score = material_data.get('total_score', material_data.get('fitness', 0.0))
            novelty = material_data.get('novelty_score', 0.0)
            surprise = material_data.get('surprise_level', 0.0)
            
            # Multiple breakthrough criteria
            is_breakthrough = (
                score > breakthrough_threshold or
                novelty > 0.8 or
                surprise > 0.8
            )
            
            if is_breakthrough:
                breakthrough_data = material_data.copy()
                breakthrough_data['breakthrough_reason'] = self._determine_breakthrough_reason(
                    score, novelty, surprise
                )
                breakthrough_data['breakthrough_timestamp'] = time.time()
                
                breakthroughs.append(breakthrough_data)
                self.breakthrough_materials.append(breakthrough_data)
                
                logger.info(f"🚀 Breakthrough material discovered: {breakthrough_data['breakthrough_reason']}")
        
        return breakthroughs
    
    def _determine_breakthrough_reason(self, score: float, novelty: float, surprise: float) -> str:
        """Determine reason for breakthrough classification."""
        
        reasons = []
        
        if score > 0.95:
            reasons.append("exceptional_performance")
        if novelty > 0.8:
            reasons.append("high_novelty")
        if surprise > 0.8:
            reasons.append("serendipitous_discovery")
        
        return "_".join(reasons) if reasons else "unknown"
    
    def _validate_hypotheses(self, new_materials: List[Dict]) -> None:
        """Validate research hypotheses against new discoveries."""
        
        validation_threshold = self.config.get('hypothesis_validation_threshold', 0.7)
        
        for hypothesis in self.research_hypotheses:
            if hypothesis.validation_status == "proposed":
                # Check if new materials support or contradict hypothesis
                for material_data in new_materials:
                    material = material_data['material']
                    properties = material_data.get('predicted_properties', torch.zeros(self.property_dim))
                    
                    # Calculate similarity to hypothesis predictions
                    if hypothesis.predicted_materials:
                        predicted_material = hypothesis.predicted_materials[0]
                        material_similarity = 1.0 / (1.0 + torch.norm(material - predicted_material).item())
                        
                        # Check property match
                        target_props = torch.tensor(list(hypothesis.target_properties.values()))
                        property_similarity = 1.0 / (1.0 + torch.norm(properties - target_props).item())
                        
                        overall_similarity = (material_similarity + property_similarity) / 2.0
                        
                        # Add evidence
                        evidence = {
                            'material': material,
                            'properties': properties,
                            'similarity': overall_similarity,
                            'timestamp': time.time(),
                            'weight': overall_similarity
                        }
                        
                        if overall_similarity > validation_threshold:
                            hypothesis.supporting_evidence.append(evidence)
                        else:
                            hypothesis.contradicting_evidence.append(evidence)
                
                # Update hypothesis validation status
                evidence_balance = hypothesis.get_evidence_balance()
                
                if len(hypothesis.supporting_evidence) + len(hypothesis.contradicting_evidence) >= 3:
                    if evidence_balance > 0.7:
                        hypothesis.validation_status = "validated"
                        hypothesis.validation_score = evidence_balance
                        logger.info(f"✅ Hypothesis {hypothesis.hypothesis_id} validated")
                    elif evidence_balance < 0.3:
                        hypothesis.validation_status = "refuted"
                        hypothesis.validation_score = evidence_balance
                        logger.info(f"❌ Hypothesis {hypothesis.hypothesis_id} refuted")
                    else:
                        hypothesis.validation_status = "testing"
    
    def _process_discovery_results(self, discovery_result: Dict[str, Any]) -> None:
        """Process and store discovery results."""
        
        with self.discovery_lock:
            # Store discovered materials
            for material_data in discovery_result['materials_discovered']:
                self.discovered_materials.append(material_data)
                self.discovery_history.append({
                    'type': 'material_discovery',
                    'data': material_data,
                    'timestamp': time.time()
                })
            
            # Store breakthroughs
            for breakthrough in discovery_result['breakthroughs']:
                self.discovery_history.append({
                    'type': 'breakthrough',
                    'data': breakthrough,
                    'timestamp': time.time()
                })
            
            # Update metrics
            self.exploration_metrics.materials_explored += len(discovery_result['materials_discovered'])
            self.exploration_metrics.breakthrough_materials += len(discovery_result['breakthroughs'])
            self.exploration_metrics.hypotheses_generated += len(discovery_result['hypotheses_generated'])
    
    def _update_exploration_metrics(self, cycle_results: Dict[str, Any], cycle_time: float) -> None:
        """Update exploration metrics."""
        
        materials_discovered = len(cycle_results['materials_discovered'])
        breakthroughs = len(cycle_results['breakthroughs'])
        
        # Update basic metrics
        self.exploration_metrics.materials_explored += materials_discovered
        self.exploration_metrics.breakthrough_materials += breakthroughs
        self.exploration_metrics.exploration_time += cycle_time
        
        # Calculate rates
        if self.exploration_metrics.materials_explored > 0:
            self.exploration_metrics.success_rate = (
                self.exploration_metrics.breakthrough_materials / 
                self.exploration_metrics.materials_explored
            )
        
        # Novel materials (simplified - based on recent discoveries)
        recent_count = len([m for m in self.discovered_materials[-10:]])
        self.exploration_metrics.novel_materials_discovered = recent_count
        
        # Calculate averages
        if cycle_results['materials_discovered']:
            scores = [
                m.get('total_score', m.get('fitness', 0.0)) 
                for m in cycle_results['materials_discovered']
            ]
            if scores:
                self.exploration_metrics.average_property_accuracy = np.mean(scores)
                self.exploration_metrics.best_property_accuracy = max(scores)
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get comprehensive discovery summary."""
        
        return {
            'exploration_metrics': self.exploration_metrics.to_dict(),
            'discovered_materials_count': len(self.discovered_materials),
            'breakthrough_materials_count': len(self.breakthrough_materials),
            'active_hypotheses_count': len([h for h in self.research_hypotheses if h.validation_status == "proposed"]),
            'validated_hypotheses_count': len([h for h in self.research_hypotheses if h.validation_status == "validated"]),
            'discovery_history_length': len(self.discovery_history),
            'discovery_active': self.discovery_active,
            'recent_breakthroughs': list(self.breakthrough_materials)[-5:],
            'surprising_discoveries': self.serendipity_engine.get_surprising_discoveries()
        }
    
    def export_discovery_data(self, filepath: str) -> None:
        """Export discovery data to file."""
        
        export_data = {
            'timestamp': time.time(),
            'exploration_metrics': self.exploration_metrics.to_dict(),
            'discovered_materials': [
                {
                    'material': m['material'].tolist(),
                    'predicted_properties': m.get('predicted_properties', torch.zeros(self.property_dim)).tolist(),
                    'discovery_strategy': m.get('discovery_strategy', 'unknown').value if hasattr(m.get('discovery_strategy'), 'value') else str(m.get('discovery_strategy', 'unknown')),
                    'discovery_timestamp': m.get('discovery_timestamp', 0.0),
                    'scores': {k: v for k, v in m.items() if isinstance(v, (int, float))}
                }
                for m in self.discovered_materials
            ],
            'research_hypotheses': [
                {
                    'hypothesis_id': h.hypothesis_id,
                    'description': h.description,
                    'target_properties': h.target_properties,
                    'confidence_level': h.confidence_level,
                    'validation_status': h.validation_status,
                    'validation_score': h.validation_score,
                    'evidence_balance': h.get_evidence_balance(),
                    'discovery_strategy': h.discovery_strategy.value
                }
                for h in self.research_hypotheses
            ],
            'breakthrough_materials': [
                {
                    'material': b['material'].tolist(),
                    'breakthrough_reason': b.get('breakthrough_reason', 'unknown'),
                    'breakthrough_timestamp': b.get('breakthrough_timestamp', 0.0),
                    'scores': {k: v for k, v in b.items() if isinstance(v, (int, float))}
                }
                for b in self.breakthrough_materials
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Discovery data exported to {filepath}")