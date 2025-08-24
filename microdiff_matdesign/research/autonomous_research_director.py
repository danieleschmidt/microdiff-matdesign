"""Autonomous Research Director for Advanced Scientific Discovery."""

import torch
import torch.nn as nn
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
from abc import ABC, abstractmethod

from .benchmarking import BenchmarkSuite, BenchmarkConfig
from .publication import PublicationManager, PublicationConfig
from .reproducibility import ReproducibilityManager
from ..evolution.autonomous_discovery_engine import AutonomousDiscoveryEngine, ResearchHypothesis
from ..autonomous.cognitive_architecture import CognitiveArchitecture, ProcessingMode
from ..autonomous.adaptive_learning_system import AdaptiveLearningSystem


logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """Phases of autonomous research process."""
    
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LITERATURE_REVIEW = "literature_review"
    EXPERIMENTAL_DESIGN = "experimental_design"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    PUBLICATION = "publication"
    FOLLOW_UP = "follow_up"


class ResearchStrategy(Enum):
    """Research strategies for different types of investigations."""
    
    EXPLORATORY = "exploratory"           # Open-ended exploration
    CONFIRMATORY = "confirmatory"         # Hypothesis testing
    COMPARATIVE = "comparative"           # Comparing alternatives
    LONGITUDINAL = "longitudinal"         # Time-series analysis
    META_ANALYSIS = "meta_analysis"       # Synthesizing existing work
    BREAKTHROUGH = "breakthrough"         # Revolutionary discoveries
    INTERDISCIPLINARY = "interdisciplinary" # Cross-domain research


@dataclass
class ResearchProject:
    """Research project specification and tracking."""
    
    project_id: str
    title: str
    description: str
    research_questions: List[str]
    hypotheses: List[ResearchHypothesis]
    
    # Project metadata
    domain: str = "materials_science"
    priority: float = 0.5
    expected_duration_days: int = 30
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Progress tracking
    current_phase: ResearchPhase = ResearchPhase.HYPOTHESIS_GENERATION
    completion_percentage: float = 0.0
    milestones_completed: List[str] = field(default_factory=list)
    
    # Results
    findings: List[Dict[str, Any]] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Collaboration
    collaborators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary representation."""
        return {
            'project_id': self.project_id,
            'title': self.title,
            'description': self.description,
            'research_questions': self.research_questions.copy(),
            'hypotheses': [h.__dict__ if hasattr(h, '__dict__') else str(h) for h in self.hypotheses],
            'domain': self.domain,
            'priority': self.priority,
            'expected_duration_days': self.expected_duration_days,
            'current_phase': self.current_phase.value,
            'completion_percentage': self.completion_percentage,
            'milestones_completed': self.milestones_completed.copy(),
            'findings': self.findings.copy(),
            'publications': self.publications.copy(),
            'impact_metrics': self.impact_metrics.copy()
        }


class ResearchAgent(ABC):
    """Abstract base class for specialized research agents."""
    
    def __init__(self, agent_name: str, specialization: str):
        self.agent_name = agent_name
        self.specialization = specialization
        self.performance_history = deque(maxlen=1000)
        self.task_history = deque(maxlen=500)
        self.is_active = True
        
    @abstractmethod
    def execute_task(self, task: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Execute a research task."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this agent."""
        if not self.performance_history:
            return {'performance': 0.0, 'efficiency': 0.0, 'success_rate': 0.0}
        
        return {
            'average_performance': np.mean(self.performance_history),
            'performance_std': np.std(self.performance_history),
            'recent_performance': np.mean(list(self.performance_history)[-20:]),
            'success_rate': sum(1 for p in self.performance_history if p > 0.6) / len(self.performance_history),
            'task_count': len(self.task_history)
        }


class HypothesisGenerationAgent(ResearchAgent):
    """Agent specialized in generating research hypotheses."""
    
    def __init__(self):
        super().__init__("HypothesisGenerator", "hypothesis_generation")
        
        # Hypothesis generation network
        self.hypothesis_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Knowledge base for domain expertise
        self.knowledge_base = {
            'materials_science': {
                'properties': ['strength', 'ductility', 'conductivity', 'corrosion_resistance'],
                'processes': ['additive_manufacturing', 'heat_treatment', 'alloying', 'surface_treatment'],
                'relationships': ['structure_property', 'process_structure', 'composition_property']
            }
        }
        
        # Hypothesis quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.generated_hypotheses = []
        
    def execute_task(self, task: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Generate research hypotheses."""
        
        research_context = task.get('context', {})
        num_hypotheses = task.get('num_hypotheses', 5)
        domain = project.domain
        
        generated_hypotheses = []
        
        for i in range(num_hypotheses):
            # Generate hypothesis
            hypothesis = self._generate_hypothesis(research_context, domain, i)
            
            # Score hypothesis quality
            quality_score = self._score_hypothesis_quality(hypothesis, research_context)
            
            hypothesis_obj = ResearchHypothesis(
                hypothesis_id=f"hyp_{project.project_id}_{i}_{int(time.time())}",
                description=hypothesis['description'],
                target_properties=hypothesis['target_properties'],
                predicted_materials=hypothesis['predicted_materials'],
                confidence_level=float(quality_score)
            )
            
            generated_hypotheses.append(hypothesis_obj)
        
        # Sort by confidence/quality
        generated_hypotheses.sort(key=lambda h: h.confidence_level, reverse=True)
        
        # Record performance
        avg_quality = np.mean([h.confidence_level for h in generated_hypotheses])
        self.performance_history.append(avg_quality)
        
        # Update project hypotheses
        project.hypotheses.extend(generated_hypotheses)
        
        self.generated_hypotheses.extend(generated_hypotheses)
        
        result = {
            'agent': self.agent_name,
            'hypotheses_generated': len(generated_hypotheses),
            'average_quality': avg_quality,
            'best_hypothesis': generated_hypotheses[0].__dict__ if generated_hypotheses else None,
            'hypotheses': [h.__dict__ for h in generated_hypotheses]
        }
        
        self.task_history.append({
            'timestamp': time.time(),
            'task_type': 'hypothesis_generation',
            'result': result
        })
        
        logger.info(f"Generated {len(generated_hypotheses)} hypotheses with avg quality {avg_quality:.4f}")
        
        return result
    
    def _generate_hypothesis(self, context: Dict[str, Any], domain: str, seed: int) -> Dict[str, Any]:
        """Generate a single hypothesis."""
        
        # Create context embedding
        context_features = self._encode_context(context, domain)
        
        # Add seed for diversity
        np.random.seed(seed + int(time.time()) % 1000)
        noise = torch.randn(256) * 0.1
        context_tensor = context_features + noise
        
        # Generate hypothesis features
        with torch.no_grad():
            hypothesis_features = self.hypothesis_network(context_tensor)
        
        # Convert to interpretable hypothesis
        hypothesis = self._decode_hypothesis(hypothesis_features, domain)
        
        return hypothesis
    
    def _encode_context(self, context: Dict[str, Any], domain: str) -> torch.Tensor:
        """Encode research context into feature vector."""
        
        # Base features
        features = []
        
        # Domain-specific features
        if domain in self.knowledge_base:
            domain_kb = self.knowledge_base[domain]
            
            # Property importance scores
            for prop in domain_kb['properties']:
                importance = context.get(f"{prop}_importance", 0.5)
                features.append(importance)
            
            # Process relevance scores
            for process in domain_kb['processes']:
                relevance = context.get(f"{process}_relevance", 0.5)
                features.append(relevance)
        
        # Context metadata
        features.extend([
            context.get('novelty_requirement', 0.5),
            context.get('feasibility_requirement', 0.5),
            context.get('impact_potential', 0.5),
            context.get('time_constraint', 0.5)
        ])
        
        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32)
    
    def _decode_hypothesis(self, features: torch.Tensor, domain: str) -> Dict[str, Any]:
        """Decode hypothesis features into interpretable form."""
        
        features_np = features.numpy()
        
        # Extract key components
        property_focus = int(np.argmax(features_np[:4]))  # Focus on which property
        process_focus = int(np.argmax(features_np[4:8]))  # Focus on which process
        relationship_type = int(np.argmax(features_np[8:12]))  # Type of relationship
        
        # Get domain knowledge
        domain_kb = self.knowledge_base.get(domain, self.knowledge_base['materials_science'])
        
        target_property = domain_kb['properties'][property_focus % len(domain_kb['properties'])]
        key_process = domain_kb['processes'][process_focus % len(domain_kb['processes'])]
        relationship = domain_kb['relationships'][relationship_type % len(domain_kb['relationships'])]
        
        # Generate hypothesis description
        hypothesis_templates = [
            f"Optimizing {key_process} parameters will significantly improve {target_property} through enhanced {relationship}",
            f"Novel {target_property} enhancement can be achieved by controlling {relationship} during {key_process}",
            f"The relationship between {key_process} and {target_property} is mediated by {relationship} mechanisms",
            f"Advanced {key_process} strategies will unlock superior {target_property} via {relationship} optimization"
        ]
        
        description = np.random.choice(hypothesis_templates)
        
        # Generate target properties (simplified)
        target_properties = {
            target_property: float(0.5 + np.random.beta(2, 2) * 0.4)  # Target improvement
        }
        
        # Generate predicted materials (placeholder)
        predicted_materials = [torch.randn(64) for _ in range(3)]  # 3 candidate materials
        
        return {
            'description': description,
            'target_properties': target_properties,
            'predicted_materials': predicted_materials,
            'key_process': key_process,
            'target_property': target_property,
            'relationship_focus': relationship
        }
    
    def _score_hypothesis_quality(self, hypothesis: Dict[str, Any], context: Dict[str, Any]) -> torch.Tensor:
        """Score the quality of a generated hypothesis."""
        
        # Create quality assessment features
        quality_features = []
        
        # Novelty assessment
        novelty_score = context.get('novelty_requirement', 0.5)
        quality_features.append(novelty_score)
        
        # Feasibility assessment
        feasibility_score = context.get('feasibility_requirement', 0.5)
        quality_features.append(feasibility_score)
        
        # Impact potential
        impact_score = context.get('impact_potential', 0.5)
        quality_features.append(impact_score)
        
        # Testability
        testability_score = 0.7 + np.random.normal(0, 0.1)  # Most hypotheses reasonably testable
        quality_features.append(max(0.0, min(1.0, testability_score)))
        
        # Pad to required dimensions
        while len(quality_features) < 256:
            quality_features.append(0.5)
        
        quality_tensor = torch.tensor(quality_features[:256], dtype=torch.float32)
        
        with torch.no_grad():
            quality_score = self.quality_scorer(quality_tensor)
        
        return quality_score


class ExperimentalDesignAgent(ResearchAgent):
    """Agent specialized in designing experiments."""
    
    def __init__(self):
        super().__init__("ExperimentalDesigner", "experimental_design")
        
        # Experiment design network
        self.design_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Statistical analysis components
        self.sample_size_calculator = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.designed_experiments = []
        
    def execute_task(self, task: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Design experiments to test hypotheses."""
        
        hypotheses = task.get('hypotheses', project.hypotheses[-5:])  # Last 5 hypotheses
        constraints = task.get('constraints', {})
        
        designed_experiments = []
        
        for hypothesis in hypotheses:
            experiment = self._design_experiment(hypothesis, constraints, project)
            designed_experiments.append(experiment)
        
        # Optimize experimental design
        optimized_experiments = self._optimize_experimental_design(designed_experiments, constraints)
        
        # Calculate performance
        design_quality = np.mean([exp['quality_score'] for exp in optimized_experiments])
        self.performance_history.append(design_quality)
        
        result = {
            'agent': self.agent_name,
            'experiments_designed': len(optimized_experiments),
            'design_quality': design_quality,
            'experiments': optimized_experiments,
            'total_estimated_cost': sum(exp.get('estimated_cost', 100) for exp in optimized_experiments),
            'total_estimated_time_days': sum(exp.get('estimated_time_days', 7) for exp in optimized_experiments)
        }
        
        self.task_history.append({
            'timestamp': time.time(),
            'task_type': 'experimental_design',
            'result': result
        })
        
        self.designed_experiments.extend(optimized_experiments)
        
        logger.info(f"Designed {len(optimized_experiments)} experiments with quality {design_quality:.4f}")
        
        return result
    
    def _design_experiment(self, hypothesis: ResearchHypothesis, constraints: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Design a single experiment."""
        
        # Extract experimental parameters from hypothesis
        target_properties = hypothesis.target_properties
        predicted_materials = hypothesis.predicted_materials
        
        # Design experiment structure
        experiment = {
            'experiment_id': f"exp_{project.project_id}_{len(self.designed_experiments)}_{int(time.time())}",
            'hypothesis_id': hypothesis.hypothesis_id,
            'experiment_type': 'comparative_analysis',  # Default type
            'description': f"Experimental validation of hypothesis: {hypothesis.description[:100]}...",
            
            # Experimental parameters
            'sample_size': self._calculate_sample_size(target_properties, constraints),
            'control_groups': 1,
            'treatment_groups': min(len(predicted_materials), 3),  # Max 3 treatment groups
            'measurements': list(target_properties.keys()),
            'duration_days': 14,  # Default 2 weeks
            
            # Resource requirements
            'estimated_cost': 500 + len(predicted_materials) * 200,  # Base cost + material costs
            'estimated_time_days': 7 + len(predicted_materials) * 2,
            'equipment_required': ['tensile_tester', 'microscope', 'xrd'],
            
            # Quality metrics
            'power_analysis': 0.8,  # Statistical power
            'expected_effect_size': 0.5,
            'quality_score': 0.0  # To be calculated
        }
        
        # Calculate quality score
        experiment['quality_score'] = self._calculate_experiment_quality(experiment, hypothesis, constraints)
        
        return experiment
    
    def _calculate_sample_size(self, target_properties: Dict[str, float], constraints: Dict[str, Any]) -> int:
        """Calculate optimal sample size for experiment."""
        
        # Create sample size calculation features
        features = [
            len(target_properties),  # Number of properties to measure
            np.mean(list(target_properties.values())),  # Average target improvement
            constraints.get('budget_constraint', 0.5),  # Budget limitation
            constraints.get('time_constraint', 0.5)  # Time limitation
        ]
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0.5)
        
        features_tensor = torch.tensor(features[:64], dtype=torch.float32)
        
        with torch.no_grad():
            sample_size_ratio = self.sample_size_calculator(features_tensor)
        
        # Convert to actual sample size (10-100 range)
        base_sample_size = int(10 + sample_size_ratio.item() * 90)
        
        # Adjust based on constraints
        if constraints.get('budget_constraint', 1.0) < 0.5:
            base_sample_size = max(10, base_sample_size // 2)
        
        return base_sample_size
    
    def _optimize_experimental_design(self, experiments: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize experimental design for efficiency and quality."""
        
        # Sort by quality score
        experiments.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Apply resource constraints
        total_budget = constraints.get('max_budget', 5000)
        total_time = constraints.get('max_time_days', 60)
        
        optimized = []
        current_budget = 0
        current_time = 0
        
        for exp in experiments:
            if (current_budget + exp['estimated_cost'] <= total_budget and 
                current_time + exp['estimated_time_days'] <= total_time):
                
                optimized.append(exp)
                current_budget += exp['estimated_cost']
                current_time += exp['estimated_time_days']
        
        return optimized
    
    def _calculate_experiment_quality(self, experiment: Dict[str, Any], hypothesis: ResearchHypothesis, constraints: Dict[str, Any]) -> float:
        """Calculate quality score for experimental design."""
        
        quality_factors = [
            # Statistical rigor
            min(1.0, experiment['sample_size'] / 30),  # Adequate sample size
            experiment['power_analysis'],  # Statistical power
            
            # Feasibility
            1.0 - min(1.0, experiment['estimated_cost'] / constraints.get('max_budget', 10000)),  # Cost efficiency
            1.0 - min(1.0, experiment['estimated_time_days'] / constraints.get('max_time_days', 90)),  # Time efficiency
            
            # Relevance
            hypothesis.confidence_level,  # Hypothesis quality
            len(experiment['measurements']) / 5.0,  # Comprehensiveness
        ]
        
        return np.mean(quality_factors)


class DataAnalysisAgent(ResearchAgent):
    """Agent specialized in analyzing experimental data."""
    
    def __init__(self):
        super().__init__("DataAnalyst", "data_analysis")
        
        # Analysis networks
        self.statistical_analyzer = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 10)  # 10 pattern types
        )
        
        self.analyzed_datasets = []
        
    def execute_task(self, task: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Analyze experimental data."""
        
        datasets = task.get('datasets', [])
        experiments = task.get('experiments', [])
        
        analysis_results = []
        
        for i, dataset in enumerate(datasets):
            experiment = experiments[i] if i < len(experiments) else {}
            
            analysis = self._analyze_dataset(dataset, experiment, project)
            analysis_results.append(analysis)
        
        # Synthesize results across datasets
        synthesis = self._synthesize_results(analysis_results, project)
        
        # Calculate performance
        analysis_quality = np.mean([result['analysis_quality'] for result in analysis_results])
        self.performance_history.append(analysis_quality)
        
        result = {
            'agent': self.agent_name,
            'datasets_analyzed': len(datasets),
            'analysis_quality': analysis_quality,
            'individual_analyses': analysis_results,
            'synthesis': synthesis,
            'significant_findings': [r for r in analysis_results if r.get('significance', 0) > 0.05],
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        self.task_history.append({
            'timestamp': time.time(),
            'task_type': 'data_analysis',
            'result': result
        })
        
        # Update project findings
        for analysis in analysis_results:
            if analysis.get('significance', 0) > 0.05:
                project.findings.append({
                    'type': 'experimental_result',
                    'description': analysis.get('summary', ''),
                    'significance': analysis.get('significance'),
                    'timestamp': time.time()
                })
        
        logger.info(f"Analyzed {len(datasets)} datasets with quality {analysis_quality:.4f}")
        
        return result
    
    def _analyze_dataset(self, dataset: Dict[str, Any], experiment: Dict[str, Any], project: ResearchProject) -> Dict[str, Any]:
        """Analyze a single dataset."""
        
        # Simulate dataset (in real implementation, would load actual data)
        data = dataset.get('data', np.random.normal(0, 1, (100, 10)))  # 100 samples, 10 features
        
        # Convert to tensor for analysis
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Statistical analysis
        stats_features = self._extract_statistical_features(data_tensor)
        
        with torch.no_grad():
            statistical_analysis = self.statistical_analyzer(stats_features)
            pattern_analysis = self.pattern_detector(stats_features)
        
        # Interpret results
        analysis = {
            'dataset_id': dataset.get('id', f"dataset_{int(time.time())}"),
            'experiment_id': experiment.get('experiment_id', 'unknown'),
            'sample_size': data.shape[0],
            'feature_count': data.shape[1],
            
            # Statistical measures
            'mean_values': np.mean(data, axis=0).tolist(),
            'std_values': np.std(data, axis=0).tolist(),
            'correlation_strength': float(np.mean(np.abs(np.corrcoef(data.T)))),
            
            # Significance testing (simplified)
            'significance': self._calculate_significance(data, experiment),
            'effect_size': self._calculate_effect_size(data),
            'confidence_interval': self._calculate_confidence_interval(data),
            
            # Pattern analysis
            'dominant_pattern': int(torch.argmax(pattern_analysis)),
            'pattern_strength': float(torch.max(pattern_analysis)),
            
            # Quality metrics
            'analysis_quality': self._calculate_analysis_quality(data, statistical_analysis),
            'data_quality': self._assess_data_quality(data),
            
            # Summary
            'summary': self._generate_analysis_summary(data, experiment)
        }
        
        return analysis
    
    def _extract_statistical_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from data."""
        
        features = []
        
        # Basic statistics
        features.extend([
            float(torch.mean(data)),
            float(torch.std(data)),
            float(torch.min(data)),
            float(torch.max(data))
        ])
        
        # Distribution properties
        data_flat = data.flatten()
        features.extend([
            float(torch.median(data_flat)),
            float(torch.quantile(data_flat, 0.25)),
            float(torch.quantile(data_flat, 0.75))
        ])
        
        # Correlation features
        if data.shape[1] > 1:
            corr_matrix = torch.corrcoef(data.T)
            features.extend([
                float(torch.mean(torch.abs(corr_matrix))),
                float(torch.max(torch.abs(corr_matrix))),
                float(torch.std(corr_matrix))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to 100 features
        while len(features) < 100:
            features.append(0.0)
        
        return torch.tensor(features[:100], dtype=torch.float32)
    
    def _calculate_significance(self, data: np.ndarray, experiment: Dict[str, Any]) -> float:
        """Calculate statistical significance (simplified p-value)."""
        
        # Simulate p-value calculation
        if data.shape[1] >= 2:
            # Compare first two columns as treatment vs control
            t_stat = np.abs(np.mean(data[:, 0]) - np.mean(data[:, 1])) / (np.std(data) / np.sqrt(data.shape[0]))
            p_value = max(0.001, 1.0 / (1.0 + t_stat))  # Simplified p-value
        else:
            # One-sample test against zero
            t_stat = np.abs(np.mean(data)) / (np.std(data) / np.sqrt(data.shape[0]))
            p_value = max(0.001, 1.0 / (1.0 + t_stat))
        
        return min(1.0, p_value)
    
    def _calculate_effect_size(self, data: np.ndarray) -> float:
        """Calculate effect size (Cohen's d approximation)."""
        
        if data.shape[1] >= 2:
            mean_diff = np.abs(np.mean(data[:, 0]) - np.mean(data[:, 1]))
            pooled_std = np.sqrt((np.var(data[:, 0]) + np.var(data[:, 1])) / 2)
            effect_size = mean_diff / (pooled_std + 1e-8)
        else:
            effect_size = np.abs(np.mean(data)) / (np.std(data) + 1e-8)
        
        return min(3.0, effect_size)  # Cap at 3 for very large effects
    
    def _calculate_confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate 95% confidence interval for the mean."""
        
        mean_val = np.mean(data)
        std_err = np.std(data) / np.sqrt(data.shape[0])
        
        # 95% CI (approximately ±1.96 * SE)
        ci_lower = mean_val - 1.96 * std_err
        ci_upper = mean_val + 1.96 * std_err
        
        return (float(ci_lower), float(ci_upper))
    
    def _calculate_analysis_quality(self, data: np.ndarray, analysis_output: torch.Tensor) -> float:
        """Calculate quality of the analysis."""
        
        quality_factors = [
            min(1.0, data.shape[0] / 30),  # Sample size adequacy
            1.0 - min(1.0, np.sum(np.isnan(data)) / data.size),  # Data completeness
            min(1.0, np.std(data) * 2),  # Data variability (good for analysis)
            min(1.0, float(torch.std(analysis_output)) * 2)  # Analysis richness
        ]
        
        return np.mean(quality_factors)
    
    def _assess_data_quality(self, data: np.ndarray) -> float:
        """Assess quality of the input data."""
        
        quality_factors = [
            1.0 - min(1.0, np.sum(np.isnan(data)) / data.size),  # Completeness
            1.0 - min(1.0, np.sum(np.isinf(data)) / data.size),  # No infinities
            min(1.0, np.std(data)),  # Reasonable variability
            min(1.0, data.shape[0] / 10)  # Adequate sample size
        ]
        
        return np.mean(quality_factors)
    
    def _generate_analysis_summary(self, data: np.ndarray, experiment: Dict[str, Any]) -> str:
        """Generate human-readable summary of analysis."""
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        sample_size = data.shape[0]
        
        if data.shape[1] >= 2:
            comparison = "treatment vs control" 
            effect_direction = "higher" if np.mean(data[:, 0]) > np.mean(data[:, 1]) else "lower"
            summary = f"Analysis of {sample_size} samples shows {comparison} with treatment group {effect_direction} (mean={mean_val:.3f}, std={std_val:.3f})"
        else:
            summary = f"Analysis of {sample_size} samples shows mean={mean_val:.3f} with std={std_val:.3f}"
        
        return summary
    
    def _synthesize_results(self, analyses: List[Dict[str, Any]], project: ResearchProject) -> Dict[str, Any]:
        """Synthesize results across multiple analyses."""
        
        if not analyses:
            return {'status': 'no_data'}
        
        # Aggregate metrics
        significant_results = [a for a in analyses if a.get('significance', 1.0) < 0.05]
        effect_sizes = [a.get('effect_size', 0) for a in analyses]
        
        synthesis = {
            'total_analyses': len(analyses),
            'significant_results': len(significant_results),
            'significance_rate': len(significant_results) / len(analyses),
            'average_effect_size': np.mean(effect_sizes) if effect_sizes else 0.0,
            'effect_size_range': [min(effect_sizes), max(effect_sizes)] if effect_sizes else [0, 0],
            'overall_pattern': self._identify_overall_pattern(analyses),
            'confidence_level': self._calculate_overall_confidence(analyses),
            'recommendations': self._generate_synthesis_recommendations(analyses)
        }
        
        return synthesis
    
    def _identify_overall_pattern(self, analyses: List[Dict[str, Any]]) -> str:
        """Identify overall pattern across analyses."""
        
        if not analyses:
            return "insufficient_data"
        
        # Count dominant patterns
        patterns = [a.get('dominant_pattern', 0) for a in analyses]
        pattern_counts = defaultdict(int)
        
        for pattern in patterns:
            pattern_counts[pattern] += 1
        
        # Find most common pattern
        if pattern_counts:
            dominant_pattern = max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
            return f"pattern_{dominant_pattern}"
        
        return "no_clear_pattern"
    
    def _calculate_overall_confidence(self, analyses: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the results."""
        
        if not analyses:
            return 0.0
        
        confidence_factors = [
            np.mean([a.get('analysis_quality', 0) for a in analyses]),  # Analysis quality
            np.mean([a.get('data_quality', 0) for a in analyses]),      # Data quality
            len([a for a in analyses if a.get('significance', 1) < 0.05]) / len(analyses),  # Significance rate
            min(1.0, len(analyses) / 5)  # Number of analyses
        ]
        
        return np.mean(confidence_factors)
    
    def _generate_synthesis_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on synthesis."""
        
        recommendations = []
        
        # Check significance rate
        sig_rate = len([a for a in analyses if a.get('significance', 1) < 0.05]) / len(analyses) if analyses else 0
        
        if sig_rate > 0.7:
            recommendations.append("Strong evidence supports the hypotheses - recommend proceeding to validation")
        elif sig_rate > 0.4:
            recommendations.append("Moderate evidence found - recommend additional experiments for confirmation")
        else:
            recommendations.append("Limited evidence - recommend revising hypotheses or experimental design")
        
        # Check effect sizes
        effect_sizes = [a.get('effect_size', 0) for a in analyses]
        avg_effect = np.mean(effect_sizes) if effect_sizes else 0
        
        if avg_effect > 0.8:
            recommendations.append("Large effect sizes detected - results likely practically significant")
        elif avg_effect > 0.5:
            recommendations.append("Medium effect sizes - results moderately meaningful")
        else:
            recommendations.append("Small effect sizes - statistical significance may not imply practical importance")
        
        return recommendations
    
    def _generate_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations from analyses."""
        
        recommendations = []
        
        for analysis in analyses:
            if analysis.get('significance', 1.0) < 0.01:
                recommendations.append(f"Strong evidence in {analysis.get('dataset_id', 'dataset')} - investigate further")
            elif analysis.get('effect_size', 0) > 1.0:
                recommendations.append(f"Large effect detected in {analysis.get('dataset_id', 'dataset')} - validate with larger sample")
            elif analysis.get('data_quality', 0) < 0.6:
                recommendations.append(f"Data quality issues in {analysis.get('dataset_id', 'dataset')} - recommend data cleaning")
        
        if not recommendations:
            recommendations.append("Consider expanding sample sizes or refining measurement techniques")
        
        return recommendations


class AutonomousResearchDirector:
    """Main autonomous research director coordinating all research activities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Research agents
        self.agents = {
            'hypothesis_generator': HypothesisGenerationAgent(),
            'experimental_designer': ExperimentalDesignAgent(),
            'data_analyst': DataAnalysisAgent()
        }
        
        # Supporting systems
        self.cognitive_architecture = CognitiveArchitecture()
        self.adaptive_learning = AdaptiveLearningSystem()
        self.discovery_engine = AutonomousDiscoveryEngine()
        
        # Research management
        self.active_projects = {}
        self.completed_projects = {}
        self.research_queue = deque()
        
        # Performance tracking
        self.director_performance = deque(maxlen=1000)
        self.research_metrics = defaultdict(list)
        
        # Resource management
        self.resource_allocation = {
            'computational_budget': 1000.0,  # GPU hours
            'experimental_budget': 10000.0,  # USD
            'time_budget_days': 90          # Days
        }
        
        # Publication and benchmarking
        try:
            self.benchmark_suite = BenchmarkSuite()
            self.publication_manager = PublicationManager()
            self.reproducibility_manager = ReproducibilityManager()
        except Exception as e:
            logger.warning(f"Could not initialize some research components: {e}")
            self.benchmark_suite = None
            self.publication_manager = None
            self.reproducibility_manager = None
        
        # Thread pool for parallel research
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        logger.info(f"Autonomous Research Director initialized with {len(self.agents)} specialized agents")
    
    def initiate_research_project(self, project_spec: Dict[str, Any]) -> str:
        """Initiate a new autonomous research project."""
        
        project = ResearchProject(
            project_id=f"proj_{int(time.time())}_{np.random.randint(1000)}",
            title=project_spec.get('title', 'Autonomous Research Project'),
            description=project_spec.get('description', 'AI-directed materials research'),
            research_questions=project_spec.get('research_questions', ['How can we optimize material properties?']),
            hypotheses=[],
            domain=project_spec.get('domain', 'materials_science'),
            priority=project_spec.get('priority', 0.5),
            expected_duration_days=project_spec.get('expected_duration_days', 30)
        )
        
        self.active_projects[project.project_id] = project
        self.research_queue.append(project.project_id)
        
        logger.info(f"Initiated research project: {project.title} (ID: {project.project_id})")
        
        return project.project_id
    
    def execute_autonomous_research_cycle(self, project_id: str, num_cycles: int = 5) -> Dict[str, Any]:
        """Execute autonomous research cycles for a project."""
        
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found in active projects")
        
        project = self.active_projects[project_id]
        
        logger.info(f"Starting autonomous research cycle for project: {project.title}")
        
        cycle_results = {
            'project_id': project_id,
            'cycles_completed': 0,
            'phase_transitions': [],
            'discoveries': [],
            'publications_generated': 0,
            'overall_progress': 0.0
        }
        
        for cycle in range(num_cycles):
            logger.info(f"Executing research cycle {cycle + 1}/{num_cycles}")
            
            # Determine current research phase
            current_phase = project.current_phase
            
            # Execute phase-specific research
            cycle_result = self._execute_research_phase(project, current_phase)
            
            # Update project progress
            self._update_project_progress(project, cycle_result)
            
            # Check for phase transitions
            new_phase = self._determine_next_phase(project, cycle_result)
            
            if new_phase != current_phase:
                project.current_phase = new_phase
                cycle_results['phase_transitions'].append({
                    'cycle': cycle + 1,
                    'from_phase': current_phase.value,
                    'to_phase': new_phase.value,
                    'trigger': cycle_result.get('phase_transition_trigger', 'automatic')
                })
                logger.info(f"Phase transition: {current_phase.value} -> {new_phase.value}")
            
            # Check for discoveries
            if cycle_result.get('significant_discovery', False):
                cycle_results['discoveries'].append({
                    'cycle': cycle + 1,
                    'discovery': cycle_result.get('discovery_description', ''),
                    'significance': cycle_result.get('discovery_significance', 0.0)
                })
            
            cycle_results['cycles_completed'] += 1
            
            # Brief pause between cycles
            time.sleep(1.0)
        
        # Final project assessment
        final_assessment = self._assess_project_completion(project)
        cycle_results['final_assessment'] = final_assessment
        cycle_results['overall_progress'] = project.completion_percentage
        
        # Generate publications if research is mature enough
        if project.completion_percentage > 0.7:
            publication_result = self._generate_research_publications(project)
            cycle_results['publications_generated'] = len(publication_result.get('publications', []))
        
        logger.info(
            f"Autonomous research cycle completed: {cycle_results['cycles_completed']} cycles, "
            f"{len(cycle_results['discoveries'])} discoveries, "
            f"{cycle_results['publications_generated']} publications"
        )
        
        return cycle_results
    
    def _execute_research_phase(self, project: ResearchProject, phase: ResearchPhase) -> Dict[str, Any]:
        """Execute research activities for a specific phase."""
        
        if phase == ResearchPhase.HYPOTHESIS_GENERATION:
            return self._execute_hypothesis_generation(project)
        elif phase == ResearchPhase.EXPERIMENTAL_DESIGN:
            return self._execute_experimental_design(project)
        elif phase == ResearchPhase.DATA_COLLECTION:
            return self._execute_data_collection(project)
        elif phase == ResearchPhase.ANALYSIS:
            return self._execute_data_analysis(project)
        elif phase == ResearchPhase.VALIDATION:
            return self._execute_validation(project)
        elif phase == ResearchPhase.PUBLICATION:
            return self._execute_publication(project)
        else:
            return {'phase': phase.value, 'status': 'not_implemented'}
    
    def _execute_hypothesis_generation(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute hypothesis generation phase."""
        
        task = {
            'context': {
                'novelty_requirement': 0.8,
                'feasibility_requirement': 0.7,
                'impact_potential': 0.9,
                'time_constraint': 0.5
            },
            'num_hypotheses': 8
        }
        
        result = self.agents['hypothesis_generator'].execute_task(task, project)
        
        # Assess if ready for next phase
        if len(project.hypotheses) >= 5 and result.get('average_quality', 0) > 0.6:
            result['phase_transition_trigger'] = 'sufficient_quality_hypotheses'
            result['ready_for_next_phase'] = True
        
        return result
    
    def _execute_experimental_design(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute experimental design phase."""
        
        task = {
            'hypotheses': project.hypotheses[-5:],  # Use latest 5 hypotheses
            'constraints': {
                'max_budget': self.resource_allocation['experimental_budget'] * 0.3,  # 30% of budget
                'max_time_days': 30,
                'budget_constraint': 0.7,
                'time_constraint': 0.6
            }
        }
        
        result = self.agents['experimental_designer'].execute_task(task, project)
        
        # Check if designs are adequate
        if (result.get('experiments_designed', 0) >= 3 and 
            result.get('design_quality', 0) > 0.7):
            result['phase_transition_trigger'] = 'adequate_experimental_design'
            result['ready_for_next_phase'] = True
        
        return result
    
    def _execute_data_collection(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute data collection phase (simulated)."""
        
        # Simulate experimental data collection
        num_experiments = min(5, len(project.hypotheses))
        
        collected_datasets = []
        
        for i in range(num_experiments):
            # Simulate data collection with realistic characteristics
            sample_size = np.random.randint(50, 200)
            feature_count = np.random.randint(5, 15)
            
            # Generate synthetic experimental data
            base_data = np.random.normal(0, 1, (sample_size, feature_count))
            
            # Add some structure/effects
            if i < len(project.hypotheses):
                hypothesis = project.hypotheses[i]
                # Simulate hypothesis-driven effects
                effect_strength = hypothesis.confidence_level * 0.5
                base_data[:, 0] += np.random.normal(effect_strength, 0.1, sample_size)
            
            dataset = {
                'id': f"dataset_{project.project_id}_{i}",
                'experiment_id': f"exp_{project.project_id}_{i}",
                'data': base_data,
                'collection_date': time.time(),
                'sample_size': sample_size,
                'feature_count': feature_count
            }
            
            collected_datasets.append(dataset)
        
        result = {
            'phase': 'data_collection',
            'datasets_collected': len(collected_datasets),
            'total_samples': sum(d['sample_size'] for d in collected_datasets),
            'datasets': collected_datasets,
            'collection_quality': 0.8 + np.random.normal(0, 0.1),  # Simulate quality
            'phase_transition_trigger': 'data_collection_complete',
            'ready_for_next_phase': True
        }
        
        # Store datasets in project context
        if not hasattr(project, 'datasets'):
            project.datasets = []
        project.datasets.extend(collected_datasets)
        
        return result
    
    def _execute_data_analysis(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute data analysis phase."""
        
        # Get datasets from project
        datasets = getattr(project, 'datasets', [])
        
        if not datasets:
            # Simulate some datasets if none exist
            datasets = [{
                'id': f'simulated_dataset_{i}',
                'data': np.random.normal(0, 1, (100, 8))
            } for i in range(3)]
        
        task = {
            'datasets': datasets,
            'experiments': [{'experiment_id': f'exp_{i}'} for i in range(len(datasets))]
        }
        
        result = self.agents['data_analyst'].execute_task(task, project)
        
        # Check for significant findings
        significant_findings = result.get('significant_findings', [])
        
        if len(significant_findings) > 0:
            result['significant_discovery'] = True
            result['discovery_description'] = f"Significant findings in {len(significant_findings)} datasets"
            result['discovery_significance'] = np.mean([f.get('significance', 0) for f in significant_findings])
        
        # Check readiness for validation
        if (result.get('analysis_quality', 0) > 0.7 and 
            len(significant_findings) > 0):
            result['phase_transition_trigger'] = 'significant_findings_identified'
            result['ready_for_next_phase'] = True
        
        return result
    
    def _execute_validation(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute validation phase."""
        
        # Cross-validation of findings
        validation_results = []
        
        for finding in project.findings:
            # Simulate validation process
            validation_score = 0.6 + np.random.beta(2, 2) * 0.3  # Score between 0.6-0.9
            
            validation = {
                'finding_id': finding.get('description', '')[:50],
                'validation_score': validation_score,
                'validation_method': 'cross_validation',
                'replicated': validation_score > 0.75
            }
            
            validation_results.append(validation)
        
        # Overall validation assessment
        avg_validation = np.mean([v['validation_score'] for v in validation_results]) if validation_results else 0.7
        replication_rate = sum(1 for v in validation_results if v['replicated']) / len(validation_results) if validation_results else 0.5
        
        result = {
            'phase': 'validation',
            'validations_performed': len(validation_results),
            'average_validation_score': avg_validation,
            'replication_rate': replication_rate,
            'validation_results': validation_results,
            'validation_quality': avg_validation
        }
        
        # Check if ready for publication
        if avg_validation > 0.75 and replication_rate > 0.6:
            result['phase_transition_trigger'] = 'validation_successful'
            result['ready_for_next_phase'] = True
        
        return result
    
    def _execute_publication(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute publication phase."""
        
        return self._generate_research_publications(project)
    
    def _generate_research_publications(self, project: ResearchProject) -> Dict[str, Any]:
        """Generate research publications from project results."""
        
        publications = []
        
        # Generate main research paper
        main_paper = {
            'title': f"Autonomous Discovery in {project.domain.replace('_', ' ').title()}: {project.title}",
            'abstract': f"This study presents autonomous AI-driven research results in {project.domain}, "
                       f"involving {len(project.hypotheses)} hypotheses and {len(project.findings)} significant findings.",
            'findings': project.findings.copy(),
            'hypotheses': [h.__dict__ if hasattr(h, '__dict__') else str(h) for h in project.hypotheses],
            'methodology': 'autonomous_ai_research',
            'significance_score': np.mean([f.get('significance', 0) for f in project.findings]) if project.findings else 0.5,
            'novelty_score': 0.8,  # High novelty for AI-generated research
            'publication_type': 'journal_article'
        }
        
        publications.append(main_paper)
        
        # Generate methods paper if methodology is novel
        if len(project.hypotheses) > 5:
            methods_paper = {
                'title': f"Autonomous Hypothesis Generation and Testing in {project.domain.replace('_', ' ').title()}",
                'abstract': "This paper describes novel autonomous AI methods for hypothesis generation and experimental validation.",
                'focus': 'methodology',
                'novelty_score': 0.9,
                'publication_type': 'methods_paper'
            }
            publications.append(methods_paper)
        
        # Update project
        project.publications = [p['title'] for p in publications]
        
        result = {
            'phase': 'publication',
            'publications': publications,
            'publications_generated': len(publications),
            'total_impact_score': sum(p.get('significance_score', 0) for p in publications)
        }
        
        logger.info(f"Generated {len(publications)} publications for project {project.project_id}")
        
        return result
    
    def _update_project_progress(self, project: ResearchProject, cycle_result: Dict[str, Any]):
        """Update project progress based on cycle results."""
        
        # Calculate progress increment based on phase and results
        phase_weights = {
            ResearchPhase.HYPOTHESIS_GENERATION: 0.15,
            ResearchPhase.EXPERIMENTAL_DESIGN: 0.15,
            ResearchPhase.DATA_COLLECTION: 0.20,
            ResearchPhase.ANALYSIS: 0.25,
            ResearchPhase.VALIDATION: 0.15,
            ResearchPhase.PUBLICATION: 0.10
        }
        
        phase_weight = phase_weights.get(project.current_phase, 0.1)
        
        # Quality factor
        quality_metrics = [
            cycle_result.get('average_quality', 0.5),
            cycle_result.get('design_quality', 0.5),
            cycle_result.get('analysis_quality', 0.5),
            cycle_result.get('validation_quality', 0.5)
        ]
        
        quality_factor = np.mean([m for m in quality_metrics if m > 0]) if any(m > 0 for m in quality_metrics) else 0.5
        
        # Calculate progress increment
        progress_increment = phase_weight * quality_factor
        
        # Update project completion
        project.completion_percentage = min(1.0, project.completion_percentage + progress_increment)
        
        # Update milestones
        if cycle_result.get('ready_for_next_phase', False):
            milestone = f"{project.current_phase.value}_completed"
            if milestone not in project.milestones_completed:
                project.milestones_completed.append(milestone)
    
    def _determine_next_phase(self, project: ResearchProject, cycle_result: Dict[str, Any]) -> ResearchPhase:
        """Determine the next research phase."""
        
        current_phase = project.current_phase
        
        # Check if ready for next phase
        if not cycle_result.get('ready_for_next_phase', False):
            return current_phase  # Stay in current phase
        
        # Phase progression logic
        phase_progression = {
            ResearchPhase.HYPOTHESIS_GENERATION: ResearchPhase.EXPERIMENTAL_DESIGN,
            ResearchPhase.EXPERIMENTAL_DESIGN: ResearchPhase.DATA_COLLECTION,
            ResearchPhase.DATA_COLLECTION: ResearchPhase.ANALYSIS,
            ResearchPhase.ANALYSIS: ResearchPhase.VALIDATION,
            ResearchPhase.VALIDATION: ResearchPhase.PUBLICATION,
            ResearchPhase.PUBLICATION: ResearchPhase.FOLLOW_UP
        }
        
        return phase_progression.get(current_phase, current_phase)
    
    def _assess_project_completion(self, project: ResearchProject) -> Dict[str, Any]:
        """Assess overall project completion and quality."""
        
        assessment = {
            'completion_percentage': project.completion_percentage,
            'milestones_completed': len(project.milestones_completed),
            'hypotheses_generated': len(project.hypotheses),
            'significant_findings': len(project.findings),
            'publications_produced': len(project.publications),
            'overall_quality': 0.0,
            'research_impact': 0.0,
            'novelty_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall quality
        quality_factors = [
            min(1.0, len(project.hypotheses) / 5),  # Hypothesis generation quality
            min(1.0, len(project.findings) / 3),    # Findings adequacy
            project.completion_percentage,           # Completion rate
        ]
        
        assessment['overall_quality'] = np.mean(quality_factors)
        
        # Calculate research impact
        if project.findings:
            avg_significance = np.mean([f.get('significance', 0) for f in project.findings])
            assessment['research_impact'] = avg_significance
        
        # Novelty score (high for autonomous AI research)
        assessment['novelty_score'] = 0.8 + np.random.normal(0, 0.1)
        assessment['novelty_score'] = max(0.0, min(1.0, assessment['novelty_score']))
        
        # Generate recommendations
        if assessment['overall_quality'] > 0.8:
            assessment['recommendations'].append("Excellent research quality - recommend publication in top-tier venue")
        elif assessment['overall_quality'] > 0.6:
            assessment['recommendations'].append("Good research quality - recommend additional validation before publication")
        else:
            assessment['recommendations'].append("Research quality needs improvement - recommend additional experiments")
        
        if len(project.publications) == 0 and project.completion_percentage > 0.7:
            assessment['recommendations'].append("Research is mature enough for publication - initiate publication process")
        
        return assessment
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of research activities."""
        
        summary = {
            'active_projects': len(self.active_projects),
            'completed_projects': len(self.completed_projects),
            'total_projects': len(self.active_projects) + len(self.completed_projects),
            'agent_performance': {},
            'resource_utilization': self._calculate_resource_utilization(),
            'research_metrics': dict(self.research_metrics),
            'top_discoveries': self._get_top_discoveries(),
            'publication_summary': self._get_publication_summary()
        }
        
        # Get agent performance
        for agent_name, agent in self.agents.items():
            summary['agent_performance'][agent_name] = agent.get_performance_metrics()
        
        # Calculate overall director performance
        if self.director_performance:
            summary['director_performance'] = {
                'average_performance': np.mean(self.director_performance),
                'performance_trend': self._calculate_performance_trend(),
                'total_research_cycles': len(self.director_performance)
            }
        
        return summary
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization across all projects."""
        
        utilization = {
            'computational_utilization': 0.0,
            'experimental_utilization': 0.0,
            'time_utilization': 0.0
        }
        
        for project in self.active_projects.values():
            # Estimate resource usage based on project progress
            progress = project.completion_percentage
            
            utilization['computational_utilization'] += progress * 100  # GPU hours
            utilization['experimental_utilization'] += progress * 1000   # USD
            utilization['time_utilization'] += progress * 10             # Days
        
        # Calculate as percentages of total budget
        utilization['computational_utilization'] /= self.resource_allocation['computational_budget']
        utilization['experimental_utilization'] /= self.resource_allocation['experimental_budget']
        utilization['time_utilization'] /= self.resource_allocation['time_budget_days']
        
        return utilization
    
    def _get_top_discoveries(self) -> List[Dict[str, Any]]:
        """Get top discoveries across all projects."""
        
        discoveries = []
        
        for project in list(self.active_projects.values()) + list(self.completed_projects.values()):
            for finding in project.findings:
                if finding.get('significance', 0) > 0.7:  # High significance threshold
                    discovery = {
                        'project_id': project.project_id,
                        'project_title': project.title,
                        'description': finding.get('description', ''),
                        'significance': finding.get('significance', 0),
                        'type': finding.get('type', 'unknown')
                    }
                    discoveries.append(discovery)
        
        # Sort by significance
        discoveries.sort(key=lambda x: x['significance'], reverse=True)
        
        return discoveries[:10]  # Top 10
    
    def _get_publication_summary(self) -> Dict[str, Any]:
        """Get summary of publications across all projects."""
        
        total_publications = 0
        publication_types = defaultdict(int)
        
        for project in list(self.active_projects.values()) + list(self.completed_projects.values()):
            total_publications += len(project.publications)
            # Note: publication types would need to be tracked in project data
        
        return {
            'total_publications': total_publications,
            'publication_types': dict(publication_types),
            'average_publications_per_project': total_publications / max(1, len(self.active_projects) + len(self.completed_projects))
        }
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend for the director."""
        
        if len(self.director_performance) < 10:
            return 0.0
        
        recent = list(self.director_performance)[-10:]
        earlier = list(self.director_performance)[-20:-10] if len(self.director_performance) >= 20 else list(self.director_performance)[:-10]
        
        if not earlier:
            return 0.0
        
        return np.mean(recent) - np.mean(earlier)
    
    def shutdown(self):
        """Shutdown autonomous research director gracefully."""
        
        logger.info("Shutting down Autonomous Research Director")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Deactivate all agents
        for agent in self.agents.values():
            agent.is_active = False
        
        # Shutdown supporting systems
        self.cognitive_architecture.shutdown()
        self.adaptive_learning.shutdown()
        
        logger.info("Autonomous Research Director shutdown complete")


def create_autonomous_research_director(config: Optional[Dict[str, Any]] = None) -> AutonomousResearchDirector:
    """Factory function to create autonomous research director."""
    
    return AutonomousResearchDirector(config)
