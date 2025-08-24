"""Comprehensive test suite for Generation 5 Autonomous Excellence capabilities."""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import Generation 5 modules
try:
    from microdiff_matdesign.evolution.meta_learning_optimizer import (
        MetaLearningOptimizer, MetaLearningConfig, MetaLearningStrategy
    )
    from microdiff_matdesign.autonomous.cognitive_architecture import (
        CognitiveArchitecture, CognitiveState, ProcessingMode
    )
    from microdiff_matdesign.autonomous.adaptive_learning_system import (
        AdaptiveLearningSystem, LearningContext, AdaptationStrategy
    )
    from microdiff_matdesign.research.autonomous_research_director import (
        AutonomousResearchDirector, ResearchProject, ResearchPhase
    )
except ImportError as e:
    pytest.skip(f"Generation 5 modules not available: {e}", allow_module_level=True)


class TestMetaLearningOptimizer:
    """Test suite for Meta-Learning Optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        config = MetaLearningConfig(
            meta_learning_rate=1e-3,
            inner_learning_rate=1e-2,
            meta_batch_size=16,
            inner_update_steps=3
        )
        return MetaLearningOptimizer(config)
    
    def test_meta_optimizer_initialization(self, optimizer):
        """Test meta-learning optimizer initialization."""
        assert optimizer is not None
        assert optimizer.meta_network is not None
        assert optimizer.meta_optimizer is not None
        assert len(optimizer.performance_history) == 0
        assert len(optimizer.task_performance) == 0
    
    def test_task_context_extraction(self, optimizer):
        """Test task context extraction."""
        task = {
            'complexity': 0.8,
            'novelty': 0.6,
            'resource_requirement': 0.4,
            'urgency': 0.9,
            'dependencies': ['dep1', 'dep2'],
            'success_probability': 0.7,
            'type': 'optimization'
        }
        
        context = optimizer.extract_task_context(task)
        
        assert isinstance(context, torch.Tensor)
        assert context.shape == (64,)
        assert 0.0 <= context[0].item() <= 1.0  # complexity
        assert 0.0 <= context[1].item() <= 1.0  # novelty
    
    def test_meta_learning_task_execution(self, optimizer):
        """Test meta-learning task execution."""
        task = {
            'id': 'test_task_1',
            'type': 'optimization',
            'complexity': 0.5,
            'novelty': 0.6,
            'resource_requirement': 0.3
        }
        
        result = optimizer.meta_learn_task(task)
        
        assert 'task_id' in result
        assert 'strategy' in result
        assert 'performance' in result
        assert 'adaptive_lr' in result
        assert 0.0 <= result['performance'] <= 1.0
        assert result['strategy'] in [s.value for s in MetaLearningStrategy]
    
    def test_learning_to_learn_strategy(self, optimizer):
        """Test learning-to-learn meta-strategy."""
        task = {'type': 'learning', 'complexity': 0.6}
        
        result = optimizer._learning_to_learn(task, learning_rate=0.01)
        
        assert result['strategy'] == 'learning_to_learn'
        assert 'performance' in result
        assert 'inner_losses' in result
        assert len(result['inner_losses']) == optimizer.config.inner_update_steps
        assert all(loss >= 0 for loss in result['inner_losses'])
    
    def test_self_modifying_code_strategy(self, optimizer):
        """Test self-modifying code strategy."""
        optimizer.config.allow_code_modification = True
        optimizer.config.safety_checks = True
        
        task = {'type': 'optimization', 'complexity': 0.7}
        
        result = optimizer._self_modifying_code(task)
        
        assert result['strategy'] == 'self_modifying_code'
        assert 'modifications' in result
        assert 'safety_checks_passed' in result
        assert result['safety_checks_passed'] is True
    
    def test_evolutionary_algorithms_strategy(self, optimizer):
        """Test evolutionary algorithms strategy."""
        task = {'type': 'evolution', 'complexity': 0.8}
        
        result = optimizer._evolutionary_algorithms(task)
        
        assert result['strategy'] == 'evolutionary_algorithms'
        assert 'generations' in result
        assert 'best_fitness' in result
        assert 'fitness_history' in result
        assert len(result['fitness_history']) == result['generations']
    
    def test_neural_architecture_search(self, optimizer):
        """Test neural architecture search strategy."""
        task = {'type': 'architecture', 'complexity': 0.6}
        arch_params = torch.rand(10)
        
        result = optimizer._neural_architecture_search(task, arch_params)
        
        assert result['strategy'] == 'neural_architecture_search'
        assert 'architectures_evaluated' in result
        assert 'best_architecture' in result
        assert result['architectures_evaluated'] > 0
    
    def test_performance_tracking(self, optimizer):
        """Test performance tracking across multiple tasks."""
        tasks = [
            {'id': f'task_{i}', 'type': 'test', 'complexity': 0.5}
            for i in range(5)
        ]
        
        for task in tasks:
            optimizer.meta_learn_task(task)
        
        summary = optimizer.get_performance_summary()
        
        assert summary['total_tasks'] == 5
        assert 'average_performance' in summary
        assert 'recent_performance' in summary
        assert 'improvement_trend' in summary
    
    def test_autonomous_evolution_cycle(self, optimizer):
        """Test autonomous evolution cycle."""
        # Short cycle for testing
        result = optimizer.autonomous_evolution_cycle(duration_hours=0.01)  # 36 seconds
        
        assert 'start_time' in result
        assert 'duration_hours' in result
        assert 'tasks_completed' in result
        assert 'improvements_discovered' in result
        assert result['tasks_completed'] > 0


class TestCognitiveArchitecture:
    """Test suite for Cognitive Architecture."""
    
    @pytest.fixture
    def cognitive_arch(self):
        return CognitiveArchitecture()
    
    def test_cognitive_architecture_initialization(self, cognitive_arch):
        """Test cognitive architecture initialization."""
        assert cognitive_arch is not None
        assert cognitive_arch.cognitive_state is not None
        assert len(cognitive_arch.processors) > 0
        assert cognitive_arch.global_controller is not None
    
    def test_cognitive_state_management(self, cognitive_arch):
        """Test cognitive state management."""
        state = cognitive_arch.cognitive_state
        
        assert isinstance(state, CognitiveState)
        assert 0.0 <= state.cognitive_load <= 1.0
        assert 0.0 <= state.arousal_level <= 1.0
        assert 0.0 <= state.confidence_level <= 1.0
        
        # Test state conversion
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert 'cognitive_load' in state_dict
        assert 'confidence_level' in state_dict
    
    def test_input_processing_pipeline(self, cognitive_arch):
        """Test input processing through cognitive pipeline."""
        # Create test input
        test_input = torch.randn(512)
        
        result = cognitive_arch.process_input(test_input)
        
        assert 'results' in result
        assert 'cognitive_state' in result
        assert 'processing_time' in result
        assert result['processing_time'] > 0
        
        # Check that processors were called
        assert 'perception' in result['results']
        assert 'memory' in result['results']
    
    def test_processing_modes(self, cognitive_arch):
        """Test different processing modes."""
        test_input = torch.randn(512)
        
        # Test automatic mode
        result_auto = cognitive_arch.process_input(test_input, ProcessingMode.AUTOMATIC)
        assert result_auto['cognitive_state']['processing_mode'] == 'automatic'
        
        # Test controlled mode
        result_controlled = cognitive_arch.process_input(test_input, ProcessingMode.CONTROLLED)
        assert result_controlled['cognitive_state']['processing_mode'] == 'controlled'
    
    def test_autonomous_thinking_loop(self, cognitive_arch):
        """Test autonomous thinking loop."""
        # Short thinking session for testing
        result = cognitive_arch.autonomous_thinking_loop(duration_minutes=0.05)  # 3 seconds
        
        assert 'thoughts_generated' in result
        assert 'insights_discovered' in result
        assert 'creative_outputs' in result
        assert result['thoughts_generated'] > 0
    
    def test_cognitive_summary(self, cognitive_arch):
        """Test cognitive summary generation."""
        # Process some inputs first
        for _ in range(3):
            test_input = torch.randn(512)
            cognitive_arch.process_input(test_input)
        
        summary = cognitive_arch.get_cognitive_summary()
        
        assert 'cognitive_state' in summary
        assert 'processor_performance' in summary
        assert 'processing_statistics' in summary
    
    def test_memory_operations(self, cognitive_arch):
        """Test memory storage and retrieval."""
        memory_processor = cognitive_arch.processors['memory']
        
        # Store some memories
        test_data = torch.randn(512)
        store_input = {'store': test_data, 'query': None, 'retrieve': False}
        
        result, _ = memory_processor.process(store_input, cognitive_arch.cognitive_state)
        
        assert 'stored_memory_id' in result
        
        # Retrieve memories
        query_input = {'query': test_data, 'store': None, 'retrieve': True}
        
        result, _ = memory_processor.process(query_input, cognitive_arch.cognitive_state)
        
        assert 'retrieved_memories' in result
    
    def test_reasoning_operations(self, cognitive_arch):
        """Test reasoning operations."""
        reasoning_processor = cognitive_arch.processors['reasoning']
        
        # Test logical reasoning
        premises = [torch.randn(512), torch.randn(512)]
        reasoning_input = {'premises': premises, 'type': 'logical'}
        
        result, _ = reasoning_processor.process(reasoning_input, cognitive_arch.cognitive_state)
        
        assert 'conclusion' in result
        assert 'confidence' in result
        assert 'reasoning_type' in result
        assert result['reasoning_type'] == 'logical'


class TestAdaptiveLearningSystem:
    """Test suite for Adaptive Learning System."""
    
    @pytest.fixture
    def learning_system(self):
        return AdaptiveLearningSystem(input_dim=256)
    
    def test_adaptive_learning_initialization(self, learning_system):
        """Test adaptive learning system initialization."""
        assert learning_system is not None
        assert len(learning_system.learning_modules) > 0
        assert learning_system.meta_optimizer is not None
        assert learning_system.cognitive_architecture is not None
    
    def test_learning_context_creation(self, learning_system):
        """Test learning context creation."""
        test_data = torch.randn(256)
        
        context = learning_system._create_learning_context(test_data)
        
        assert isinstance(context, LearningContext)
        assert 0.0 <= context.task_complexity <= 1.0
        assert 0.0 <= context.task_novelty <= 1.0
        assert context.task_type == "materials_design"
    
    def test_adaptation_strategy_selection(self, learning_system):
        """Test adaptation strategy selection."""
        context = LearningContext(
            task_complexity=0.7,
            task_novelty=0.8,
            environment_stability=0.6
        )
        
        strategy = learning_system._select_adaptation_strategy(context)
        
        assert isinstance(strategy, AdaptationStrategy)
        assert strategy in list(AdaptationStrategy)
    
    def test_adaptive_learning_cycle(self, learning_system):
        """Test adaptive learning cycle."""
        test_data = torch.randn(256)
        
        result = learning_system.adaptive_learn(test_data)
        
        assert 'learning_results' in result
        assert 'integrated_performance' in result
        assert 'adaptation_strategy' in result
        assert 'learning_time' in result
        assert 0.0 <= result['integrated_performance'] <= 1.0
    
    def test_self_supervised_learning(self, learning_system):
        """Test self-supervised learning module."""
        ssl_module = learning_system.learning_modules['self_supervised']
        test_data = torch.randn(256)
        context = LearningContext()
        
        result = ssl_module.learn(test_data, context)
        
        assert 'reconstruction_loss' in result
        assert 'contrastive_loss' in result
        assert 'performance' in result
        assert result['performance'] > 0.0
    
    def test_continual_learning(self, learning_system):
        """Test continual learning module."""
        continual_module = learning_system.learning_modules['continual']
        test_data = torch.randn(256)
        context = LearningContext()
        
        result = continual_module.learn(test_data, context)
        
        assert 'classification_loss' in result
        assert 'current_task' in result
        assert 'predicted_task' in result
        assert result['current_task'] >= 0
    
    def test_module_adaptation(self, learning_system):
        """Test learning module adaptation."""
        ssl_module = learning_system.learning_modules['self_supervised']
        
        feedback = {
            'performance': 0.3,  # Low performance to trigger adaptation
            'target_performance': 0.8
        }
        
        context = LearningContext()
        adapted = ssl_module.adapt(feedback, context)
        
        assert isinstance(adapted, bool)
        if adapted:
            assert ssl_module.adaptation_count > 0
    
    def test_continuous_learning_cycle(self, learning_system):
        """Test continuous learning cycle."""
        # Short cycle for testing
        result = learning_system.continuous_learning_cycle(duration_hours=0.01)  # 36 seconds
        
        assert 'learning_iterations' in result
        assert 'adaptations_made' in result
        assert 'average_performance' in result
        assert 'performance_improvement' in result
        assert result['learning_iterations'] > 0
    
    def test_system_summary(self, learning_system):
        """Test system summary generation."""
        # Perform some learning first
        test_data = torch.randn(256)
        learning_system.adaptive_learn(test_data)
        
        summary = learning_system.get_system_summary()
        
        assert 'system_performance' in summary
        assert 'learning_modules' in summary
        assert 'adaptation_statistics' in summary
        assert 'resource_usage' in summary


class TestAutonomousResearchDirector:
    """Test suite for Autonomous Research Director."""
    
    @pytest.fixture
    def research_director(self):
        return AutonomousResearchDirector()
    
    def test_research_director_initialization(self, research_director):
        """Test research director initialization."""
        assert research_director is not None
        assert len(research_director.agents) > 0
        assert 'hypothesis_generator' in research_director.agents
        assert 'experimental_designer' in research_director.agents
        assert 'data_analyst' in research_director.agents
    
    def test_project_initiation(self, research_director):
        """Test research project initiation."""
        project_spec = {
            'title': 'Test Autonomous Research',
            'description': 'Test project for autonomous AI research',
            'research_questions': [
                'How can we optimize material properties autonomously?',
                'What novel materials can AI discover?'
            ],
            'domain': 'materials_science',
            'priority': 0.8
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        
        assert project_id is not None
        assert project_id in research_director.active_projects
        
        project = research_director.active_projects[project_id]
        assert project.title == project_spec['title']
        assert project.domain == project_spec['domain']
        assert len(project.research_questions) == 2
    
    def test_hypothesis_generation_agent(self, research_director):
        """Test hypothesis generation agent."""
        hyp_agent = research_director.agents['hypothesis_generator']
        
        # Create a test project
        project = ResearchProject(
            project_id='test_proj',
            title='Test Project',
            description='Test',
            research_questions=['Test question?'],
            hypotheses=[]
        )
        
        task = {
            'context': {
                'novelty_requirement': 0.8,
                'feasibility_requirement': 0.7,
                'impact_potential': 0.9
            },
            'num_hypotheses': 3
        }
        
        result = hyp_agent.execute_task(task, project)
        
        assert 'hypotheses_generated' in result
        assert 'average_quality' in result
        assert result['hypotheses_generated'] == 3
        assert len(project.hypotheses) == 3
    
    def test_experimental_design_agent(self, research_director):
        """Test experimental design agent."""
        exp_agent = research_director.agents['experimental_designer']
        
        # Create project with hypotheses
        project = ResearchProject(
            project_id='test_proj',
            title='Test Project',
            description='Test',
            research_questions=['Test question?'],
            hypotheses=[]
        )
        
        # Add mock hypotheses
        from microdiff_matdesign.evolution.autonomous_discovery_engine import ResearchHypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id='test_hyp',
            description='Test hypothesis',
            target_properties={'strength': 0.8},
            predicted_materials=[torch.randn(64)],
            confidence_level=0.7
        )
        project.hypotheses.append(hypothesis)
        
        task = {
            'hypotheses': project.hypotheses,
            'constraints': {
                'max_budget': 5000,
                'max_time_days': 30
            }
        }
        
        result = exp_agent.execute_task(task, project)
        
        assert 'experiments_designed' in result
        assert 'design_quality' in result
        assert 'experiments' in result
        assert result['experiments_designed'] > 0
    
    def test_data_analysis_agent(self, research_director):
        """Test data analysis agent."""
        data_agent = research_director.agents['data_analyst']
        
        # Create test datasets
        datasets = [
            {
                'id': 'test_dataset_1',
                'data': np.random.normal(0, 1, (50, 8))
            },
            {
                'id': 'test_dataset_2', 
                'data': np.random.normal(0.5, 1, (60, 8))  # Slight shift for effect
            }
        ]
        
        experiments = [
            {'experiment_id': 'exp_1'},
            {'experiment_id': 'exp_2'}
        ]
        
        project = ResearchProject(
            project_id='test_proj',
            title='Test Project',
            description='Test',
            research_questions=['Test question?'],
            hypotheses=[]
        )
        
        task = {
            'datasets': datasets,
            'experiments': experiments
        }
        
        result = data_agent.execute_task(task, project)
        
        assert 'datasets_analyzed' in result
        assert 'analysis_quality' in result
        assert 'individual_analyses' in result
        assert 'synthesis' in result
        assert result['datasets_analyzed'] == 2
    
    def test_autonomous_research_cycle(self, research_director):
        """Test autonomous research cycle execution."""
        # Create a test project
        project_spec = {
            'title': 'Test Autonomous Cycle',
            'description': 'Testing autonomous research cycle',
            'research_questions': ['Can AI conduct autonomous research?'],
            'domain': 'materials_science'
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        
        # Run short research cycle
        result = research_director.execute_autonomous_research_cycle(project_id, num_cycles=3)
        
        assert 'project_id' in result
        assert 'cycles_completed' in result
        assert 'phase_transitions' in result
        assert 'discoveries' in result
        assert result['cycles_completed'] == 3
        assert result['project_id'] == project_id
    
    def test_research_phase_execution(self, research_director):
        """Test individual research phase execution."""
        project = ResearchProject(
            project_id='test_proj',
            title='Test Project',
            description='Test',
            research_questions=['Test question?'],
            hypotheses=[],
            current_phase=ResearchPhase.HYPOTHESIS_GENERATION
        )
        
        # Test hypothesis generation phase
        result = research_director._execute_research_phase(
            project, ResearchPhase.HYPOTHESIS_GENERATION
        )
        
        assert 'hypotheses_generated' in result or 'phase' in result
        
        # Test data collection phase  
        result = research_director._execute_research_phase(
            project, ResearchPhase.DATA_COLLECTION
        )
        
        assert 'phase' in result
        assert result['phase'] == 'data_collection'
    
    def test_project_progress_tracking(self, research_director):
        """Test project progress tracking."""
        project = ResearchProject(
            project_id='test_proj',
            title='Test Project',
            description='Test',
            research_questions=['Test question?'],
            hypotheses=[]
        )
        
        initial_progress = project.completion_percentage
        
        # Simulate cycle result
        cycle_result = {
            'average_quality': 0.8,
            'ready_for_next_phase': True
        }
        
        research_director._update_project_progress(project, cycle_result)
        
        assert project.completion_percentage > initial_progress
    
    def test_research_summary(self, research_director):
        """Test research summary generation."""
        # Create and run a test project
        project_spec = {
            'title': 'Summary Test Project',
            'description': 'Testing summary generation'
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        research_director.execute_autonomous_research_cycle(project_id, num_cycles=2)
        
        summary = research_director.get_research_summary()
        
        assert 'active_projects' in summary
        assert 'agent_performance' in summary
        assert 'resource_utilization' in summary
        assert summary['active_projects'] >= 1


class TestIntegrationGeneration5:
    """Integration tests for Generation 5 components working together."""
    
    def test_full_autonomous_pipeline(self):
        """Test full autonomous pipeline integration."""
        # Initialize all components
        research_director = AutonomousResearchDirector()
        meta_optimizer = MetaLearningOptimizer()
        
        # Create research project
        project_spec = {
            'title': 'Integration Test Research',
            'description': 'Full autonomous pipeline test',
            'research_questions': [
                'Can all Generation 5 components work together?',
                'What level of autonomy can be achieved?'
            ],
            'domain': 'materials_science',
            'priority': 0.9
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        
        # Execute research with meta-learning
        meta_task = {
            'id': 'research_meta_task',
            'type': 'research_coordination',
            'complexity': 0.8,
            'novelty': 0.9,
            'resource_requirement': 0.6
        }
        
        meta_result = meta_optimizer.meta_learn_task(meta_task)
        research_result = research_director.execute_autonomous_research_cycle(project_id, num_cycles=2)
        
        # Verify integration
        assert meta_result['performance'] > 0.0
        assert research_result['cycles_completed'] == 2
        assert project_id in research_director.active_projects
    
    def test_cognitive_research_integration(self):
        """Test cognitive architecture and research director integration."""
        cognitive_arch = CognitiveArchitecture()
        research_director = AutonomousResearchDirector()
        
        # Process research-related thoughts
        research_context = torch.randn(512)
        cognitive_result = cognitive_arch.process_input(research_context)
        
        # Use cognitive insights for research
        project_spec = {
            'title': 'Cognitively-Informed Research',
            'description': 'Research guided by cognitive architecture insights'
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        
        # Both should complete successfully
        assert cognitive_result['processing_time'] > 0
        assert project_id in research_director.active_projects
    
    def test_adaptive_learning_research_synergy(self):
        """Test adaptive learning system and research director synergy."""
        adaptive_learning = AdaptiveLearningSystem()
        research_director = AutonomousResearchDirector()
        
        # Learn from research data
        research_data = torch.randn(512)
        learning_result = adaptive_learning.adaptive_learn(research_data)
        
        # Apply learning to research
        project_spec = {
            'title': 'Adaptive Learning Research',
            'description': 'Research enhanced by adaptive learning'
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        
        # Verify both systems function
        assert learning_result['integrated_performance'] > 0
        assert project_id is not None
    
    def test_generation5_performance_metrics(self):
        """Test Generation 5 performance metrics and quality assurance."""
        # Initialize all Generation 5 components
        components = {
            'meta_optimizer': MetaLearningOptimizer(),
            'cognitive_arch': CognitiveArchitecture(),
            'adaptive_learning': AdaptiveLearningSystem(),
            'research_director': AutonomousResearchDirector()
        }
        
        performance_scores = {}
        
        # Test each component
        for name, component in components.items():
            start_time = time.time()
            
            if name == 'meta_optimizer':
                task = {'id': 'perf_test', 'type': 'performance', 'complexity': 0.6}
                result = component.meta_learn_task(task)
                performance_scores[name] = result['performance']
                
            elif name == 'cognitive_arch':
                test_input = torch.randn(512)
                result = component.process_input(test_input)
                performance_scores[name] = 1.0 / (result['processing_time'] + 0.1)
                
            elif name == 'adaptive_learning':
                test_data = torch.randn(512)
                result = component.adaptive_learn(test_data)
                performance_scores[name] = result['integrated_performance']
                
            elif name == 'research_director':
                project_spec = {'title': 'Performance Test'}
                project_id = component.initiate_research_project(project_spec)
                result = component.execute_autonomous_research_cycle(project_id, num_cycles=1)
                performance_scores[name] = result['cycles_completed'] / 1.0
            
            elapsed = time.time() - start_time
            assert elapsed < 30.0, f"{name} took too long: {elapsed:.2f}s"
        
        # Verify all components meet performance thresholds
        for name, score in performance_scores.items():
            assert score > 0.0, f"{name} performance too low: {score}"
            
        # Calculate overall Generation 5 score
        overall_score = np.mean(list(performance_scores.values()))
        assert overall_score > 0.3, f"Overall Generation 5 performance too low: {overall_score:.4f}"
        
        print(f"\nGeneration 5 Performance Summary:")
        for name, score in performance_scores.items():
            print(f"  {name}: {score:.4f}")
        print(f"  Overall: {overall_score:.4f}")


if __name__ == "__main__":
    # Run comprehensive Generation 5 tests
    pytest.main([__file__, "-v", "--tb=short"])
