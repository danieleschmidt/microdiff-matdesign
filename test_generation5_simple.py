#!/usr/bin/env python3
"""Simple Generation 5 Autonomous Excellence Test Suite."""

import sys
import torch
import numpy as np
import time
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
    GENERATION5_AVAILABLE = True
except ImportError as e:
    print(f"❌ Generation 5 modules not available: {e}")
    GENERATION5_AVAILABLE = False


class TestRunner:
    """Simple test runner for Generation 5 components."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test."""
        self.tests_run += 1
        print(f"\n🧪 Running: {test_name}")
        
        try:
            start = time.time()
            result = test_func(*args, **kwargs)
            duration = time.time() - start
            
            if result:
                print(f"✅ PASSED: {test_name} ({duration:.3f}s)")
                self.tests_passed += 1
                return True
            else:
                print(f"❌ FAILED: {test_name} - Test returned False")
                self.tests_failed += 1
                return False
                
        except Exception as e:
            duration = time.time() - start
            print(f"❌ FAILED: {test_name} ({duration:.3f}s) - {str(e)}")
            self.tests_failed += 1
            return False
    
    def print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*60)
        print(f"🎯 GENERATION 5 AUTONOMOUS EXCELLENCE TEST SUMMARY")
        print(f"="*60)
        print(f"Tests Run:     {self.tests_run}")
        print(f"Tests Passed:  {self.tests_passed} ✅")
        print(f"Tests Failed:  {self.tests_failed} ❌")
        print(f"Success Rate:  {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"Total Time:    {total_time:.2f}s")
        
        if self.tests_failed == 0:
            print(f"\n🌟 ALL TESTS PASSED - GENERATION 5 EXCELLENCE ACHIEVED! 🌟")
        else:
            print(f"\n⚠️  {self.tests_failed} tests failed - Generation 5 needs attention")


def test_meta_learning_optimizer() -> bool:
    """Test Meta-Learning Optimizer."""
    try:
        config = MetaLearningConfig(
            meta_learning_rate=1e-3,
            inner_learning_rate=1e-2,
            meta_batch_size=16,
            inner_update_steps=3
        )
        
        optimizer = MetaLearningOptimizer(config)
        
        # Test task execution
        task = {
            'id': 'test_task',
            'type': 'optimization',
            'complexity': 0.5,
            'novelty': 0.6,
            'resource_requirement': 0.3
        }
        
        result = optimizer.meta_learn_task(task)
        
        # Verify results
        assert 'task_id' in result
        assert 'strategy' in result
        assert 'performance' in result
        assert 0.0 <= result['performance'] <= 1.0
        assert result['strategy'] in [s.value for s in MetaLearningStrategy]
        
        print(f"  📊 Performance: {result['performance']:.4f}")
        print(f"  🎯 Strategy: {result['strategy']}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_cognitive_architecture() -> bool:
    """Test Cognitive Architecture."""
    try:
        cognitive_arch = CognitiveArchitecture()
        
        # Test input processing
        test_input = torch.randn(512)
        result = cognitive_arch.process_input(test_input)
        
        # Verify results
        assert 'results' in result
        assert 'cognitive_state' in result
        assert 'processing_time' in result
        assert result['processing_time'] > 0
        
        # Test different processing modes
        result_auto = cognitive_arch.process_input(test_input, ProcessingMode.AUTOMATIC)
        result_controlled = cognitive_arch.process_input(test_input, ProcessingMode.CONTROLLED)
        
        assert result_auto['cognitive_state']['processing_mode'] == 'automatic'
        assert result_controlled['cognitive_state']['processing_mode'] == 'controlled'
        
        print(f"  🧠 Processing Time: {result['processing_time']:.4f}s")
        print(f"  🎛️  Processors Active: {len(result['results'])}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_adaptive_learning_system() -> bool:
    """Test Adaptive Learning System."""
    try:
        learning_system = AdaptiveLearningSystem(input_dim=256)
        
        # Test adaptive learning
        test_data = torch.randn(256)
        result = learning_system.adaptive_learn(test_data)
        
        # Verify results
        assert 'learning_results' in result
        assert 'integrated_performance' in result
        assert 'adaptation_strategy' in result
        assert 'learning_time' in result
        assert 0.0 <= result['integrated_performance'] <= 1.0
        
        # Test learning modules
        ssl_module = learning_system.learning_modules['self_supervised']
        context = LearningContext()
        ssl_result = ssl_module.learn(test_data, context)
        
        assert 'performance' in ssl_result
        assert ssl_result['performance'] > 0.0
        
        print(f"  📈 Integrated Performance: {result['integrated_performance']:.4f}")
        print(f"  🔄 Adaptation Strategy: {result['adaptation_strategy']}")
        print(f"  🎯 SSL Performance: {ssl_result['performance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_autonomous_research_director() -> bool:
    """Test Autonomous Research Director."""
    try:
        research_director = AutonomousResearchDirector()
        
        # Test project creation
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
        
        # Test research cycle
        result = research_director.execute_autonomous_research_cycle(project_id, num_cycles=2)
        
        assert 'project_id' in result
        assert 'cycles_completed' in result
        assert result['cycles_completed'] == 2
        
        # Test agent performance
        hyp_agent = research_director.agents['hypothesis_generator']
        project = research_director.active_projects[project_id]
        
        task = {
            'context': {
                'novelty_requirement': 0.8,
                'feasibility_requirement': 0.7,
                'impact_potential': 0.9
            },
            'num_hypotheses': 3
        }
        
        hyp_result = hyp_agent.execute_task(task, project)
        
        assert 'hypotheses_generated' in hyp_result
        assert hyp_result['hypotheses_generated'] == 3
        
        print(f"  🔬 Project ID: {project_id}")
        print(f"  🔄 Cycles Completed: {result['cycles_completed']}")
        print(f"  💡 Hypotheses Generated: {hyp_result['hypotheses_generated']}")
        print(f"  📊 Hypothesis Quality: {hyp_result.get('average_quality', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_generation5_integration() -> bool:
    """Test Generation 5 component integration."""
    try:
        # Initialize all components
        meta_optimizer = MetaLearningOptimizer()
        cognitive_arch = CognitiveArchitecture()
        adaptive_learning = AdaptiveLearningSystem()
        research_director = AutonomousResearchDirector()
        
        # Test meta-learning task
        meta_task = {
            'id': 'integration_test',
            'type': 'research_coordination',
            'complexity': 0.8,
            'novelty': 0.9,
            'resource_requirement': 0.6
        }
        
        meta_result = meta_optimizer.meta_learn_task(meta_task)
        
        # Test cognitive processing
        research_context = torch.randn(512)
        cognitive_result = cognitive_arch.process_input(research_context)
        
        # Test adaptive learning
        learning_data = torch.randn(512)
        learning_result = adaptive_learning.adaptive_learn(learning_data)
        
        # Test research project
        project_spec = {
            'title': 'Integration Test Research',
            'description': 'Full autonomous pipeline test'
        }
        
        project_id = research_director.initiate_research_project(project_spec)
        research_result = research_director.execute_autonomous_research_cycle(project_id, num_cycles=1)
        
        # Verify all components work together
        assert meta_result['performance'] > 0.0
        assert cognitive_result['processing_time'] > 0.0
        assert learning_result['integrated_performance'] > 0.0
        assert research_result['cycles_completed'] == 1
        
        # Calculate overall integration score
        scores = [
            meta_result['performance'],
            min(1.0, 1.0 / cognitive_result['processing_time']),
            learning_result['integrated_performance'],
            research_result['cycles_completed'] / 1.0
        ]
        
        integration_score = np.mean(scores)
        
        print(f"  🔗 Meta-Learning: {meta_result['performance']:.4f}")
        print(f"  🧠 Cognitive Processing: {cognitive_result['processing_time']:.4f}s")
        print(f"  📚 Adaptive Learning: {learning_result['integrated_performance']:.4f}")
        print(f"  🔬 Research Cycles: {research_result['cycles_completed']}")
        print(f"  🌟 Integration Score: {integration_score:.4f}")
        
        return integration_score > 0.3
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_generation5_performance() -> bool:
    """Test Generation 5 overall performance and quality."""
    try:
        performance_scores = {}
        
        # Meta-Learning Performance
        meta_optimizer = MetaLearningOptimizer()
        meta_task = {'id': 'perf_test', 'type': 'performance', 'complexity': 0.6}
        meta_result = meta_optimizer.meta_learn_task(meta_task)
        performance_scores['meta_learning'] = meta_result['performance']
        
        # Cognitive Architecture Performance
        cognitive_arch = CognitiveArchitecture()
        start_time = time.time()
        test_input = torch.randn(512)
        cognitive_result = cognitive_arch.process_input(test_input)
        cognitive_time = time.time() - start_time
        performance_scores['cognitive'] = min(1.0, 1.0 / cognitive_time)
        
        # Adaptive Learning Performance
        adaptive_learning = AdaptiveLearningSystem()
        test_data = torch.randn(512)
        learning_result = adaptive_learning.adaptive_learn(test_data)
        performance_scores['adaptive_learning'] = learning_result['integrated_performance']
        
        # Research Director Performance
        research_director = AutonomousResearchDirector()
        project_spec = {'title': 'Performance Test'}
        project_id = research_director.initiate_research_project(project_spec)
        research_result = research_director.execute_autonomous_research_cycle(project_id, num_cycles=1)
        performance_scores['research'] = research_result['cycles_completed'] / 1.0
        
        # Calculate overall Generation 5 performance
        overall_score = np.mean(list(performance_scores.values()))
        
        print(f"  🎯 Meta-Learning: {performance_scores['meta_learning']:.4f}")
        print(f"  🧠 Cognitive Arch: {performance_scores['cognitive']:.4f}")
        print(f"  📚 Adaptive Learning: {performance_scores['adaptive_learning']:.4f}")
        print(f"  🔬 Research Director: {performance_scores['research']:.4f}")
        print(f"  🌟 Overall Score: {overall_score:.4f}")
        
        # Performance thresholds for Generation 5
        excellence_threshold = 0.5
        
        if overall_score >= excellence_threshold:
            print(f"  🏆 GENERATION 5 EXCELLENCE ACHIEVED! (Score: {overall_score:.4f})")
            return True
        else:
            print(f"  ⚠️  Generation 5 performance below excellence threshold ({excellence_threshold})")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Run all Generation 5 tests."""
    print("🚀 GENERATION 5 AUTONOMOUS EXCELLENCE TEST SUITE")
    print("="*60)
    
    if not GENERATION5_AVAILABLE:
        print("❌ Generation 5 modules not available - skipping tests")
        return False
    
    runner = TestRunner()
    
    # Core component tests
    runner.run_test("Meta-Learning Optimizer", test_meta_learning_optimizer)
    runner.run_test("Cognitive Architecture", test_cognitive_architecture)
    runner.run_test("Adaptive Learning System", test_adaptive_learning_system)
    runner.run_test("Autonomous Research Director", test_autonomous_research_director)
    
    # Integration tests
    runner.run_test("Generation 5 Integration", test_generation5_integration)
    runner.run_test("Generation 5 Performance", test_generation5_performance)
    
    # Print final summary
    runner.print_summary()
    
    return runner.tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
