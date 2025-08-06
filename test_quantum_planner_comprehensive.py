"""Comprehensive test suite for quantum task planner."""

import pytest
import numpy as np
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import time

# Import quantum planner components
from quantum_planner.core.task import Task, TaskPriority, TaskStatus
from quantum_planner.core.scheduler import QuantumInspiredScheduler, Resource
from quantum_planner.core.quantum_engine import QuantumEngine, QuantumState
from quantum_planner.algorithms.quantum_annealing import QuantumAnnealingOptimizer
from quantum_planner.algorithms.superposition import SuperpositionScheduler
from quantum_planner.utils.validation import TaskValidator, ResourceValidator, ValidationResult
from quantum_planner.utils.error_handling import ErrorHandler, QuantumPlannerError
from quantum_planner.utils.performance import PerformanceMonitor
from quantum_planner.utils.scaling import ScalingOrchestrator, ScalingConfig
from quantum_planner.utils.logging_config import setup_logging


class TestTaskManagement:
    """Test task creation, validation, and management."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.validator = TaskValidator()
    
    def test_task_creation_basic(self):
        """Test basic task creation."""
        task = Task(
            name="Test Task",
            description="A test task",
            priority=TaskPriority.HIGH,
            estimated_duration=60
        )
        
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.estimated_duration == 60
        assert task.is_ready == True
        assert task.is_overdue == False
    
    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task1 = Task(name="Task 1", estimated_duration=30)
        task2 = Task(name="Task 2", estimated_duration=45)
        
        task2.add_dependency(task1.id)
        
        assert task1.id in task2.dependencies
        assert task2.is_ready == False  # Has unmet dependencies
    
    def test_task_with_deadline(self):
        """Test task with deadline."""
        future_deadline = datetime.now() + timedelta(hours=2)
        past_deadline = datetime.now() - timedelta(hours=1)
        
        task_future = Task(
            name="Future Deadline Task",
            deadline=future_deadline,
            estimated_duration=60
        )
        
        task_past = Task(
            name="Past Deadline Task", 
            deadline=past_deadline,
            estimated_duration=60
        )
        
        assert task_future.is_overdue == False
        assert task_past.is_overdue == True
        assert task_past.urgency_factor > task_future.urgency_factor
    
    def test_task_quantum_properties(self):
        """Test quantum properties of tasks."""
        task = Task(name="Quantum Task", estimated_duration=60)
        
        # Test superposition states
        task.add_superposition_state(
            0.6, duration=50, outcome="fast"
        )
        task.add_superposition_state(
            0.4, duration=80, outcome="slow"
        )
        
        assert len(task.superposition_states) == 3  # Including default state
        
        # Test entanglement
        other_task = Task(name="Other Task")
        task.entangle_with(other_task.id)
        
        assert other_task.id in task.entanglement_partners
    
    def test_task_validation(self):
        """Test task validation."""
        # Valid task
        valid_task = Task(
            name="Valid Task",
            priority=TaskPriority.HIGH,
            estimated_duration=120
        )
        
        result = self.validator.validate_task(valid_task)
        assert result.is_valid() == True
        
        # Invalid task
        invalid_task = Task(
            name="",  # Empty name
            estimated_duration=-10  # Negative duration
        )
        
        result = self.validator.validate_task(invalid_task)
        assert result.is_valid() == False
        assert len(result.errors) > 0
    
    def test_task_serialization(self):
        """Test task serialization/deserialization."""
        original_task = Task(
            name="Serialization Test",
            description="Test task serialization",
            priority=TaskPriority.MEDIUM,
            estimated_duration=90,
            deadline=datetime.now() + timedelta(hours=24)
        )
        original_task.add_dependency("dep-1")
        original_task.add_conflict("conflict-1")
        
        # Serialize
        task_dict = original_task.to_dict()
        assert isinstance(task_dict, dict)
        assert task_dict['name'] == "Serialization Test"
        
        # Deserialize
        restored_task = Task.from_dict(task_dict)
        assert restored_task.name == original_task.name
        assert restored_task.priority == original_task.priority
        assert restored_task.estimated_duration == original_task.estimated_duration
        assert restored_task.dependencies == original_task.dependencies


class TestQuantumEngine:
    """Test quantum computation engine."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.engine = QuantumEngine(num_qubits=8, max_iterations=100)
    
    def test_quantum_engine_initialization(self):
        """Test quantum engine initialization."""
        assert self.engine.num_qubits == 8
        assert self.engine.max_iterations == 100
        assert len(self.engine.quantum_states) == 0
    
    def test_quantum_state_initialization(self):
        """Test quantum state space initialization."""
        # Create sample tasks and resources
        tasks = [
            Task(name=f"Task {i}", estimated_duration=30 + i*10)
            for i in range(3)
        ]
        resources = {"cpu": 2, "memory": 4, "storage": 1}
        
        self.engine.initialize_quantum_state(tasks, resources)
        
        assert len(self.engine.quantum_states) > 0
        assert len(self.engine.task_qubit_map) == len(tasks)
        
        # Check that all states have valid properties
        for state in self.engine.quantum_states:
            assert isinstance(state.amplitude, complex)
            assert 0 <= state.probability <= 1
            assert isinstance(state.task_assignments, dict)
            assert isinstance(state.energy, float)
    
    def test_quantum_annealing(self):
        """Test quantum annealing optimization."""
        tasks = [
            Task(name="High Priority", priority=TaskPriority.HIGH, estimated_duration=60),
            Task(name="Medium Priority", priority=TaskPriority.MEDIUM, estimated_duration=45),
            Task(name="Low Priority", priority=TaskPriority.LOW, estimated_duration=30)
        ]
        resources = {"cpu": 2, "memory": 2}
        
        assignment = self.engine.apply_quantum_annealing(tasks, resources)
        
        assert isinstance(assignment, dict)
        assert len(assignment) == len(tasks)
        
        # Verify all tasks are assigned
        for task in tasks:
            assert task.id in assignment
            assert assignment[task.id] in resources
    
    def test_quantum_superposition(self):
        """Test quantum superposition application."""
        task = Task(name="Superposition Task", estimated_duration=60)
        superposition_states = self.engine.apply_quantum_superposition(task.id)
        
        assert isinstance(superposition_states, list)
        if len(superposition_states) > 0:
            for state in superposition_states:
                assert 'amplitude' in state
                assert 'probability' in state
                assert 'scenario' in state
    
    def test_quantum_entanglement(self):
        """Test quantum entanglement creation."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        tasks = [task1, task2]
        resources = {"cpu": 1}
        
        self.engine.initialize_quantum_state(tasks, resources)
        self.engine.create_task_entanglement(task1.id, task2.id, strength=0.8)
        
        assert self.engine.entanglement_matrix[0, 1] == 0.8
        assert self.engine.entanglement_matrix[1, 0] == 0.8
    
    def test_quantum_measurement(self):
        """Test quantum state measurement."""
        tasks = [Task(name="Measurement Task", estimated_duration=30)]
        resources = {"cpu": 1}
        
        self.engine.initialize_quantum_state(tasks, resources)
        assignment = self.engine.measure_quantum_state()
        
        assert isinstance(assignment, dict)
        
        # After measurement, should have only one state (collapsed)
        assignment_collapsed = self.engine.measure_quantum_state(collapse_probability=True)
        assert len(self.engine.quantum_states) == 1
    
    def test_quantum_metrics(self):
        """Test quantum metrics calculation."""
        tasks = [Task(name=f"Task {i}", estimated_duration=30) for i in range(3)]
        resources = {"cpu": 2}
        
        self.engine.initialize_quantum_state(tasks, resources)
        metrics = self.engine.get_quantum_metrics()
        
        assert isinstance(metrics, dict)
        expected_keys = ['min_energy', 'max_energy', 'avg_energy', 'entropy', 'num_states']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


class TestSchedulingAlgorithms:
    """Test quantum scheduling algorithms."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tasks = [
            Task(name="Critical Task", priority=TaskPriority.CRITICAL, estimated_duration=30),
            Task(name="High Task", priority=TaskPriority.HIGH, estimated_duration=45),
            Task(name="Medium Task", priority=TaskPriority.MEDIUM, estimated_duration=60),
            Task(name="Low Task", priority=TaskPriority.LOW, estimated_duration=20)
        ]
        
        self.resources = {
            "cpu": Resource("cpu", "CPU Resource", 2),
            "memory": Resource("memory", "Memory Resource", 3),
            "storage": Resource("storage", "Storage Resource", 1)
        }
    
    def test_quantum_annealing_optimizer(self):
        """Test quantum annealing optimizer."""
        optimizer = QuantumAnnealingOptimizer()
        resource_dict = {rid: r.capacity for rid, r in self.resources.items()}
        
        assignment = optimizer.optimize(self.tasks, resource_dict)
        
        assert isinstance(assignment, dict)
        assert len(assignment) == len(self.tasks)
        
        # Verify optimization metrics
        metrics = optimizer.get_optimization_metrics()
        assert 'best_energy' in metrics
        assert 'improvement_ratio' in metrics
    
    def test_superposition_scheduler(self):
        """Test superposition scheduler."""
        scheduler = SuperpositionScheduler()
        schedule = scheduler.create_schedule(self.tasks, self.resources)
        
        assert isinstance(schedule, dict)
        
        # Verify schedule structure
        for task_id, task_schedule in schedule.items():
            assert 'start_time' in task_schedule
            assert 'end_time' in task_schedule
            assert 'duration' in task_schedule
            assert 'resource_id' in task_schedule
            
            # Verify timing consistency
            assert task_schedule['end_time'] > task_schedule['start_time']
    
    def test_scheduler_with_dependencies(self):
        """Test scheduling with task dependencies."""
        # Create tasks with dependencies
        task1 = Task(name="Foundation", estimated_duration=30)
        task2 = Task(name="Dependent", estimated_duration=45)
        task2.add_dependency(task1.id)
        
        tasks_with_deps = [task1, task2]
        
        scheduler = SuperpositionScheduler()
        schedule = scheduler.create_schedule(tasks_with_deps, self.resources)
        
        # Verify dependency is respected
        if task1.id in schedule and task2.id in schedule:
            task1_end = schedule[task1.id]['end_time']
            task2_start = schedule[task2.id]['start_time']
            assert task2_start >= task1_end
    
    def test_scheduler_with_conflicts(self):
        """Test scheduling with task conflicts."""
        task1 = Task(name="Conflicting Task 1", estimated_duration=60)
        task2 = Task(name="Conflicting Task 2", estimated_duration=45)
        task1.add_conflict(task2.id)
        task2.add_conflict(task1.id)
        
        conflicting_tasks = [task1, task2]
        
        scheduler = SuperpositionScheduler()
        schedule = scheduler.create_schedule(conflicting_tasks, self.resources)
        
        # Verify conflicts are handled
        if task1.id in schedule and task2.id in schedule:
            task1_info = schedule[task1.id]
            task2_info = schedule[task2.id]
            
            # Tasks should not overlap if on same resource
            if task1_info['resource_id'] == task2_info['resource_id']:
                assert (task1_info['end_time'] <= task2_info['start_time'] or
                       task2_info['end_time'] <= task1_info['start_time'])


class TestQuantumInspiredScheduler:
    """Test main quantum-inspired scheduler."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.scheduler = QuantumInspiredScheduler(
            max_parallel_tasks=5,
            optimization_method="quantum_annealing"
        )
        
        # Create test tasks
        self.test_tasks = [
            Task(name="Task A", priority=TaskPriority.HIGH, estimated_duration=60),
            Task(name="Task B", priority=TaskPriority.MEDIUM, estimated_duration=45),
            Task(name="Task C", priority=TaskPriority.LOW, estimated_duration=30)
        ]
        
        # Create test resources
        self.test_resources = [
            Resource("cpu-1", "CPU Resource 1", 2),
            Resource("cpu-2", "CPU Resource 2", 1),
            Resource("memory-1", "Memory Resource", 3)
        ]
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.max_parallel_tasks == 5
        assert self.scheduler.optimization_method == "quantum_annealing"
        assert len(self.scheduler.tasks) == 0
        assert len(self.scheduler.resources) == 0
    
    def test_add_tasks_and_resources(self):
        """Test adding tasks and resources."""
        # Add tasks
        for task in self.test_tasks:
            self.scheduler.add_task(task)
        
        assert len(self.scheduler.tasks) == len(self.test_tasks)
        
        # Add resources
        for resource in self.test_resources:
            self.scheduler.add_resource(resource)
        
        assert len(self.scheduler.resources) == len(self.test_resources)
    
    def test_create_optimal_schedule(self):
        """Test creating optimal schedule."""
        # Add tasks and resources
        for task in self.test_tasks:
            self.scheduler.add_task(task)
        
        for resource in self.test_resources:
            self.scheduler.add_resource(resource)
        
        # Create schedule
        result = self.scheduler.create_optimal_schedule(optimization_time_limit=30)
        
        assert result.success == True
        assert isinstance(result.schedule, dict)
        assert result.total_completion_time > 0
        assert isinstance(result.resource_utilization, dict)
    
    def test_schedule_execution_simulation(self):
        """Test schedule execution simulation."""
        # Setup tasks and resources
        for task in self.test_tasks:
            self.scheduler.add_task(task)
        
        for resource in self.test_resources:
            self.scheduler.add_resource(resource)
        
        # Create and execute schedule
        schedule_result = self.scheduler.create_optimal_schedule()
        
        if schedule_result.success:
            execution_result = self.scheduler.execute_schedule(schedule_result, simulate=True)
            
            assert execution_result['success'] == True
            assert execution_result['simulation'] == True
            assert 'execution_log' in execution_result
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        # Setup
        for task in self.test_tasks:
            self.scheduler.add_task(task)
        
        for resource in self.test_resources:
            self.scheduler.add_resource(resource)
        
        # Multi-objective optimization
        objectives = {
            'completion_time': 200,  # Target completion time
            'resource_utilization': 0.8  # Target utilization
        }
        
        result = self.scheduler.optimize_for_multiple_objectives(objectives)
        
        assert isinstance(result.optimization_metrics, dict)
        if result.success:
            assert 'multi_objective_score' in result.optimization_metrics
    
    def test_resource_utilization_tracking(self):
        """Test resource utilization tracking."""
        # Add resources
        for resource in self.test_resources:
            self.scheduler.add_resource(resource)
        
        utilization = self.scheduler.get_resource_utilization()
        
        assert isinstance(utilization, dict)
        for resource_id, util_info in utilization.items():
            assert 'capacity' in util_info
            assert 'available_capacity' in util_info
            assert 'utilization_percentage' in util_info
    
    def test_dependency_graph_generation(self):
        """Test dependency graph generation."""
        # Create tasks with dependencies
        task1 = Task(name="Base Task", estimated_duration=30)
        task2 = Task(name="Dependent Task", estimated_duration=45)
        task2.add_dependency(task1.id)
        task2.entangle_with(task1.id)
        
        self.scheduler.add_task(task1)
        self.scheduler.add_task(task2)
        
        graph = self.scheduler.get_task_dependencies_graph()
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert len(graph['nodes']) == 2
        assert len(graph['edges']) >= 1  # At least the dependency edge


class TestValidationSystem:
    """Test comprehensive validation system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.task_validator = TaskValidator(strict_mode=False)
        self.resource_validator = ResourceValidator()
    
    def test_task_validation_success(self):
        """Test successful task validation."""
        valid_task = Task(
            name="Valid Task",
            description="A properly configured task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=90,
            deadline=datetime.now() + timedelta(hours=24)
        )
        
        result = self.task_validator.validate_task(valid_task)
        assert result.is_valid() == True
        assert len(result.errors) == 0
    
    def test_task_validation_failures(self):
        """Test task validation failures."""
        invalid_task = Task(
            name="",  # Invalid: empty name
            estimated_duration=-30,  # Invalid: negative duration
            deadline=datetime.now() - timedelta(hours=1)  # Warning: past deadline
        )
        
        result = self.task_validator.validate_task(invalid_task)
        assert result.is_valid() == False
        assert len(result.errors) > 0
        assert len(result.warnings) >= 0
    
    def test_task_list_validation(self):
        """Test validation of task lists."""
        task1 = Task(name="Task 1", estimated_duration=60)
        task2 = Task(name="Task 2", estimated_duration=45)
        task2.add_dependency(task1.id)  # Valid dependency
        
        task3 = Task(name="Task 3", estimated_duration=30)
        task3.add_dependency("non-existent-task")  # Invalid dependency
        
        tasks = [task1, task2, task3]
        result = self.task_validator.validate_task_list(tasks)
        
        assert len(result.errors) > 0  # Should have error for non-existent dependency
    
    def test_resource_validation(self):
        """Test resource validation."""
        valid_resource = Resource("cpu-1", "CPU Resource", 4, 2, 0.5)
        invalid_resource = Resource("", "Invalid", -1, 5, -0.1)  # Multiple errors
        
        valid_result = self.resource_validator.validate_resource(valid_resource)
        invalid_result = self.resource_validator.validate_resource(invalid_resource)
        
        assert valid_result.is_valid() == True
        assert invalid_result.is_valid() == False
        assert len(invalid_result.errors) > 0
    
    def test_dependency_cycle_detection(self):
        """Test dependency cycle detection."""
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2") 
        task3 = Task(name="Task 3")
        
        # Create cycle: 1 -> 2 -> 3 -> 1
        task2.add_dependency(task1.id)
        task3.add_dependency(task2.id)
        task1.add_dependency(task3.id)  # Creates cycle
        
        tasks = [task1, task2, task3]
        result = self.task_validator.validate_task_list(tasks)
        
        # Should detect the cycle
        cycle_errors = [error for error in result.errors if "cycle" in error.lower()]
        assert len(cycle_errors) > 0


class TestErrorHandlingSystem:
    """Test error handling and recovery system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.error_handler = ErrorHandler()
    
    def test_error_handling_basic(self):
        """Test basic error handling."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            success = self.error_handler.handle_error(e, {"test": "context"})
            
        # Check error was recorded
        stats = self.error_handler.get_error_statistics()
        assert stats['total_errors'] > 0
        
        # Check recent errors
        recent = self.error_handler.get_recent_errors(1)
        assert len(recent) > 0
        assert recent[0].message == "Test error"
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Register custom recovery handler
        def test_recovery_handler(exception, error_info):
            return True  # Simulate successful recovery
        
        self.error_handler.register_recovery_handler(ValueError, test_recovery_handler)
        
        try:
            raise ValueError("Recoverable error")
        except Exception as e:
            recovery_successful = self.error_handler.handle_error(e, attempt_recovery=True)
            
        assert recovery_successful == True
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        # Generate various errors
        errors = [
            ValueError("Validation error"),
            RuntimeError("Runtime error"),
            MemoryError("Memory error")
        ]
        
        for error in errors:
            try:
                raise error
            except Exception as e:
                self.error_handler.handle_error(e)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] >= len(errors)
        assert 'severity_distribution' in stats
        assert 'category_distribution' in stats
        assert 'most_common_errors' in stats
    
    def test_custom_quantum_errors(self):
        """Test custom quantum planner errors."""
        from quantum_planner.utils.error_handling import QuantumComputationError, ValidationError
        
        quantum_error = QuantumComputationError(
            "Quantum state collapse failed",
            operation="measurement",
            quantum_state={"amplitude": 0.5}
        )
        
        validation_error = ValidationError(
            "Invalid task priority",
            field="priority",
            value="INVALID"
        )
        
        # Handle custom errors
        for error in [quantum_error, validation_error]:
            try:
                raise error
            except Exception as e:
                self.error_handler.handle_error(e)
        
        # Verify custom error properties are preserved
        recent_errors = self.error_handler.get_recent_errors(5)
        quantum_errors = [e for e in recent_errors if "quantum" in e.message.lower()]
        validation_errors = [e for e in recent_errors if "validation" in e.message.lower()]
        
        assert len(quantum_errors) > 0
        assert len(validation_errors) > 0


class TestPerformanceMonitoring:
    """Test performance monitoring system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.monitor = PerformanceMonitor()
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        operation_id = self.monitor.start_operation("test_operation")
        
        # Simulate work
        time.sleep(0.1)
        
        duration = self.monitor.end_operation("test_operation", operation_id)
        
        assert duration >= 0.1
        
        # Check statistics
        stats = self.monitor.get_operation_stats("test_operation")
        assert "test_operation" in stats
        assert stats["test_operation"].count == 1
        assert stats["test_operation"].avg_time >= 0.1
    
    def test_custom_metrics(self):
        """Test custom metric recording."""
        self.monitor.record_metric("custom_metric", 42.5, category="test")
        self.monitor.record_metric("custom_metric", 35.7, category="test")
        
        summary = self.monitor.get_metrics_summary()
        
        assert 'metric_stats' in summary
        assert 'custom_metric' in summary['metric_stats']
        
        metric_stats = summary['metric_stats']['custom_metric']
        assert metric_stats['count'] == 2
        assert metric_stats['avg'] > 35
        assert metric_stats['max'] == 42.5
    
    def test_system_metrics(self):
        """Test system metrics collection."""
        metrics = self.monitor.get_system_metrics()
        
        expected_keys = ['memory_total_gb', 'memory_used_gb', 'cpu_percent']
        for key in expected_keys:
            if key in metrics:  # Some metrics might not be available in test environment
                assert isinstance(metrics[key], (int, float))
                assert metrics[key] >= 0
    
    def test_performance_alerts(self):
        """Test performance alert generation."""
        # Simulate slow operation
        operation_id = self.monitor.start_operation("slow_operation")
        time.sleep(0.01)  # Simulate some work
        self.monitor.end_operation("slow_operation", operation_id)
        
        alerts = self.monitor.get_performance_alerts()
        assert isinstance(alerts, list)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        self.monitor.record_metric("export_test", 123.45)
        
        # Export as JSON
        json_export = self.monitor.export_metrics("json")
        assert isinstance(json_export, str)
        assert "export_test" in json_export
        
        # Export as CSV
        csv_export = self.monitor.export_metrics("csv")
        assert isinstance(csv_export, str)
        assert "timestamp,name,value,category" in csv_export


class TestScalingSystem:
    """Test scaling and optimization system."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = ScalingConfig(max_workers=4, enable_multiprocessing=True)
        self.orchestrator = ScalingOrchestrator(self.config)
    
    def test_scaling_config(self):
        """Test scaling configuration."""
        assert self.config.max_workers == 4
        assert self.config.enable_multiprocessing == True
        assert self.config.adaptive_scaling == True
    
    def test_task_processing_scaling(self):
        """Test scaled task processing."""
        # Create test tasks
        tasks = [
            Task(name=f"Task {i}", estimated_duration=30 + i*5)
            for i in range(10)
        ]
        
        def simple_processing_func(task):
            # Simulate processing
            return f"Processed {task.name}"
        
        # Process with scaling
        results = self.orchestrator.scale_task_processing(
            tasks, simple_processing_func
        )
        
        assert len(results) == len(tasks)
        assert all(result is not None for result in results)
    
    def test_worker_pool_management(self):
        """Test worker pool management."""
        from quantum_planner.utils.scaling import WorkerPool
        
        with WorkerPool(self.config) as pool:
            # Test CPU-intensive task submission
            def cpu_task():
                return sum(range(1000))
            
            future = pool.submit_cpu_intensive(cpu_task)
            result = future.result(timeout=10)
            
            assert result == sum(range(1000))
            
            # Test I/O-intensive task submission
            def io_task():
                time.sleep(0.01)
                return "IO Complete"
            
            future = pool.submit_io_intensive(io_task)
            result = future.result(timeout=10)
            
            assert result == "IO Complete"
    
    def test_parallel_mapping(self):
        """Test parallel mapping functionality."""
        from quantum_planner.utils.scaling import WorkerPool
        
        def square_function(x):
            return x ** 2
        
        input_data = list(range(20))
        
        with WorkerPool(self.config) as pool:
            results = pool.map_parallel(square_function, input_data, cpu_intensive=True)
        
        expected = [x ** 2 for x in input_data]
        assert results == expected
    
    def test_scaling_metrics(self):
        """Test scaling metrics collection."""
        metrics = self.orchestrator.optimize_system_resources()
        
        assert isinstance(metrics, dict)
        assert 'worker_metrics' in metrics or 'system_metrics' in metrics
    
    def test_scaling_recommendations(self):
        """Test scaling recommendations."""
        recommendations = self.orchestrator.get_scaling_recommendations()
        
        assert isinstance(recommendations, list)
        assert all(isinstance(rec, str) for rec in recommendations)


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def setup_method(self):
        """Setup for integration tests."""
        setup_logging(log_level="ERROR")  # Reduce log noise during tests
        
        # Create a realistic scenario
        self.tasks = [
            Task(name="Data Preparation", priority=TaskPriority.HIGH, estimated_duration=120),
            Task(name="Model Training", priority=TaskPriority.CRITICAL, estimated_duration=480),
            Task(name="Model Validation", priority=TaskPriority.HIGH, estimated_duration=90),
            Task(name="Report Generation", priority=TaskPriority.MEDIUM, estimated_duration=60),
            Task(name="Cleanup", priority=TaskPriority.LOW, estimated_duration=30)
        ]
        
        # Add dependencies
        self.tasks[1].add_dependency(self.tasks[0].id)  # Training depends on preparation
        self.tasks[2].add_dependency(self.tasks[1].id)  # Validation depends on training
        self.tasks[3].add_dependency(self.tasks[2].id)  # Report depends on validation
        self.tasks[4].add_dependency(self.tasks[3].id)  # Cleanup depends on report
        
        # Resources
        self.resources = [
            Resource("cpu-cluster", "CPU Cluster", 4, 4, 0.10),
            Resource("gpu-node", "GPU Node", 2, 2, 0.50),
            Resource("storage", "Storage System", 10, 10, 0.05)
        ]
    
    def test_complete_scheduling_workflow(self):
        """Test complete scheduling workflow from start to finish."""
        # Initialize scheduler with performance monitoring
        scheduler = QuantumInspiredScheduler(
            optimization_method="hybrid",
            performance_monitoring=True
        )
        
        # Add tasks and resources
        for task in self.tasks:
            scheduler.add_task(task)
        
        for resource in self.resources:
            scheduler.add_resource(resource)
        
        # Validate setup
        validator = TaskValidator()
        validation_result = validator.validate_task_list(self.tasks)
        assert validation_result.is_valid()
        
        # Create optimal schedule
        schedule_result = scheduler.create_optimal_schedule(optimization_time_limit=60)
        assert schedule_result.success
        
        # Verify schedule quality
        assert schedule_result.total_completion_time > 0
        assert schedule_result.dependencies_satisfied >= 4  # We have 4 dependencies
        assert len(schedule_result.schedule) == len(self.tasks)
        
        # Test schedule execution simulation
        execution_result = scheduler.execute_schedule(schedule_result, simulate=True)
        assert execution_result['success']
        assert execution_result['simulation']
    
    def test_file_io_integration(self):
        """Test file I/O integration for tasks and resources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_file = os.path.join(temp_dir, "tasks.json")
            resources_file = os.path.join(temp_dir, "resources.json")
            
            # Export tasks to JSON
            tasks_data = [task.to_dict() for task in self.tasks]
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, default=str)
            
            # Export resources to JSON
            resources_data = [
                {
                    'id': r.id,
                    'name': r.name,
                    'capacity': r.capacity,
                    'available_capacity': r.available_capacity,
                    'cost_per_minute': r.cost_per_minute
                } for r in self.resources
            ]
            with open(resources_file, 'w') as f:
                json.dump(resources_data, f)
            
            # Reload and verify
            with open(tasks_file, 'r') as f:
                loaded_tasks_data = json.load(f)
            
            loaded_tasks = [Task.from_dict(task_data) for task_data in loaded_tasks_data]
            
            assert len(loaded_tasks) == len(self.tasks)
            for original, loaded in zip(self.tasks, loaded_tasks):
                assert loaded.name == original.name
                assert loaded.priority == original.priority
                assert loaded.estimated_duration == original.estimated_duration
    
    def test_error_recovery_integration(self):
        """Test error recovery during scheduling."""
        # Create problematic scenario
        bad_task = Task(name="Problematic Task", estimated_duration=0)  # Will cause issues
        
        scheduler = QuantumInspiredScheduler()
        scheduler.add_task(bad_task)
        
        # This should handle errors gracefully
        result = scheduler.create_optimal_schedule()
        
        # Should either succeed with corrections or fail gracefully
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error_message != ""
    
    def test_performance_under_load(self):
        """Test system performance under load."""
        # Create large number of tasks
        large_task_set = [
            Task(name=f"Load Test Task {i}", 
                 priority=TaskPriority.MEDIUM,
                 estimated_duration=30 + (i % 60))
            for i in range(100)
        ]
        
        # Add some dependencies
        for i in range(1, len(large_task_set)):
            if i % 10 == 0:  # Every 10th task depends on previous
                large_task_set[i].add_dependency(large_task_set[i-1].id)
        
        # Create resources
        load_resources = [
            Resource(f"resource-{i}", f"Resource {i}", 3)
            for i in range(10)
        ]
        
        # Initialize scheduler with scaling
        config = ScalingConfig(max_workers=4, enable_multiprocessing=True)
        orchestrator = ScalingOrchestrator(config)
        
        scheduler = QuantumInspiredScheduler(optimization_method="superposition")
        
        for task in large_task_set:
            scheduler.add_task(task)
        for resource in load_resources:
            scheduler.add_resource(resource)
        
        # Measure performance
        start_time = time.time()
        result = scheduler.create_optimal_schedule(optimization_time_limit=30)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 60  # Should complete within reasonable time
        if result.success:
            assert len(result.schedule) <= len(large_task_set)
            assert result.total_completion_time > 0


# Pytest configuration and fixtures
@pytest.fixture
def sample_tasks():
    """Provide sample tasks for testing."""
    return [
        Task(name="Sample Task 1", priority=TaskPriority.HIGH, estimated_duration=60),
        Task(name="Sample Task 2", priority=TaskPriority.MEDIUM, estimated_duration=45),
        Task(name="Sample Task 3", priority=TaskPriority.LOW, estimated_duration=30)
    ]


@pytest.fixture 
def sample_resources():
    """Provide sample resources for testing."""
    return [
        Resource("cpu", "CPU Resource", 2),
        Resource("memory", "Memory Resource", 4),
        Resource("storage", "Storage Resource", 1)
    ]


@pytest.fixture
def configured_scheduler():
    """Provide configured scheduler for testing."""
    scheduler = QuantumInspiredScheduler(
        optimization_method="quantum_annealing",
        enable_superposition=True,
        enable_entanglement=True
    )
    return scheduler


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])