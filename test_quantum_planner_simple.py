"""Simplified test for quantum task planner without heavy dependencies."""

import sys
import os
import time
from datetime import datetime, timedelta

# Add quantum planner to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_functionality():
    """Test basic quantum planner functionality."""
    print("Testing basic quantum planner functionality...")
    
    # Test task creation
    from quantum_planner.core.task import Task, TaskPriority, TaskStatus
    
    task = Task(
        name="Test Task",
        description="A simple test task",
        priority=TaskPriority.HIGH,
        estimated_duration=60
    )
    
    assert task.name == "Test Task"
    assert task.priority == TaskPriority.HIGH
    assert task.status == TaskStatus.PENDING
    assert task.estimated_duration == 60
    
    print("✓ Task creation successful")
    
    # Test task dependencies
    task2 = Task(name="Dependent Task", estimated_duration=30)
    task2.add_dependency(task.id)
    
    assert task.id in task2.dependencies
    assert not task2.is_ready  # Should not be ready due to dependency
    
    print("✓ Task dependencies work correctly")
    
    # Test task quantum properties
    task.add_superposition_state(0.7, duration=50, outcome="fast")
    assert len(task.superposition_states) == 2  # Including default
    
    other_task = Task(name="Entangled Task")
    task.entangle_with(other_task.id)
    assert other_task.id in task.entanglement_partners
    
    print("✓ Quantum properties functional")
    
    # Test serialization
    task_dict = task.to_dict()
    restored_task = Task.from_dict(task_dict)
    
    assert restored_task.name == task.name
    assert restored_task.priority == task.priority
    assert restored_task.estimated_duration == task.estimated_duration
    
    print("✓ Task serialization works")
    
    return True


def test_validation():
    """Test validation functionality."""
    print("Testing validation system...")
    
    from quantum_planner.utils.validation import TaskValidator
    from quantum_planner.core.task import Task, TaskPriority
    
    validator = TaskValidator()
    
    # Test valid task
    valid_task = Task(
        name="Valid Task",
        priority=TaskPriority.MEDIUM,
        estimated_duration=120
    )
    
    result = validator.validate_task(valid_task)
    assert result.is_valid()
    
    print("✓ Valid task validation passed")
    
    # Test invalid task
    invalid_task = Task(name="", estimated_duration=-10)
    
    result = validator.validate_task(invalid_task)
    assert not result.is_valid()
    assert len(result.errors) > 0
    
    print("✓ Invalid task validation caught errors")
    
    return True


def test_error_handling():
    """Test error handling system."""
    print("Testing error handling...")
    
    from quantum_planner.utils.error_handling import ErrorHandler, QuantumPlannerError
    
    error_handler = ErrorHandler()
    
    # Test basic error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        success = error_handler.handle_error(e, {"test_context": "validation"})
    
    # Check statistics
    stats = error_handler.get_error_statistics()
    assert stats['total_errors'] > 0
    
    print("✓ Error handling system functional")
    
    # Test custom errors
    try:
        raise QuantumPlannerError("Custom quantum error")
    except Exception as e:
        error_handler.handle_error(e)
    
    recent_errors = error_handler.get_recent_errors(5)
    assert len(recent_errors) > 0
    
    print("✓ Custom error types work")
    
    return True


def test_logging_config():
    """Test logging configuration."""
    print("Testing logging configuration...")
    
    from quantum_planner.utils.logging_config import setup_logging, LogContext
    import logging
    
    # Setup logging
    config = setup_logging(log_level="INFO")
    assert config is not None
    
    # Test context logging
    with LogContext(operation="test", task_id="test-123"):
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
    
    print("✓ Logging system configured")
    
    return True


def test_integration():
    """Test basic integration without heavy dependencies."""
    print("Testing basic integration...")
    
    from quantum_planner.core.task import Task, TaskPriority
    from quantum_planner.utils.validation import TaskValidator
    from quantum_planner.utils.error_handling import ErrorHandler
    
    # Create test scenario
    tasks = [
        Task(name="Preparation", priority=TaskPriority.HIGH, estimated_duration=60),
        Task(name="Processing", priority=TaskPriority.CRITICAL, estimated_duration=120),
        Task(name="Cleanup", priority=TaskPriority.LOW, estimated_duration=30)
    ]
    
    # Add dependency
    tasks[1].add_dependency(tasks[0].id)
    tasks[2].add_dependency(tasks[1].id)
    
    # Validate
    validator = TaskValidator()
    result = validator.validate_task_list(tasks)
    
    if not result.is_valid():
        print(f"Validation errors: {result.errors}")
        return False
    
    print("✓ Integration test passed")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("QUANTUM TASK PLANNER - SIMPLIFIED TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_validation,
        test_error_handling, 
        test_logging_config,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            if test():
                print(f"✅ {test.__name__} PASSED")
                passed += 1
            else:
                print(f"❌ {test.__name__} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("💥 SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)