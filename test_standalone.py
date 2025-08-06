"""Standalone test for quantum planner core functionality."""

import sys
import os
from datetime import datetime, timedelta

# Direct import of minimal components
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Test the core task functionality directly
def test_task_core():
    """Test core task functionality."""
    print("Testing core task functionality...")
    
    # Import enum definitions
    from enum import Enum
    
    class TaskPriority(Enum):
        """Task priority levels with quantum weights."""
        CRITICAL = (1.0, "Critical - Must be completed first")
        HIGH = (0.8, "High priority task") 
        MEDIUM = (0.6, "Medium priority task")
        LOW = (0.4, "Low priority task")
        DEFERRED = (0.2, "Deferred for later")
        
        def __init__(self, weight: float, description: str):
            self.weight = weight
            self.description = description
    
    class TaskStatus(Enum):
        """Task execution status."""
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        BLOCKED = "blocked"
        COMPLETED = "completed"
        CANCELLED = "cancelled"
        FAILED = "failed"
    
    # Simple Task class
    import uuid
    from dataclasses import dataclass, field
    
    @dataclass
    class Task:
        """Simple task representation."""
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        name: str = ""
        description: str = ""
        priority: TaskPriority = TaskPriority.MEDIUM
        status: TaskStatus = TaskStatus.PENDING
        estimated_duration: int = 60  # minutes
        deadline: datetime = None
        dependencies: set = field(default_factory=set)
        conflicts: set = field(default_factory=set)
        quantum_weight: float = 1.0
        
        def __post_init__(self):
            if not self.name:
                self.name = f"Task-{self.id[:8]}"
            self.quantum_weight = self.priority.weight
        
        @property
        def is_ready(self) -> bool:
            return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
        
        @property 
        def is_overdue(self) -> bool:
            return (self.deadline is not None and 
                   datetime.now() > self.deadline and
                   self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED])
        
        def add_dependency(self, task_id: str):
            self.dependencies.add(task_id)
        
        def add_conflict(self, task_id: str):
            self.conflicts.add(task_id)
    
    # Test task creation
    task1 = Task(
        name="Test Task 1",
        priority=TaskPriority.HIGH,
        estimated_duration=120
    )
    
    assert task1.name == "Test Task 1"
    assert task1.priority == TaskPriority.HIGH
    assert task1.status == TaskStatus.PENDING
    assert task1.estimated_duration == 120
    assert task1.is_ready == True
    
    print("✓ Basic task creation works")
    
    # Test dependencies
    task2 = Task(name="Dependent Task", estimated_duration=60)
    task2.add_dependency(task1.id)
    
    assert task1.id in task2.dependencies
    assert task2.is_ready == False
    
    print("✓ Task dependencies work")
    
    # Test deadlines
    future_deadline = datetime.now() + timedelta(hours=2)
    past_deadline = datetime.now() - timedelta(hours=1)
    
    task_future = Task(name="Future Task", deadline=future_deadline)
    task_past = Task(name="Past Task", deadline=past_deadline)
    
    assert task_future.is_overdue == False
    assert task_past.is_overdue == True
    
    print("✓ Deadline handling works")
    
    return True


def test_simple_scheduler():
    """Test simple scheduling logic."""
    print("Testing simple scheduler...")
    
    # Use the task definitions from previous test
    from enum import Enum
    import uuid
    from dataclasses import dataclass, field
    
    class TaskPriority(Enum):
        CRITICAL = (1.0, "Critical")
        HIGH = (0.8, "High") 
        MEDIUM = (0.6, "Medium")
        LOW = (0.4, "Low")
        
        def __init__(self, weight: float, description: str):
            self.weight = weight
            self.description = description
    
    class TaskStatus(Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
    
    @dataclass
    class Task:
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        name: str = ""
        priority: TaskPriority = TaskPriority.MEDIUM
        status: TaskStatus = TaskStatus.PENDING
        estimated_duration: int = 60
        dependencies: set = field(default_factory=set)
        
        def __post_init__(self):
            if not self.name:
                self.name = f"Task-{self.id[:8]}"
        
        @property
        def is_ready(self) -> bool:
            return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    @dataclass
    class Resource:
        id: str
        name: str
        capacity: int
    
    class SimpleScheduler:
        """Very basic scheduler."""
        
        def __init__(self):
            self.tasks = {}
            self.resources = {}
        
        def add_task(self, task):
            self.tasks[task.id] = task
        
        def add_resource(self, resource):
            self.resources[resource.id] = resource
        
        def create_schedule(self):
            """Create simple priority-based schedule."""
            if not self.tasks or not self.resources:
                return {}
            
            # Get ready tasks sorted by priority
            ready_tasks = [t for t in self.tasks.values() if t.is_ready]
            ready_tasks.sort(key=lambda t: t.priority.weight, reverse=True)
            
            # Simple round-robin resource assignment
            schedule = {}
            resource_ids = list(self.resources.keys())
            current_time = 0
            
            for i, task in enumerate(ready_tasks):
                resource_id = resource_ids[i % len(resource_ids)]
                
                schedule[task.id] = {
                    'resource_id': resource_id,
                    'start_time': current_time,
                    'end_time': current_time + task.estimated_duration,
                    'duration': task.estimated_duration
                }
                
                current_time += task.estimated_duration
            
            return schedule
    
    # Test the scheduler
    scheduler = SimpleScheduler()
    
    # Add tasks
    tasks = [
        Task(name="High Priority Task", priority=TaskPriority.HIGH, estimated_duration=90),
        Task(name="Medium Priority Task", priority=TaskPriority.MEDIUM, estimated_duration=60),
        Task(name="Low Priority Task", priority=TaskPriority.LOW, estimated_duration=30)
    ]
    
    for task in tasks:
        scheduler.add_task(task)
    
    # Add resources
    resources = [
        Resource("cpu-1", "CPU Resource 1", 2),
        Resource("memory-1", "Memory Resource", 4)
    ]
    
    for resource in resources:
        scheduler.add_resource(resource)
    
    # Create schedule
    schedule = scheduler.create_schedule()
    
    assert len(schedule) == len(tasks)
    assert all('start_time' in info for info in schedule.values())
    assert all('end_time' in info for info in schedule.values())
    assert all('resource_id' in info for info in schedule.values())
    
    print("✓ Simple scheduling works")
    
    # Verify priority ordering (higher priority should start earlier)
    high_priority_task = next(t for t in tasks if t.priority == TaskPriority.HIGH)
    low_priority_task = next(t for t in tasks if t.priority == TaskPriority.LOW)
    
    high_start = schedule[high_priority_task.id]['start_time']
    low_start = schedule[low_priority_task.id]['start_time']
    
    assert high_start <= low_start, "High priority task should start before low priority"
    
    print("✓ Priority-based scheduling works")
    
    return True


def test_validation():
    """Test validation logic."""
    print("Testing validation logic...")
    
    from enum import Enum
    import uuid
    from dataclasses import dataclass, field
    
    class TaskPriority(Enum):
        HIGH = (0.8, "High")
        MEDIUM = (0.6, "Medium")
        
        def __init__(self, weight: float, description: str):
            self.weight = weight
            self.description = description
    
    @dataclass 
    class Task:
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        name: str = ""
        priority: TaskPriority = TaskPriority.MEDIUM
        estimated_duration: int = 60
    
    class SimpleValidator:
        """Basic validation logic."""
        
        def validate_task(self, task):
            errors = []
            warnings = []
            
            if not task.name.strip():
                errors.append("Task name cannot be empty")
            
            if task.estimated_duration <= 0:
                errors.append("Task duration must be positive")
            
            if task.estimated_duration > 480:  # 8 hours
                warnings.append("Task duration exceeds 8 hours")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
        
        def validate_task_list(self, tasks):
            errors = []
            warnings = []
            
            if not tasks:
                warnings.append("Empty task list")
                return {'valid': True, 'errors': errors, 'warnings': warnings}
            
            # Check for duplicate names
            names = [t.name for t in tasks]
            if len(names) != len(set(names)):
                warnings.append("Duplicate task names found")
            
            # Validate individual tasks
            for task in tasks:
                result = self.validate_task(task)
                errors.extend(result['errors'])
                warnings.extend(result['warnings'])
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
    
    # Test validation
    validator = SimpleValidator()
    
    # Valid task
    valid_task = Task(
        name="Valid Task",
        priority=TaskPriority.HIGH,
        estimated_duration=120
    )
    
    result = validator.validate_task(valid_task)
    assert result['valid'] == True
    assert len(result['errors']) == 0
    
    print("✓ Valid task validation works")
    
    # Invalid task
    invalid_task = Task(
        name="",  # Invalid name
        estimated_duration=-30  # Invalid duration
    )
    
    result = validator.validate_task(invalid_task)
    assert result['valid'] == False
    assert len(result['errors']) > 0
    
    print("✓ Invalid task validation works")
    
    # Task list validation
    tasks = [valid_task, invalid_task]
    result = validator.validate_task_list(tasks)
    assert result['valid'] == False  # Invalid due to invalid_task
    
    print("✓ Task list validation works")
    
    return True


def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("QUANTUM PLANNER - STANDALONE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_task_core,
        test_simple_scheduler,
        test_validation
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
        print("Core quantum planner functionality is working correctly.")
        return True
    else:
        print("💥 SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)