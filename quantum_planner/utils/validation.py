"""Comprehensive validation utilities for quantum task planner."""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import uuid

from ..core.task import Task, TaskPriority, TaskStatus

# Try to import from main scheduler, fall back to simple scheduler
try:
    from ..core.scheduler import Resource, SchedulingResult
except ImportError:
    from ..core.simple_scheduler import Resource, SchedulingResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    errors: List[str]
    warnings: List[str] 
    info: List[str]
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add validation info."""
        self.info.append(message)
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.valid and len(self.errors) == 0
    
    def get_summary(self) -> str:
        """Get validation summary."""
        status = "VALID" if self.is_valid() else "INVALID"
        return f"Validation {status}: {len(self.errors)} errors, {len(self.warnings)} warnings"


class TaskValidator:
    """Comprehensive task validation."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize task validator."""
        self.strict_mode = strict_mode
        logger.debug(f"Initialized TaskValidator (strict_mode={strict_mode})")
    
    def validate_task(self, task: Task) -> ValidationResult:
        """Validate a single task."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        try:
            # Basic validation
            self._validate_basic_properties(task, result)
            
            # ID validation
            self._validate_task_id(task, result)
            
            # Name validation
            self._validate_task_name(task, result)
            
            # Duration validation
            self._validate_duration(task, result)
            
            # Date validation
            self._validate_dates(task, result)
            
            # Priority validation
            self._validate_priority(task, result)
            
            # Status validation
            self._validate_status(task, result)
            
            # Resource validation
            self._validate_resources(task, result)
            
            # Dependency validation
            self._validate_dependencies(task, result)
            
            # Conflict validation
            self._validate_conflicts(task, result)
            
            # Quantum properties validation
            self._validate_quantum_properties(task, result)
            
            # Cross-validation
            self._validate_consistency(task, result)
            
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            logger.error(f"Task validation failed for {task.id}: {e}")
        
        return result
    
    def validate_task_list(self, tasks: List[Task]) -> ValidationResult:
        """Validate a list of tasks and their relationships."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        if not tasks:
            result.add_warning("Empty task list")
            return result
        
        try:
            # Validate individual tasks
            for task in tasks:
                task_result = self.validate_task(task)
                result.errors.extend(task_result.errors)
                result.warnings.extend(task_result.warnings)
                result.info.extend(task_result.info)
                
                if not task_result.is_valid():
                    result.valid = False
            
            # Validate task relationships
            self._validate_task_relationships(tasks, result)
            
            # Validate dependency cycles
            self._validate_dependency_cycles(tasks, result)
            
            # Validate resource conflicts
            self._validate_global_resource_conflicts(tasks, result)
            
            # Validate scheduling feasibility
            self._validate_scheduling_feasibility(tasks, result)
            
        except Exception as e:
            result.add_error(f"Task list validation error: {str(e)}")
            logger.error(f"Task list validation failed: {e}")
        
        return result
    
    def _validate_basic_properties(self, task: Task, result: ValidationResult):
        """Validate basic task properties."""
        if not isinstance(task, Task):
            result.add_error("Invalid task type")
            return
        
        if not hasattr(task, 'id') or not task.id:
            result.add_error("Task ID is required")
        
        if not hasattr(task, 'name'):
            result.add_error("Task name property is required")
        
        if not hasattr(task, 'estimated_duration'):
            result.add_error("Task estimated_duration property is required")
        
        if not hasattr(task, 'priority'):
            result.add_error("Task priority property is required")
        
        if not hasattr(task, 'status'):
            result.add_error("Task status property is required")
    
    def _validate_task_id(self, task: Task, result: ValidationResult):
        """Validate task ID format and uniqueness."""
        if not task.id:
            result.add_error("Task ID cannot be empty")
            return
        
        if not isinstance(task.id, str):
            result.add_error("Task ID must be a string")
            return
        
        # Check for valid UUID format (if generated automatically)
        try:
            uuid.UUID(task.id)
            result.add_info("Task ID is valid UUID format")
        except ValueError:
            # Not a UUID, check for valid identifier
            if not re.match(r'^[a-zA-Z0-9_-]+$', task.id):
                if self.strict_mode:
                    result.add_error("Task ID contains invalid characters")
                else:
                    result.add_warning("Task ID should contain only alphanumeric characters, underscores, and hyphens")
        
        if len(task.id) > 255:
            result.add_error("Task ID too long (max 255 characters)")
        
        if len(task.id) < 1:
            result.add_error("Task ID too short")
    
    def _validate_task_name(self, task: Task, result: ValidationResult):
        """Validate task name."""
        if not task.name:
            if self.strict_mode:
                result.add_error("Task name is required in strict mode")
            else:
                result.add_warning("Task name is empty")
            return
        
        if not isinstance(task.name, str):
            result.add_error("Task name must be a string")
            return
        
        if len(task.name) > 500:
            result.add_error("Task name too long (max 500 characters)")
        
        if len(task.name.strip()) == 0:
            result.add_warning("Task name is only whitespace")
        
        # Check for potentially problematic characters
        if any(char in task.name for char in ['\n', '\r', '\t']):
            result.add_warning("Task name contains whitespace characters")
    
    def _validate_duration(self, task: Task, result: ValidationResult):
        """Validate task duration."""
        if not isinstance(task.estimated_duration, (int, float)):
            result.add_error("Estimated duration must be a number")
            return
        
        if task.estimated_duration <= 0:
            result.add_error("Estimated duration must be positive")
        
        if task.estimated_duration > 10080:  # 1 week in minutes
            result.add_warning("Task duration exceeds 1 week")
        
        if task.estimated_duration < 1:
            result.add_warning("Task duration is less than 1 minute")
        
        # Check actual duration if available
        if task.actual_duration is not None:
            if not isinstance(task.actual_duration, (int, float)):
                result.add_error("Actual duration must be a number")
            elif task.actual_duration < 0:
                result.add_error("Actual duration cannot be negative")
    
    def _validate_dates(self, task: Task, result: ValidationResult):
        """Validate task dates."""
        now = datetime.now()
        
        # Validate deadline
        if task.deadline:
            if not isinstance(task.deadline, datetime):
                result.add_error("Deadline must be a datetime object")
            elif task.deadline < now:
                result.add_warning("Deadline is in the past")
        
        # Validate earliest start
        if task.earliest_start:
            if not isinstance(task.earliest_start, datetime):
                result.add_error("Earliest start must be a datetime object")
        
        # Validate date consistency
        if task.deadline and task.earliest_start:
            if task.earliest_start > task.deadline:
                result.add_error("Earliest start is after deadline")
            
            time_available = (task.deadline - task.earliest_start).total_seconds() / 60
            if time_available < task.estimated_duration:
                result.add_error("Not enough time between earliest start and deadline")
        
        # Validate execution dates
        if task.started_at and not isinstance(task.started_at, datetime):
            result.add_error("Started at must be a datetime object")
        
        if task.completed_at and not isinstance(task.completed_at, datetime):
            result.add_error("Completed at must be a datetime object")
        
        if task.started_at and task.completed_at:
            if task.started_at > task.completed_at:
                result.add_error("Started at is after completed at")
    
    def _validate_priority(self, task: Task, result: ValidationResult):
        """Validate task priority."""
        if not isinstance(task.priority, TaskPriority):
            result.add_error("Task priority must be a TaskPriority enum")
            return
        
        if task.quantum_weight < 0 or task.quantum_weight > 2:
            result.add_warning("Quantum weight should be between 0 and 2")
    
    def _validate_status(self, task: Task, result: ValidationResult):
        """Validate task status."""
        if not isinstance(task.status, TaskStatus):
            result.add_error("Task status must be a TaskStatus enum")
            return
        
        # Validate status transitions
        if task.status == TaskStatus.IN_PROGRESS:
            if not task.started_at:
                result.add_warning("Task marked as in progress but no start time")
        
        if task.status == TaskStatus.COMPLETED:
            if not task.completed_at:
                result.add_warning("Task marked as completed but no completion time")
            if not task.started_at:
                result.add_warning("Task marked as completed but no start time")
        
        if task.status == TaskStatus.BLOCKED:
            if 'block_reason' not in task.attributes:
                result.add_warning("Blocked task should have block reason in attributes")
    
    def _validate_resources(self, task: Task, result: ValidationResult):
        """Validate task resource requirements."""
        if not isinstance(task.required_resources, dict):
            result.add_error("Required resources must be a dictionary")
            return
        
        if not isinstance(task.preferred_resources, dict):
            result.add_error("Preferred resources must be a dictionary")
            return
        
        # Validate resource requirements
        for resource_name, amount in task.required_resources.items():
            if not isinstance(resource_name, str):
                result.add_error(f"Resource name must be string: {resource_name}")
            
            if not isinstance(amount, (int, float)):
                result.add_error(f"Resource amount must be number: {amount}")
            elif amount <= 0:
                result.add_error(f"Resource amount must be positive: {amount}")
        
        # Validate preferred resources
        for resource_name, amount in task.preferred_resources.items():
            if not isinstance(resource_name, str):
                result.add_error(f"Preferred resource name must be string: {resource_name}")
            
            if not isinstance(amount, (int, float)):
                result.add_error(f"Preferred resource amount must be number: {amount}")
            elif amount <= 0:
                result.add_error(f"Preferred resource amount must be positive: {amount}")
    
    def _validate_dependencies(self, task: Task, result: ValidationResult):
        """Validate task dependencies."""
        if not isinstance(task.dependencies, set):
            result.add_error("Dependencies must be a set")
            return
        
        if not isinstance(task.dependents, set):
            result.add_error("Dependents must be a set")
            return
        
        # Check for self-dependency
        if task.id in task.dependencies:
            result.add_error("Task cannot depend on itself")
        
        if task.id in task.dependents:
            result.add_error("Task cannot be its own dependent")
        
        # Validate dependency IDs
        for dep_id in task.dependencies:
            if not isinstance(dep_id, str):
                result.add_error(f"Dependency ID must be string: {dep_id}")
            elif not dep_id.strip():
                result.add_error("Dependency ID cannot be empty")
    
    def _validate_conflicts(self, task: Task, result: ValidationResult):
        """Validate task conflicts."""
        if not isinstance(task.conflicts, set):
            result.add_error("Conflicts must be a set")
            return
        
        # Check for self-conflict
        if task.id in task.conflicts:
            result.add_error("Task cannot conflict with itself")
        
        # Validate conflict IDs
        for conflict_id in task.conflicts:
            if not isinstance(conflict_id, str):
                result.add_error(f"Conflict ID must be string: {conflict_id}")
            elif not conflict_id.strip():
                result.add_error("Conflict ID cannot be empty")
    
    def _validate_quantum_properties(self, task: Task, result: ValidationResult):
        """Validate quantum properties."""
        if not isinstance(task.superposition_states, list):
            result.add_error("Superposition states must be a list")
            return
        
        if not isinstance(task.entanglement_partners, set):
            result.add_error("Entanglement partners must be a set")
            return
        
        # Validate superposition states
        total_probability = 0.0
        for i, state in enumerate(task.superposition_states):
            if not isinstance(state, dict):
                result.add_error(f"Superposition state {i} must be a dictionary")
                continue
            
            if 'probability' not in state:
                result.add_error(f"Superposition state {i} missing probability")
                continue
            
            prob = state['probability']
            if not isinstance(prob, (int, float)):
                result.add_error(f"Superposition state {i} probability must be number")
            elif prob < 0 or prob > 1:
                result.add_error(f"Superposition state {i} probability must be between 0 and 1")
            else:
                total_probability += prob
        
        if task.superposition_states and abs(total_probability - 1.0) > 0.01:
            result.add_warning("Superposition state probabilities don't sum to 1.0")
        
        # Validate entanglement partners
        if task.id in task.entanglement_partners:
            result.add_error("Task cannot be entangled with itself")
    
    def _validate_consistency(self, task: Task, result: ValidationResult):
        """Validate internal consistency."""
        # Check quantum weight consistency with priority
        expected_weight = task.priority.weight
        if abs(task.quantum_weight - expected_weight) > 0.1:
            result.add_warning("Quantum weight inconsistent with priority")
        
        # Check tags consistency
        if not isinstance(task.tags, set):
            result.add_error("Tags must be a set")
        else:
            for tag in task.tags:
                if not isinstance(tag, str):
                    result.add_error(f"Tag must be string: {tag}")
        
        # Check attributes consistency
        if not isinstance(task.attributes, dict):
            result.add_error("Attributes must be a dictionary")
    
    def _validate_task_relationships(self, tasks: List[Task], result: ValidationResult):
        """Validate relationships between tasks."""
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            # Check that dependencies exist
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    result.add_error(f"Task {task.id} depends on non-existent task {dep_id}")
            
            # Check that conflicts exist
            for conflict_id in task.conflicts:
                if conflict_id not in task_ids:
                    result.add_warning(f"Task {task.id} conflicts with non-existent task {conflict_id}")
            
            # Check that entanglement partners exist
            for entangled_id in task.entanglement_partners:
                if entangled_id not in task_ids:
                    result.add_warning(f"Task {task.id} entangled with non-existent task {entangled_id}")
    
    def _validate_dependency_cycles(self, tasks: List[Task], result: ValidationResult):
        """Validate that there are no dependency cycles."""
        task_dict = {task.id: task for task in tasks}
        
        def has_cycle(task_id: str, visited: Set[str], path: Set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            path.add(task_id)
            
            task = task_dict.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if has_cycle(dep_id, visited, path):
                        return True
            
            path.remove(task_id)
            return False
        
        visited = set()
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id, visited, set()):
                    result.add_error(f"Dependency cycle detected involving task {task.id}")
    
    def _validate_global_resource_conflicts(self, tasks: List[Task], result: ValidationResult):
        """Validate global resource conflicts."""
        # Check for impossible resource requirements
        all_resources = set()
        for task in tasks:
            all_resources.update(task.required_resources.keys())
            all_resources.update(task.preferred_resources.keys())
        
        # Check for tasks that require the same exclusive resource
        exclusive_resources = {}
        for task in tasks:
            for resource_name in task.required_resources:
                if resource_name not in exclusive_resources:
                    exclusive_resources[resource_name] = []
                exclusive_resources[resource_name].append(task.id)
        
        for resource_name, task_list in exclusive_resources.items():
            if len(task_list) > 1:
                # Check if any of these tasks are required to run simultaneously
                for i, task_id1 in enumerate(task_list):
                    task1 = task_dict.get(task_id1) if 'task_dict' in locals() else None
                    if not task1:
                        continue
                    
                    for task_id2 in task_list[i+1:]:
                        task2 = task_dict.get(task_id2) if 'task_dict' in locals() else None
                        if task2 and task_id2 not in task1.conflicts:
                            result.add_warning(f"Tasks {task_id1} and {task_id2} both require resource {resource_name} but are not marked as conflicts")
    
    def _validate_scheduling_feasibility(self, tasks: List[Task], result: ValidationResult):
        """Validate that the tasks can theoretically be scheduled."""
        # Calculate minimum possible completion time
        critical_path_time = self._calculate_critical_path(tasks)
        
        # Check for impossible deadlines
        now = datetime.now()
        for task in tasks:
            if task.deadline:
                time_until_deadline = (task.deadline - now).total_seconds() / 60
                if time_until_deadline < critical_path_time:
                    result.add_warning(f"Task {task.id} deadline may be unrealistic given dependencies")
    
    def _calculate_critical_path(self, tasks: List[Task]) -> float:
        """Calculate critical path duration."""
        # Simplified critical path calculation
        task_dict = {task.id: task for task in tasks}
        
        def get_path_duration(task_id: str, visited: Set[str]) -> float:
            if task_id in visited:
                return 0  # Avoid cycles
            
            task = task_dict.get(task_id)
            if not task:
                return 0
            
            visited.add(task_id)
            
            max_dep_duration = 0
            for dep_id in task.dependencies:
                dep_duration = get_path_duration(dep_id, visited.copy())
                max_dep_duration = max(max_dep_duration, dep_duration)
            
            return max_dep_duration + task.estimated_duration
        
        max_duration = 0
        for task in tasks:
            duration = get_path_duration(task.id, set())
            max_duration = max(max_duration, duration)
        
        return max_duration


class ResourceValidator:
    """Validate resource configurations."""
    
    def validate_resource(self, resource: Resource) -> ValidationResult:
        """Validate a single resource."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        # Basic validation
        if not isinstance(resource.id, str) or not resource.id.strip():
            result.add_error("Resource ID must be non-empty string")
        
        if not isinstance(resource.name, str) or not resource.name.strip():
            result.add_error("Resource name must be non-empty string")
        
        if not isinstance(resource.capacity, int) or resource.capacity < 0:
            result.add_error("Resource capacity must be non-negative integer")
        
        if not isinstance(resource.available_capacity, int):
            result.add_error("Resource available capacity must be integer")
        
        if resource.available_capacity > resource.capacity:
            result.add_error("Available capacity cannot exceed total capacity")
        
        if resource.available_capacity < 0:
            result.add_error("Available capacity cannot be negative")
        
        if not isinstance(resource.cost_per_minute, (int, float)) or resource.cost_per_minute < 0:
            result.add_error("Cost per minute must be non-negative number")
        
        return result
    
    def validate_resource_list(self, resources: List[Resource]) -> ValidationResult:
        """Validate list of resources."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        if not resources:
            result.add_warning("Empty resource list")
            return result
        
        # Check individual resources
        resource_ids = set()
        for resource in resources:
            res_result = self.validate_resource(resource)
            result.errors.extend(res_result.errors)
            result.warnings.extend(res_result.warnings)
            
            # Check for duplicate IDs
            if resource.id in resource_ids:
                result.add_error(f"Duplicate resource ID: {resource.id}")
            resource_ids.add(resource.id)
        
        return result


class ScheduleValidator:
    """Validate schedules and scheduling results."""
    
    def validate_schedule_result(self, schedule_result: SchedulingResult) -> ValidationResult:
        """Validate a scheduling result."""
        result = ValidationResult(valid=True, errors=[], warnings=[], info=[])
        
        if not schedule_result.success:
            result.add_warning(f"Schedule marked as unsuccessful: {schedule_result.error_message}")
        
        # Validate schedule structure
        if not isinstance(schedule_result.schedule, dict):
            result.add_error("Schedule must be a dictionary")
            return result
        
        # Validate individual task schedules
        for task_id, task_schedule in schedule_result.schedule.items():
            self._validate_task_schedule(task_id, task_schedule, result)
        
        # Validate metrics
        self._validate_metrics(schedule_result, result)
        
        # Validate resource utilization
        self._validate_resource_utilization(schedule_result, result)
        
        return result
    
    def _validate_task_schedule(self, task_id: str, task_schedule: Dict[str, Any], 
                               result: ValidationResult):
        """Validate individual task schedule."""
        required_fields = ['start_time', 'end_time', 'duration', 'resource_id']
        
        for field in required_fields:
            if field not in task_schedule:
                result.add_error(f"Task {task_id} schedule missing {field}")
        
        if 'start_time' in task_schedule and 'end_time' in task_schedule:
            start_time = task_schedule['start_time']
            end_time = task_schedule['end_time']
            
            if not isinstance(start_time, (int, float)):
                result.add_error(f"Task {task_id} start_time must be number")
            elif start_time < 0:
                result.add_error(f"Task {task_id} start_time cannot be negative")
            
            if not isinstance(end_time, (int, float)):
                result.add_error(f"Task {task_id} end_time must be number")
            elif end_time <= start_time:
                result.add_error(f"Task {task_id} end_time must be after start_time")
        
        if 'duration' in task_schedule:
            duration = task_schedule['duration']
            if not isinstance(duration, (int, float)):
                result.add_error(f"Task {task_id} duration must be number")
            elif duration <= 0:
                result.add_error(f"Task {task_id} duration must be positive")
    
    def _validate_metrics(self, schedule_result: SchedulingResult, result: ValidationResult):
        """Validate scheduling metrics."""
        if schedule_result.total_completion_time < 0:
            result.add_error("Total completion time cannot be negative")
        
        if not isinstance(schedule_result.resource_utilization, dict):
            result.add_error("Resource utilization must be a dictionary")
        
        for resource_id, utilization in schedule_result.resource_utilization.items():
            if not isinstance(utilization, (int, float)):
                result.add_error(f"Resource {resource_id} utilization must be number")
            elif utilization < 0 or utilization > 1:
                result.add_error(f"Resource {resource_id} utilization must be between 0 and 1")
    
    def _validate_resource_utilization(self, schedule_result: SchedulingResult, 
                                     result: ValidationResult):
        """Validate resource utilization values."""
        for resource_id, utilization in schedule_result.resource_utilization.items():
            if utilization > 1.0:
                result.add_warning(f"Resource {resource_id} over-utilized: {utilization:.2%}")
            elif utilization < 0.1:
                result.add_warning(f"Resource {resource_id} under-utilized: {utilization:.2%}")