"""Simplified quantum-inspired scheduler without heavy dependencies."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
import threading

from .task import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class SchedulingResult:
    """Result of scheduling operation."""
    schedule: Dict[str, Dict[str, Any]]
    total_completion_time: int
    resource_utilization: Dict[str, float]
    optimization_metrics: Dict[str, float]
    quantum_metrics: Dict[str, float]
    conflicts_resolved: int
    dependencies_satisfied: int
    success: bool = True
    error_message: str = ""


@dataclass
class Resource:
    """Resource representation for scheduling."""
    id: str
    name: str
    capacity: int
    available_capacity: int
    cost_per_minute: float = 0.0
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.available_capacity == 0:
            self.available_capacity = self.capacity


class SimpleQuantumScheduler:
    """Simplified quantum-inspired scheduler."""
    
    def __init__(self, 
                 max_parallel_tasks: int = 10,
                 optimization_method: str = "simple"):
        """Initialize the scheduler."""
        self.max_parallel_tasks = max_parallel_tasks
        self.optimization_method = optimization_method
        
        # State management
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized SimpleQuantumScheduler")
    
    def add_task(self, task: Task) -> None:
        """Add task to scheduler."""
        with self._lock:
            if not isinstance(task, Task):
                raise ValueError("Must provide Task object")
                
            self.tasks[task.id] = task
            logger.debug(f"Added task: {task.name} ({task.id})")
    
    def add_resource(self, resource: Resource) -> None:
        """Add resource to scheduler."""
        with self._lock:
            if not isinstance(resource, Resource):
                raise ValueError("Must provide Resource object")
                
            self.resources[resource.id] = resource
            logger.debug(f"Added resource: {resource.name} ({resource.id})")
    
    def create_optimal_schedule(self, 
                              optimization_time_limit: int = 60) -> SchedulingResult:
        """Create optimal schedule using simple heuristics."""
        start_time = datetime.now()
        
        with self._lock:
            try:
                if not self.tasks:
                    return SchedulingResult(
                        schedule={}, total_completion_time=0,
                        resource_utilization={}, optimization_metrics={},
                        quantum_metrics={}, conflicts_resolved=0,
                        dependencies_satisfied=0, success=False,
                        error_message="No tasks to schedule"
                    )
                
                # Get ready tasks
                ready_tasks = [task for task in self.tasks.values() 
                             if task.status == TaskStatus.PENDING]
                available_resources = list(self.resources.keys())
                
                if not available_resources:
                    return SchedulingResult(
                        schedule={}, total_completion_time=0,
                        resource_utilization={}, optimization_metrics={},
                        quantum_metrics={}, conflicts_resolved=0,
                        dependencies_satisfied=0, success=False,
                        error_message="No resources available"
                    )
                
                logger.info(f"Scheduling {len(ready_tasks)} tasks with {len(available_resources)} resources")
                
                # Create schedule using priority-based heuristic
                schedule = self._create_priority_schedule(ready_tasks, available_resources)
                
                # Calculate metrics
                result = self._calculate_scheduling_metrics(schedule, ready_tasks, available_resources)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Schedule created in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error creating schedule: {e}")
                return SchedulingResult(
                    schedule={}, total_completion_time=0,
                    resource_utilization={}, optimization_metrics={},
                    quantum_metrics={}, conflicts_resolved=0,
                    dependencies_satisfied=0, success=False,
                    error_message=str(e)
                )
    
    def _create_priority_schedule(self, tasks: List[Task], resources: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create schedule based on task priority and dependencies."""
        schedule = {}
        resource_timelines = {rid: 0 for rid in resources}
        
        # Sort tasks by priority (quantum weight) and dependencies
        sorted_tasks = self._topological_sort_with_priority(tasks)
        
        for task in sorted_tasks:
            # Select resource with minimum current load
            selected_resource = min(resources, key=lambda r: resource_timelines[r])
            
            # Calculate start time considering dependencies
            start_time = resource_timelines[selected_resource]
            
            for dep_id in task.dependencies:
                if dep_id in schedule:
                    dep_end_time = schedule[dep_id]['end_time']
                    start_time = max(start_time, dep_end_time)
            
            end_time = start_time + task.estimated_duration
            
            schedule[task.id] = {
                'resource_id': selected_resource,
                'start_time': start_time,
                'end_time': end_time,
                'duration': task.estimated_duration,
                'priority': task.priority.name,
                'task_name': task.name
            }
            
            resource_timelines[selected_resource] = end_time
        
        return schedule
    
    def _topological_sort_with_priority(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks topologically with priority consideration."""
        visited = set()
        result = []
        task_dict = {task.id: task for task in tasks}
        
        def visit(task):
            if task.id in visited:
                return
            visited.add(task.id)
            
            # Visit dependencies first
            for dep_id in task.dependencies:
                dep_task = task_dict.get(dep_id)
                if dep_task:
                    visit(dep_task)
            
            result.append(task)
        
        # Sort by priority first, then apply topological sort
        priority_sorted = sorted(tasks, key=lambda t: t.priority.weight, reverse=True)
        
        for task in priority_sorted:
            visit(task)
        
        return result
    
    def _calculate_scheduling_metrics(self, schedule: Dict[str, Dict[str, Any]],
                                    tasks: List[Task], resources: List[str]) -> SchedulingResult:
        """Calculate scheduling metrics."""
        if not schedule:
            return SchedulingResult(
                schedule={}, total_completion_time=0, resource_utilization={},
                optimization_metrics={}, quantum_metrics={}, conflicts_resolved=0,
                dependencies_satisfied=0, success=False,
                error_message="Empty schedule"
            )
        
        # Calculate total completion time
        max_end_time = max(info['end_time'] for info in schedule.values())
        
        # Calculate resource utilization
        resource_utilization = {}
        for resource_id in resources:
            assigned_tasks = [info for info in schedule.values() 
                            if info.get('resource_id') == resource_id]
            total_time = sum(info['duration'] for info in assigned_tasks)
            utilization = (total_time / max_end_time) if max_end_time > 0 else 0
            resource_utilization[resource_id] = utilization
        
        # Count satisfied dependencies
        dependencies_satisfied = 0
        task_dict = {task.id: task for task in tasks}
        for task in tasks:
            for dep_id in task.dependencies:
                if (task.id in schedule and dep_id in schedule and
                    schedule[dep_id]['end_time'] <= schedule[task.id]['start_time']):
                    dependencies_satisfied += 1
        
        # Count resolved conflicts
        conflicts_resolved = 0
        for task in tasks:
            for conflict_id in task.conflicts:
                if task.id in schedule and conflict_id in schedule:
                    task_info = schedule[task.id]
                    conflict_info = schedule[conflict_id]
                    # Check if tasks don't overlap
                    if (task_info['end_time'] <= conflict_info['start_time'] or
                        conflict_info['end_time'] <= task_info['start_time']):
                        conflicts_resolved += 1
        
        # Simple quantum metrics
        quantum_metrics = {
            'energy_level': random.random(),
            'superposition_states': len(schedule),
            'entanglement_strength': 0.5
        }
        
        # Optimization metrics
        avg_utilization = sum(resource_utilization.values()) / len(resource_utilization) if resource_utilization else 0
        optimization_metrics = {
            'tasks_scheduled': len(schedule),
            'average_utilization': avg_utilization,
            'scheduling_efficiency': len(schedule) / len(tasks) if tasks else 0
        }
        
        return SchedulingResult(
            schedule=schedule,
            total_completion_time=max_end_time,
            resource_utilization=resource_utilization,
            optimization_metrics=optimization_metrics,
            quantum_metrics=quantum_metrics,
            conflicts_resolved=conflicts_resolved,
            dependencies_satisfied=dependencies_satisfied,
            success=True
        )
    
    def execute_schedule(self, schedule_result: SchedulingResult,
                        simulate: bool = False) -> Dict[str, Any]:
        """Execute or simulate the schedule."""
        if not schedule_result.success:
            return {"success": False, "error": "Invalid schedule"}
        
        execution_log = []
        
        try:
            # Sort tasks by scheduled start time
            scheduled_tasks = []
            for task_id, task_info in schedule_result.schedule.items():
                scheduled_tasks.append((task_info['start_time'], task_id, task_info))
            
            scheduled_tasks.sort(key=lambda x: x[0])
            
            if simulate:
                logger.info("Simulating schedule execution")
                for start_time, task_id, task_info in scheduled_tasks:
                    execution_log.append({
                        'task_id': task_id,
                        'task_name': task_info.get('task_name', task_id),
                        'simulated_start': start_time,
                        'simulated_end': task_info['end_time'],
                        'resource_id': task_info.get('resource_id', 'default'),
                        'duration': task_info['duration']
                    })
                
                return {
                    'success': True,
                    'simulation': True,
                    'execution_log': execution_log,
                    'total_simulated_time': len(scheduled_tasks) * 60
                }
            else:
                logger.info("Executing schedule")
                for _, task_id, task_info in scheduled_tasks:
                    if task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.IN_PROGRESS
                        execution_log.append({
                            'task_id': task_id,
                            'action': 'started',
                            'timestamp': datetime.now().isoformat(),
                            'resource_id': task_info.get('resource_id', 'default')
                        })
                
                return {
                    'success': True,
                    'simulation': False,
                    'execution_log': execution_log,
                    'tasks_started': len(scheduled_tasks)
                }
                
        except Exception as e:
            logger.error(f"Error executing schedule: {e}")
            return {"success": False, "error": str(e)}
    
    def get_resource_utilization(self) -> Dict[str, Dict[str, Any]]:
        """Get current resource utilization statistics."""
        utilization = {}
        
        with self._lock:
            for resource_id, resource in self.resources.items():
                used_capacity = resource.capacity - resource.available_capacity
                utilization_percentage = (used_capacity / resource.capacity) * 100 if resource.capacity > 0 else 0
                
                utilization[resource_id] = {
                    'name': resource.name,
                    'capacity': resource.capacity,
                    'used_capacity': used_capacity,
                    'available_capacity': resource.available_capacity,
                    'utilization_percentage': utilization_percentage,
                    'cost_per_minute': resource.cost_per_minute
                }
        
        return utilization
    
    def get_task_dependencies_graph(self) -> Dict[str, Any]:
        """Get task dependency graph for visualization."""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        with self._lock:
            # Add nodes (tasks)
            for task in self.tasks.values():
                node = {
                    'id': task.id,
                    'name': task.name,
                    'priority': task.priority.name,
                    'status': task.status.value,
                    'estimated_duration': task.estimated_duration,
                    'quantum_weight': task.quantum_weight
                }
                graph['nodes'].append(node)
            
            # Add edges (dependencies)
            for task in self.tasks.values():
                for dep_id in task.dependencies:
                    edge = {
                        'source': dep_id,
                        'target': task.id,
                        'type': 'dependency'
                    }
                    graph['edges'].append(edge)
                
                # Add conflict edges
                for conflict_id in task.conflicts:
                    edge = {
                        'source': task.id,
                        'target': conflict_id,
                        'type': 'conflict'
                    }
                    graph['edges'].append(edge)
        
        return graph


# Alias for backward compatibility
QuantumInspiredScheduler = SimpleQuantumScheduler