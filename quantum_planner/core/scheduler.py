"""Quantum-inspired task scheduler with advanced optimization algorithms."""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .task import Task, TaskStatus, TaskPriority
from .quantum_engine import QuantumEngine
from ..algorithms.quantum_annealing import QuantumAnnealingOptimizer
from ..algorithms.superposition import SuperpositionScheduler
from ..utils.performance import PerformanceMonitor

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


class QuantumInspiredScheduler:
    """Advanced quantum-inspired task scheduler."""
    
    def __init__(self, 
                 max_parallel_tasks: int = 10,
                 optimization_method: str = "quantum_annealing",
                 enable_superposition: bool = True,
                 enable_entanglement: bool = True,
                 performance_monitoring: bool = True):
        """Initialize the quantum scheduler."""
        self.max_parallel_tasks = max_parallel_tasks
        self.optimization_method = optimization_method
        self.enable_superposition = enable_superposition
        self.enable_entanglement = enable_entanglement
        
        # Core components
        self.quantum_engine = QuantumEngine()
        self.quantum_annealing = QuantumAnnealingOptimizer()
        self.superposition_scheduler = SuperpositionScheduler()
        
        # State management
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self.active_schedules: Dict[str, SchedulingResult] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance monitoring
        if performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None
            
        logger.info(f"Initialized QuantumInspiredScheduler with {optimization_method} optimization")
    
    def add_task(self, task: Task) -> None:
        """Add task to scheduler."""
        with self._lock:
            if not isinstance(task, Task):
                raise ValueError("Must provide Task object")
                
            self.tasks[task.id] = task
            logger.debug(f"Added task: {task.name} ({task.id})")
            
            # Create quantum entanglements if enabled
            if self.enable_entanglement:
                self._create_task_entanglements(task)
    
    def add_resource(self, resource: Resource) -> None:
        """Add resource to scheduler."""
        with self._lock:
            if not isinstance(resource, Resource):
                raise ValueError("Must provide Resource object")
                
            self.resources[resource.id] = resource
            logger.debug(f"Added resource: {resource.name} ({resource.id})")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from scheduler."""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Remove from dependencies of other tasks
                for other_task in self.tasks.values():
                    other_task.dependencies.discard(task_id)
                    other_task.dependents.discard(task_id)
                    other_task.conflicts.discard(task_id)
                    other_task.entanglement_partners.discard(task_id)
                
                del self.tasks[task_id]
                logger.debug(f"Removed task: {task_id}")
                return True
            
            return False
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        with self._lock:
            if task_id in self.tasks:
                old_status = self.tasks[task_id].status
                self.tasks[task_id].status = status
                
                # Handle status-specific logic
                if status == TaskStatus.COMPLETED:
                    self._handle_task_completion(task_id)
                elif status == TaskStatus.IN_PROGRESS:
                    self.tasks[task_id].start_execution()
                
                logger.debug(f"Updated task {task_id} status: {old_status} -> {status}")
                return True
            
            return False
    
    def create_optimal_schedule(self, 
                              optimization_time_limit: int = 60,
                              use_parallel_processing: bool = True) -> SchedulingResult:
        """Create optimal schedule using quantum-inspired algorithms."""
        start_time = datetime.now()
        
        with self._lock:
            try:
                # Validate inputs
                if not self.tasks:
                    return SchedulingResult(
                        schedule={}, total_completion_time=0,
                        resource_utilization={}, optimization_metrics={},
                        quantum_metrics={}, conflicts_resolved=0,
                        dependencies_satisfied=0, success=False,
                        error_message="No tasks to schedule"
                    )
                
                # Performance monitoring
                if self.performance_monitor:
                    self.performance_monitor.start_operation("create_optimal_schedule")
                
                # Prepare tasks and resources
                ready_tasks = [task for task in self.tasks.values() 
                             if task.status in [TaskStatus.PENDING]]
                available_resources = {rid: r for rid, r in self.resources.items() 
                                     if r.available_capacity > 0}
                
                logger.info(f"Scheduling {len(ready_tasks)} tasks with {len(available_resources)} resources")
                
                # Apply quantum superposition if enabled
                if self.enable_superposition:
                    self._apply_quantum_superposition(ready_tasks)
                
                # Choose optimization method
                if self.optimization_method == "quantum_annealing":
                    schedule = self._optimize_with_quantum_annealing(ready_tasks, available_resources)
                elif self.optimization_method == "superposition":
                    schedule = self._optimize_with_superposition(ready_tasks, available_resources)
                else:
                    schedule = self._optimize_with_hybrid_approach(ready_tasks, available_resources)
                
                # Calculate metrics
                result = self._calculate_scheduling_metrics(schedule, ready_tasks, available_resources)
                
                # Store active schedule
                schedule_id = f"schedule_{int(datetime.now().timestamp())}"
                self.active_schedules[schedule_id] = result
                
                # Performance monitoring
                if self.performance_monitor:
                    self.performance_monitor.end_operation("create_optimal_schedule")
                    result.optimization_metrics.update(self.performance_monitor.get_metrics())
                
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Schedule created in {execution_time:.2f}s with {result.total_completion_time}min completion time")
                
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
    
    def execute_schedule(self, schedule_result: SchedulingResult,
                        simulate: bool = False) -> Dict[str, Any]:
        """Execute or simulate the optimal schedule."""
        if not schedule_result.success:
            return {"success": False, "error": "Invalid schedule"}
        
        execution_log = []
        current_time = datetime.now()
        
        try:
            # Sort tasks by scheduled start time
            scheduled_tasks = []
            for task_id, task_info in schedule_result.schedule.items():
                if task_id in self.tasks:
                    scheduled_tasks.append((task_info['start_time'], task_id, task_info))
            
            scheduled_tasks.sort(key=lambda x: x[0])
            
            if simulate:
                logger.info("Simulating schedule execution")
                return self._simulate_schedule_execution(scheduled_tasks, current_time)
            else:
                logger.info("Executing schedule")
                return self._execute_schedule_real(scheduled_tasks, current_time)
                
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
                
                # Count active tasks on this resource
                active_tasks = []
                for task in self.tasks.values():
                    if task.status == TaskStatus.IN_PROGRESS:
                        # This is simplified - in real implementation would track resource assignments
                        active_tasks.append(task.id)
                
                utilization[resource_id] = {
                    'name': resource.name,
                    'capacity': resource.capacity,
                    'used_capacity': used_capacity,
                    'available_capacity': resource.available_capacity,
                    'utilization_percentage': utilization_percentage,
                    'active_tasks': active_tasks[:resource.capacity],  # Limit to capacity
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
                
                # Add entanglement edges
                for entangled_id in task.entanglement_partners:
                    edge = {
                        'source': task.id,
                        'target': entangled_id,
                        'type': 'entanglement'
                    }
                    graph['edges'].append(edge)
        
        return graph
    
    def optimize_for_multiple_objectives(self, 
                                       objectives: Dict[str, float],
                                       weights: Optional[Dict[str, float]] = None) -> SchedulingResult:
        """Optimize schedule for multiple objectives."""
        if weights is None:
            weights = {obj: 1.0 for obj in objectives.keys()}
        
        # Supported objectives
        supported = ['completion_time', 'resource_utilization', 'cost', 'priority_satisfaction']
        for obj in objectives:
            if obj not in supported:
                logger.warning(f"Unsupported objective: {obj}")
        
        # Multi-objective optimization using quantum approach
        logger.info(f"Optimizing for objectives: {list(objectives.keys())}")
        
        # Use weighted sum approach with quantum annealing
        best_result = None
        best_score = float('inf')
        
        for iteration in range(10):  # Multiple optimization runs
            result = self.create_optimal_schedule()
            if not result.success:
                continue
            
            # Calculate multi-objective score
            score = 0.0
            
            if 'completion_time' in objectives:
                target_time = objectives['completion_time']
                weight = weights.get('completion_time', 1.0)
                score += weight * abs(result.total_completion_time - target_time)
            
            if 'resource_utilization' in objectives:
                target_util = objectives['resource_utilization']
                weight = weights.get('resource_utilization', 1.0)
                avg_util = np.mean(list(result.resource_utilization.values()))
                score += weight * abs(avg_util - target_util)
            
            if 'priority_satisfaction' in objectives:
                weight = weights.get('priority_satisfaction', 1.0)
                # Calculate priority satisfaction metric
                priority_score = self._calculate_priority_satisfaction()
                target_score = objectives['priority_satisfaction']
                score += weight * abs(priority_score - target_score)
            
            if score < best_score:
                best_score = score
                best_result = result
        
        if best_result:
            best_result.optimization_metrics['multi_objective_score'] = best_score
            logger.info(f"Multi-objective optimization completed. Score: {best_score:.4f}")
        
        return best_result or SchedulingResult(
            schedule={}, total_completion_time=0, resource_utilization={},
            optimization_metrics={}, quantum_metrics={}, conflicts_resolved=0,
            dependencies_satisfied=0, success=False,
            error_message="Multi-objective optimization failed"
        )
    
    def _create_task_entanglements(self, task: Task) -> None:
        """Create quantum entanglements based on task relationships."""
        # Entangle with dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                task.entangle_with(dep_id)
                self.tasks[dep_id].entangle_with(task.id)
                self.quantum_engine.create_task_entanglement(task.id, dep_id, strength=0.8)
        
        # Entangle with similar priority tasks
        for other_task in self.tasks.values():
            if (other_task.id != task.id and 
                other_task.priority == task.priority and
                len(task.entanglement_partners) < 3):  # Limit entanglements
                task.entangle_with(other_task.id)
                other_task.entangle_with(task.id)
                self.quantum_engine.create_task_entanglement(task.id, other_task.id, strength=0.3)
    
    def _apply_quantum_superposition(self, tasks: List[Task]) -> None:
        """Apply quantum superposition to explore multiple execution paths."""
        for task in tasks:
            if len(task.superposition_states) <= 1:
                # Create superposition states for uncertain tasks
                superposition_states = self.quantum_engine.apply_quantum_superposition(task.id)
                
                for state in superposition_states:
                    scenario = state['scenario']
                    task.add_superposition_state(
                        probability=state['probability'],
                        duration=int(task.estimated_duration * scenario['duration_factor']),
                        resources={k: int(v * scenario['resource_factor']) 
                                 for k, v in task.required_resources.items()},
                        outcome=f"success_{scenario['success_probability']}"
                    )
    
    def _optimize_with_quantum_annealing(self, tasks: List[Task], 
                                        resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Optimize using quantum annealing algorithm."""
        resource_dict = {rid: r.capacity for rid, r in resources.items()}
        
        # Use quantum annealing optimizer
        assignment = self.quantum_annealing.optimize(tasks, resource_dict)
        
        return self._convert_assignment_to_schedule(assignment, tasks, resources)
    
    def _optimize_with_superposition(self, tasks: List[Task],
                                   resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Optimize using superposition scheduler."""
        return self.superposition_scheduler.create_schedule(tasks, resources)
    
    def _optimize_with_hybrid_approach(self, tasks: List[Task],
                                     resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Optimize using hybrid quantum-classical approach."""
        # Combine quantum annealing with superposition
        resource_dict = {rid: r.capacity for rid, r in resources.items()}
        
        # Phase 1: Quantum annealing for resource assignment
        qa_assignment = self.quantum_annealing.optimize(tasks, resource_dict)
        
        # Phase 2: Superposition for time optimization
        schedule = self.superposition_scheduler.create_schedule(tasks, resources)
        
        # Merge results
        for task_id in qa_assignment:
            if task_id in schedule:
                schedule[task_id]['resource_assignment'] = qa_assignment[task_id]
        
        return schedule
    
    def _convert_assignment_to_schedule(self, assignment: Dict[str, str],
                                      tasks: List[Task], resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Convert resource assignment to full schedule."""
        schedule = {}
        current_time = 0
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            resource_id = assignment.get(task.id, list(resources.keys())[0])
            
            schedule[task.id] = {
                'resource_id': resource_id,
                'start_time': current_time,
                'end_time': current_time + task.estimated_duration,
                'duration': task.estimated_duration,
                'priority': task.priority.name,
                'dependencies': list(task.dependencies)
            }
            
            current_time += task.estimated_duration
        
        return schedule
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks topologically based on dependencies."""
        visited = set()
        result = []
        
        def dfs(task):
            if task.id in visited:
                return
            visited.add(task.id)
            
            # Visit dependencies first
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task:
                    dfs(dep_task)
            
            result.append(task)
        
        # Sort by priority first, then apply DFS
        priority_sorted = sorted(tasks, key=lambda t: t.priority.weight, reverse=True)
        
        for task in priority_sorted:
            dfs(task)
        
        return result
    
    def _calculate_scheduling_metrics(self, schedule: Dict[str, Dict[str, Any]],
                                    tasks: List[Task], resources: Dict[str, Resource]) -> SchedulingResult:
        """Calculate comprehensive scheduling metrics."""
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
        for resource_id, resource in resources.items():
            assigned_tasks = [info for info in schedule.values() 
                            if info.get('resource_id') == resource_id]
            total_time = sum(info['duration'] for info in assigned_tasks)
            utilization = (total_time / max_end_time) if max_end_time > 0 else 0
            resource_utilization[resource_id] = utilization
        
        # Count satisfied dependencies
        dependencies_satisfied = 0
        for task in tasks:
            for dep_id in task.dependencies:
                if (task.id in schedule and dep_id in schedule and
                    schedule[dep_id]['end_time'] <= schedule[task.id]['start_time']):
                    dependencies_satisfied += 1
        
        # Count resolved conflicts
        conflicts_resolved = 0
        for task in tasks:
            for conflict_id in task.conflicts:
                if (task.id in schedule and conflict_id in schedule):
                    task_info = schedule[task.id]
                    conflict_info = schedule[conflict_id]
                    # Check if tasks don't overlap
                    if (task_info['end_time'] <= conflict_info['start_time'] or
                        conflict_info['end_time'] <= task_info['start_time']):
                        conflicts_resolved += 1
        
        # Get quantum metrics
        quantum_metrics = self.quantum_engine.get_quantum_metrics()
        
        return SchedulingResult(
            schedule=schedule,
            total_completion_time=max_end_time,
            resource_utilization=resource_utilization,
            optimization_metrics={
                'tasks_scheduled': len(schedule),
                'average_utilization': np.mean(list(resource_utilization.values())),
                'scheduling_efficiency': len(schedule) / len(tasks) if tasks else 0
            },
            quantum_metrics=quantum_metrics,
            conflicts_resolved=conflicts_resolved,
            dependencies_satisfied=dependencies_satisfied,
            success=True
        )
    
    def _handle_task_completion(self, task_id: str) -> None:
        """Handle task completion logic."""
        completed_task = self.tasks[task_id]
        completed_task.complete_execution()
        
        # Remove completed task from dependencies of other tasks
        for task in self.tasks.values():
            if task_id in task.dependencies:
                task.remove_dependency(task_id)
                logger.debug(f"Removed dependency {task_id} from task {task.id}")
    
    def _calculate_priority_satisfaction(self) -> float:
        """Calculate priority satisfaction metric."""
        if not self.tasks:
            return 0.0
        
        total_weight = 0
        satisfied_weight = 0
        
        for task in self.tasks.values():
            weight = task.priority.weight
            total_weight += weight
            
            if task.status == TaskStatus.COMPLETED:
                satisfied_weight += weight
            elif task.status == TaskStatus.IN_PROGRESS:
                satisfied_weight += weight * 0.5
        
        return satisfied_weight / total_weight if total_weight > 0 else 0.0
    
    def _simulate_schedule_execution(self, scheduled_tasks: List[Tuple], 
                                   start_time: datetime) -> Dict[str, Any]:
        """Simulate schedule execution."""
        simulation_log = []
        current_time = start_time
        
        for start_time_offset, task_id, task_info in scheduled_tasks:
            task = self.tasks[task_id]
            execution_start = current_time + timedelta(minutes=start_time_offset)
            execution_end = execution_start + timedelta(minutes=task_info['duration'])
            
            simulation_log.append({
                'task_id': task_id,
                'task_name': task.name,
                'simulated_start': execution_start.isoformat(),
                'simulated_end': execution_end.isoformat(),
                'resource_id': task_info.get('resource_id', 'default'),
                'duration': task_info['duration']
            })
        
        return {
            'success': True,
            'simulation': True,
            'execution_log': simulation_log,
            'total_simulated_time': len(scheduled_tasks) * 60  # Simplified
        }
    
    def _execute_schedule_real(self, scheduled_tasks: List[Tuple],
                             start_time: datetime) -> Dict[str, Any]:
        """Execute schedule in reality (placeholder for actual execution)."""
        execution_log = []
        
        # This would contain real execution logic
        # For now, just mark tasks as in progress
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