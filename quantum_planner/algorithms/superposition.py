"""Superposition-based scheduling algorithm."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools

from ..core.task import Task, TaskPriority, TaskStatus
from ..core.scheduler import Resource

logger = logging.getLogger(__name__)


@dataclass
class SuperpositionState:
    """Represents a superposition state in task scheduling."""
    probability: float
    schedule: Dict[str, Dict[str, Any]]
    total_time: int
    resource_efficiency: float
    conflicts: int
    energy: float = 0.0


class SuperpositionScheduler:
    """Scheduler that uses quantum superposition principles."""
    
    def __init__(self, max_superposition_states: int = 64,
                 collapse_threshold: float = 0.01):
        """Initialize superposition scheduler."""
        self.max_states = max_superposition_states
        self.collapse_threshold = collapse_threshold
        self.superposition_states: List[SuperpositionState] = []
        
        logger.info(f"Initialized SuperpositionScheduler with {max_superposition_states} max states")
    
    def create_schedule(self, tasks: List[Task], 
                       resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Create schedule using quantum superposition."""
        if not tasks:
            return {}
        
        logger.info(f"Creating superposition schedule for {len(tasks)} tasks")
        
        # Initialize superposition of possible schedules
        self._initialize_superposition(tasks, resources)
        
        # Evolve superposition states
        self._evolve_superposition(tasks, resources)
        
        # Collapse superposition to best schedule
        best_schedule = self._collapse_superposition()
        
        logger.info(f"Superposition scheduling completed with {len(best_schedule)} scheduled tasks")
        return best_schedule
    
    def _initialize_superposition(self, tasks: List[Task], 
                                 resources: Dict[str, Resource]) -> None:
        """Initialize superposition of scheduling possibilities."""
        self.superposition_states.clear()
        
        # Generate initial set of random schedules
        num_initial_states = min(self.max_states, 32)
        
        for i in range(num_initial_states):
            schedule = self._generate_random_schedule(tasks, resources)
            
            state = SuperpositionState(
                probability=1.0 / num_initial_states,
                schedule=schedule,
                total_time=self._calculate_total_time(schedule),
                resource_efficiency=self._calculate_resource_efficiency(schedule, resources),
                conflicts=self._count_conflicts(schedule, tasks)
            )
            state.energy = self._calculate_state_energy(state, tasks, resources)
            
            self.superposition_states.append(state)
        
        logger.debug(f"Initialized {len(self.superposition_states)} superposition states")
    
    def _generate_random_schedule(self, tasks: List[Task], 
                                 resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Generate a random but valid schedule."""
        schedule = {}
        resource_timelines = {rid: 0 for rid in resources.keys()}
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_constraints(tasks)
        
        for task in sorted_tasks:
            # Select random resource with some bias toward requirements
            resource_candidates = list(resources.keys())
            
            # Bias selection toward resources that meet requirements
            weighted_candidates = []
            for resource_id in resource_candidates:
                weight = 1.0
                
                # Check resource requirements
                for req_resource, req_amount in task.required_resources.items():
                    if req_resource == resource_id:
                        resource = resources[resource_id]
                        if req_amount <= resource.capacity:
                            weight *= 2.0  # Prefer matching resources
                        else:
                            weight *= 0.1  # Penalize insufficient resources
                
                # Consider current resource load
                current_load = resource_timelines[resource_id]
                weight *= np.exp(-current_load / 1000)  # Prefer less loaded resources
                
                weighted_candidates.extend([resource_id] * max(1, int(weight * 10)))
            
            selected_resource = np.random.choice(weighted_candidates)
            
            # Schedule task on selected resource
            start_time = resource_timelines[selected_resource]
            
            # Check dependencies
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
                'priority': task.priority.name
            }
            
            resource_timelines[selected_resource] = end_time
        
        return schedule
    
    def _evolve_superposition(self, tasks: List[Task], 
                             resources: Dict[str, Resource]) -> None:
        """Evolve superposition states through quantum operations."""
        max_iterations = 100
        
        for iteration in range(max_iterations):
            # Apply quantum operations to superposition states
            self._apply_quantum_interference(tasks, resources)
            self._apply_amplitude_amplification()
            
            # Collapse low-probability states
            self._collapse_weak_states()
            
            # Generate new states through quantum mutations
            if len(self.superposition_states) < self.max_states // 2:
                self._generate_mutation_states(tasks, resources)
            
            # Check convergence
            if self._check_convergence():
                logger.debug(f"Superposition evolved to convergence at iteration {iteration}")
                break
        
        logger.debug(f"Superposition evolution completed after {iteration + 1} iterations")
    
    def _apply_quantum_interference(self, tasks: List[Task], 
                                   resources: Dict[str, Resource]) -> None:
        """Apply quantum interference between superposition states."""
        if len(self.superposition_states) < 2:
            return
        
        # Calculate interference between pairs of states
        new_states = []
        
        for i in range(len(self.superposition_states)):
            for j in range(i + 1, min(i + 3, len(self.superposition_states))):  # Limit pairs
                state1 = self.superposition_states[i]
                state2 = self.superposition_states[j]
                
                # Calculate interference coefficient
                interference = self._calculate_interference(state1, state2)
                
                if interference > 0.1:  # Constructive interference
                    # Create new state through constructive combination
                    new_schedule = self._combine_schedules(state1.schedule, state2.schedule, 
                                                         interference, tasks, resources)
                    
                    if new_schedule:
                        new_state = SuperpositionState(
                            probability=state1.probability * state2.probability * interference,
                            schedule=new_schedule,
                            total_time=self._calculate_total_time(new_schedule),
                            resource_efficiency=self._calculate_resource_efficiency(new_schedule, resources),
                            conflicts=self._count_conflicts(new_schedule, tasks)
                        )
                        new_state.energy = self._calculate_state_energy(new_state, tasks, resources)
                        new_states.append(new_state)
        
        # Add promising new states
        self.superposition_states.extend(new_states[:self.max_states // 4])
    
    def _apply_amplitude_amplification(self) -> None:
        """Apply amplitude amplification to boost good states."""
        if not self.superposition_states:
            return
        
        # Calculate quality scores
        scores = []
        for state in self.superposition_states:
            # Lower energy and fewer conflicts = higher quality
            quality = 1.0 / (1.0 + state.energy + state.conflicts)
            quality *= state.resource_efficiency
            scores.append(quality)
        
        if not scores:
            return
        
        # Amplify probabilities of high-quality states
        max_score = max(scores)
        for i, state in enumerate(self.superposition_states):
            if max_score > 0:
                amplification_factor = (scores[i] / max_score) ** 0.5
                state.probability *= amplification_factor
        
        # Renormalize probabilities
        total_prob = sum(state.probability for state in self.superposition_states)
        if total_prob > 0:
            for state in self.superposition_states:
                state.probability /= total_prob
    
    def _collapse_weak_states(self) -> None:
        """Remove states with very low probability."""
        self.superposition_states = [
            state for state in self.superposition_states 
            if state.probability >= self.collapse_threshold
        ]
        
        # Limit total number of states
        if len(self.superposition_states) > self.max_states:
            # Keep highest probability states
            self.superposition_states.sort(key=lambda s: s.probability, reverse=True)
            self.superposition_states = self.superposition_states[:self.max_states]
    
    def _generate_mutation_states(self, tasks: List[Task], 
                                 resources: Dict[str, Resource]) -> None:
        """Generate new states through quantum mutations."""
        if not self.superposition_states:
            return
        
        # Select parent states for mutation
        num_mutations = min(8, self.max_states - len(self.superposition_states))
        
        for _ in range(num_mutations):
            # Select parent state weighted by probability
            probabilities = [state.probability for state in self.superposition_states]
            if sum(probabilities) == 0:
                probabilities = [1.0] * len(self.superposition_states)
            
            parent_idx = np.random.choice(len(self.superposition_states), p=np.array(probabilities) / sum(probabilities))
            parent_state = self.superposition_states[parent_idx]
            
            # Create mutation
            mutated_schedule = self._mutate_schedule(parent_state.schedule, tasks, resources)
            
            if mutated_schedule:
                mutated_state = SuperpositionState(
                    probability=parent_state.probability * 0.1,  # Reduced probability for mutations
                    schedule=mutated_schedule,
                    total_time=self._calculate_total_time(mutated_schedule),
                    resource_efficiency=self._calculate_resource_efficiency(mutated_schedule, resources),
                    conflicts=self._count_conflicts(mutated_schedule, tasks)
                )
                mutated_state.energy = self._calculate_state_energy(mutated_state, tasks, resources)
                
                self.superposition_states.append(mutated_state)
    
    def _collapse_superposition(self) -> Dict[str, Dict[str, Any]]:
        """Collapse superposition to single best schedule."""
        if not self.superposition_states:
            return {}
        
        # Find state with minimum energy (best overall quality)
        best_state = min(self.superposition_states, key=lambda s: s.energy)
        
        logger.info(f"Collapsed to schedule with energy {best_state.energy:.4f}, "
                   f"efficiency {best_state.resource_efficiency:.3f}")
        
        return best_state.schedule
    
    def _sort_tasks_by_constraints(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks considering dependencies and priorities."""
        # Topological sort with priority ordering
        visited = set()
        result = []
        temp_mark = set()
        
        def visit(task):
            if task.id in temp_mark:
                # Cycle detected - handle gracefully
                return
            if task.id in visited:
                return
            
            temp_mark.add(task.id)
            
            # Visit dependencies first
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task:
                    visit(dep_task)
            
            temp_mark.remove(task.id)
            visited.add(task.id)
            result.append(task)
        
        # Sort by priority first
        priority_sorted = sorted(tasks, key=lambda t: (t.priority.weight, t.urgency_factor), reverse=True)
        
        for task in priority_sorted:
            visit(task)
        
        return result
    
    def _calculate_interference(self, state1: SuperpositionState, 
                               state2: SuperpositionState) -> float:
        """Calculate quantum interference between two states."""
        # Compare schedules for similarity
        common_tasks = set(state1.schedule.keys()) & set(state2.schedule.keys())
        if not common_tasks:
            return 0.0
        
        similarity = 0.0
        for task_id in common_tasks:
            s1_info = state1.schedule[task_id]
            s2_info = state2.schedule[task_id]
            
            # Resource similarity
            if s1_info['resource_id'] == s2_info['resource_id']:
                similarity += 0.5
            
            # Time similarity
            time_diff = abs(s1_info['start_time'] - s2_info['start_time'])
            max_time = max(s1_info['start_time'], s2_info['start_time'], 1)
            time_similarity = 1.0 - (time_diff / max_time)
            similarity += 0.3 * time_similarity
        
        similarity /= len(common_tasks)
        
        # Interference coefficient (constructive when similar, destructive when different)
        interference = np.cos(np.pi * similarity)
        return max(0.0, interference)  # Only constructive interference
    
    def _combine_schedules(self, schedule1: Dict[str, Dict[str, Any]], 
                          schedule2: Dict[str, Dict[str, Any]],
                          interference: float, tasks: List[Task], 
                          resources: Dict[str, Resource]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Combine two schedules through quantum superposition."""
        combined = {}
        task_dict = {task.id: task for task in tasks}
        
        all_tasks = set(schedule1.keys()) | set(schedule2.keys())
        
        for task_id in all_tasks:
            task = task_dict.get(task_id)
            if not task:
                continue
            
            if task_id in schedule1 and task_id in schedule2:
                # Both schedules have this task - combine with interference weight
                info1 = schedule1[task_id]
                info2 = schedule2[task_id]
                
                if np.random.random() < interference:
                    # Take from first schedule
                    combined[task_id] = info1.copy()
                else:
                    # Take from second schedule  
                    combined[task_id] = info2.copy()
                    
            elif task_id in schedule1:
                combined[task_id] = schedule1[task_id].copy()
            else:
                combined[task_id] = schedule2[task_id].copy()
        
        # Validate and fix conflicts in combined schedule
        return self._fix_schedule_conflicts(combined, tasks, resources)
    
    def _mutate_schedule(self, schedule: Dict[str, Dict[str, Any]], 
                        tasks: List[Task], resources: Dict[str, Resource]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Apply quantum mutation to schedule."""
        mutated = {}
        for task_id, info in schedule.items():
            mutated[task_id] = info.copy()
        
        if not mutated:
            return None
        
        # Apply random mutations
        mutation_rate = 0.2
        task_ids = list(mutated.keys())
        resource_ids = list(resources.keys())
        
        for task_id in task_ids:
            if np.random.random() < mutation_rate:
                # Resource mutation
                if resource_ids:
                    mutated[task_id]['resource_id'] = np.random.choice(resource_ids)
                
                # Time mutation (small adjustment)
                time_shift = np.random.randint(-30, 31)  # ±30 minute shift
                current_start = mutated[task_id]['start_time']
                new_start = max(0, current_start + time_shift)
                duration = mutated[task_id]['duration']
                
                mutated[task_id]['start_time'] = new_start
                mutated[task_id]['end_time'] = new_start + duration
        
        # Fix conflicts after mutation
        return self._fix_schedule_conflicts(mutated, tasks, resources)
    
    def _fix_schedule_conflicts(self, schedule: Dict[str, Dict[str, Any]], 
                               tasks: List[Task], resources: Dict[str, Resource]) -> Dict[str, Dict[str, Any]]:
        """Fix scheduling conflicts and validate dependencies."""
        fixed = schedule.copy()
        task_dict = {task.id: task for task in tasks}
        
        # Fix dependency violations
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            for task_id, info in fixed.items():
                task = task_dict.get(task_id)
                if not task:
                    continue
                
                min_start_time = 0
                for dep_id in task.dependencies:
                    if dep_id in fixed:
                        dep_end_time = fixed[dep_id]['end_time']
                        min_start_time = max(min_start_time, dep_end_time)
                
                if info['start_time'] < min_start_time:
                    # Adjust start time to satisfy dependencies
                    info['start_time'] = min_start_time
                    info['end_time'] = min_start_time + info['duration']
                    changed = True
        
        return fixed
    
    def _calculate_total_time(self, schedule: Dict[str, Dict[str, Any]]) -> int:
        """Calculate total completion time of schedule."""
        if not schedule:
            return 0
        return max(info['end_time'] for info in schedule.values())
    
    def _calculate_resource_efficiency(self, schedule: Dict[str, Dict[str, Any]], 
                                     resources: Dict[str, Resource]) -> float:
        """Calculate resource utilization efficiency."""
        if not schedule or not resources:
            return 0.0
        
        total_time = self._calculate_total_time(schedule)
        if total_time == 0:
            return 0.0
        
        # Calculate utilization per resource
        resource_usage = {rid: 0 for rid in resources.keys()}
        
        for info in schedule.values():
            resource_id = info['resource_id']
            if resource_id in resource_usage:
                resource_usage[resource_id] += info['duration']
        
        # Calculate efficiency as average utilization
        utilizations = []
        for resource_id, usage in resource_usage.items():
            utilization = usage / total_time if total_time > 0 else 0
            utilizations.append(min(1.0, utilization))
        
        return np.mean(utilizations) if utilizations else 0.0
    
    def _count_conflicts(self, schedule: Dict[str, Dict[str, Any]], tasks: List[Task]) -> int:
        """Count scheduling conflicts."""
        conflicts = 0
        task_dict = {task.id: task for task in tasks}
        
        # Check resource conflicts (overlapping tasks on same resource)
        resource_tasks = {}
        for task_id, info in schedule.items():
            resource_id = info['resource_id']
            if resource_id not in resource_tasks:
                resource_tasks[resource_id] = []
            resource_tasks[resource_id].append((task_id, info['start_time'], info['end_time']))
        
        for resource_id, task_list in resource_tasks.items():
            # Check for overlaps
            for i in range(len(task_list)):
                for j in range(i + 1, len(task_list)):
                    task1_id, start1, end1 = task_list[i]
                    task2_id, start2, end2 = task_list[j]
                    
                    # Check overlap
                    if not (end1 <= start2 or end2 <= start1):
                        conflicts += 1
        
        # Check task conflicts (specified conflicts)
        for task_id, info in schedule.items():
            task = task_dict.get(task_id)
            if task:
                for conflict_id in task.conflicts:
                    if conflict_id in schedule:
                        # Check if they're scheduled at overlapping times
                        conflict_info = schedule[conflict_id]
                        if not (info['end_time'] <= conflict_info['start_time'] or
                               conflict_info['end_time'] <= info['start_time']):
                            conflicts += 1
        
        return conflicts
    
    def _calculate_state_energy(self, state: SuperpositionState, 
                               tasks: List[Task], resources: Dict[str, Resource]) -> float:
        """Calculate energy (cost) of a superposition state."""
        energy = 0.0
        
        # Time penalty
        energy += state.total_time * 0.01
        
        # Resource efficiency bonus/penalty
        energy += (1.0 - state.resource_efficiency) * 5.0
        
        # Conflict penalty
        energy += state.conflicts * 10.0
        
        # Priority satisfaction
        task_dict = {task.id: task for task in tasks}
        priority_penalty = 0.0
        
        for task_id, info in state.schedule.items():
            task = task_dict.get(task_id)
            if task:
                # Higher priority tasks should be scheduled earlier
                normalized_start = info['start_time'] / max(1, state.total_time)
                priority_weight = 1.0 - task.priority.weight  # Invert for penalty
                priority_penalty += priority_weight * normalized_start * 2.0
        
        energy += priority_penalty
        
        return energy
    
    def _check_convergence(self) -> bool:
        """Check if superposition has converged."""
        if len(self.superposition_states) < 2:
            return True
        
        # Check energy variance
        energies = [state.energy for state in self.superposition_states]
        energy_variance = np.var(energies)
        
        return energy_variance < 0.01  # Convergence threshold
    
    def get_superposition_metrics(self) -> Dict[str, Any]:
        """Get metrics about the superposition process."""
        if not self.superposition_states:
            return {}
        
        energies = [state.energy for state in self.superposition_states]
        probabilities = [state.probability for state in self.superposition_states]
        efficiencies = [state.resource_efficiency for state in self.superposition_states]
        
        return {
            'num_states': len(self.superposition_states),
            'min_energy': min(energies),
            'max_energy': max(energies),
            'avg_energy': np.mean(energies),
            'energy_variance': np.var(energies),
            'total_probability': sum(probabilities),
            'avg_efficiency': np.mean(efficiencies),
            'entropy': -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0)
        }