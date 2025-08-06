"""Quantum annealing optimization for task scheduling."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import random
import math
from dataclasses import dataclass

from ..core.task import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class AnnealingConfig:
    """Configuration for quantum annealing."""
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95
    max_iterations: int = 1000
    min_improvement_threshold: float = 1e-6
    parallel_chains: int = 4


class QuantumAnnealingOptimizer:
    """Quantum annealing optimizer for task-resource assignment."""
    
    def __init__(self, config: Optional[AnnealingConfig] = None):
        """Initialize quantum annealing optimizer."""
        self.config = config or AnnealingConfig()
        self.best_energy = float('inf')
        self.best_assignment = {}
        self.energy_history = []
        
        logger.info("Initialized QuantumAnnealingOptimizer")
    
    def optimize(self, tasks: List[Task], resources: Dict[str, int], 
                time_limit: Optional[int] = None) -> Dict[str, str]:
        """Optimize task-resource assignment using quantum annealing."""
        if not tasks or not resources:
            logger.warning("No tasks or resources provided")
            return {}
        
        logger.info(f"Starting quantum annealing with {len(tasks)} tasks and {len(resources)} resources")
        
        # Initialize multiple parallel annealing chains
        chains = []
        for i in range(self.config.parallel_chains):
            initial_assignment = self._generate_initial_assignment(tasks, resources)
            chains.append({
                'assignment': initial_assignment,
                'energy': self._calculate_energy(initial_assignment, tasks, resources),
                'temperature': self.config.initial_temperature,
                'chain_id': i
            })
        
        self.best_energy = float('inf')
        self.best_assignment = {}
        
        # Main annealing loop
        for iteration in range(self.config.max_iterations):
            improved = False
            
            # Process each chain
            for chain in chains:
                new_assignment = self._generate_neighbor(chain['assignment'], tasks, resources)
                new_energy = self._calculate_energy(new_assignment, tasks, resources)
                
                # Acceptance criterion (Metropolis-Hastings)
                delta_energy = new_energy - chain['energy']
                
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / chain['temperature']):
                    chain['assignment'] = new_assignment
                    chain['energy'] = new_energy
                    
                    # Update global best
                    if new_energy < self.best_energy:
                        self.best_energy = new_energy
                        self.best_assignment = new_assignment.copy()
                        improved = True
                
                # Cool down temperature
                chain['temperature'] *= self.config.cooling_rate
                chain['temperature'] = max(chain['temperature'], self.config.final_temperature)
            
            # Record energy history
            self.energy_history.append(self.best_energy)
            
            # Check for convergence
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: Best energy = {self.best_energy:.4f}")
                
                if iteration > 200 and not improved:
                    # Check if we've converged
                    recent_energies = self.energy_history[-50:]
                    if len(recent_energies) >= 50:
                        energy_variance = np.var(recent_energies)
                        if energy_variance < self.config.min_improvement_threshold:
                            logger.info(f"Converged at iteration {iteration}")
                            break
            
            # Quantum tunneling (restart worst chains)
            if iteration % 200 == 0 and iteration > 0:
                self._quantum_tunneling(chains, tasks, resources)
        
        logger.info(f"Quantum annealing completed. Best energy: {self.best_energy:.4f}")
        return self.best_assignment
    
    def _generate_initial_assignment(self, tasks: List[Task], 
                                   resources: Dict[str, int]) -> Dict[str, str]:
        """Generate initial random assignment."""
        assignment = {}
        resource_ids = list(resources.keys())
        
        for task in tasks:
            # Prefer resources that match task requirements
            preferred_resources = []
            
            for resource_id in resource_ids:
                # Check if resource can handle task requirements
                can_handle = True
                for req_resource, req_amount in task.required_resources.items():
                    if req_resource == resource_id and req_amount > resources[resource_id]:
                        can_handle = False
                        break
                
                if can_handle:
                    preferred_resources.append(resource_id)
            
            # Select resource
            if preferred_resources:
                assignment[task.id] = np.random.choice(preferred_resources)
            else:
                assignment[task.id] = np.random.choice(resource_ids)
        
        return assignment
    
    def _generate_neighbor(self, current_assignment: Dict[str, str],
                          tasks: List[Task], resources: Dict[str, int]) -> Dict[str, str]:
        """Generate neighboring assignment through local search."""
        neighbor = current_assignment.copy()
        resource_ids = list(resources.keys())
        
        if not neighbor:
            return neighbor
        
        # Randomly select task to reassign
        task_ids = list(neighbor.keys())
        selected_task_id = np.random.choice(task_ids)
        
        # Find task object
        selected_task = next((t for t in tasks if t.id == selected_task_id), None)
        if not selected_task:
            return neighbor
        
        # Strategy selection
        strategy = np.random.choice(['random_reassign', 'smart_reassign', 'swap_tasks'], 
                                   p=[0.4, 0.4, 0.2])
        
        if strategy == 'random_reassign':
            # Randomly reassign to different resource
            current_resource = neighbor[selected_task_id]
            other_resources = [r for r in resource_ids if r != current_resource]
            if other_resources:
                neighbor[selected_task_id] = np.random.choice(other_resources)
        
        elif strategy == 'smart_reassign':
            # Reassign based on task requirements and resource availability
            best_resource = self._find_best_resource(selected_task, resources, neighbor)
            if best_resource and best_resource != neighbor[selected_task_id]:
                neighbor[selected_task_id] = best_resource
        
        elif strategy == 'swap_tasks':
            # Swap resources between two tasks
            if len(task_ids) > 1:
                other_task_id = np.random.choice([tid for tid in task_ids if tid != selected_task_id])
                neighbor[selected_task_id], neighbor[other_task_id] = \
                    neighbor[other_task_id], neighbor[selected_task_id]
        
        return neighbor
    
    def _find_best_resource(self, task: Task, resources: Dict[str, int],
                          current_assignment: Dict[str, str]) -> Optional[str]:
        """Find best resource for a task based on requirements and load."""
        best_resource = None
        best_score = float('inf')
        
        for resource_id, capacity in resources.items():
            score = 0.0
            
            # Check resource requirements satisfaction
            requirements_met = True
            for req_resource, req_amount in task.required_resources.items():
                if req_resource == resource_id and req_amount > capacity:
                    requirements_met = False
                    break
            
            if not requirements_met:
                continue
            
            # Calculate resource load
            resource_load = sum(1 for assignment in current_assignment.values() 
                              if assignment == resource_id)
            load_penalty = resource_load / capacity if capacity > 0 else float('inf')
            score += load_penalty
            
            # Prefer resources with better capacity match
            capacity_match = abs(capacity - sum(task.required_resources.values()))
            score += capacity_match * 0.1
            
            if score < best_score:
                best_score = score
                best_resource = resource_id
        
        return best_resource
    
    def _calculate_energy(self, assignment: Dict[str, str], 
                         tasks: List[Task], resources: Dict[str, int]) -> float:
        """Calculate total energy (cost) of assignment."""
        if not assignment:
            return float('inf')
        
        energy = 0.0
        task_dict = {task.id: task for task in tasks}
        
        # Resource utilization energy
        resource_loads = {rid: 0 for rid in resources.keys()}
        for task_id, resource_id in assignment.items():
            resource_loads[resource_id] += 1
        
        for resource_id, load in resource_loads.items():
            capacity = resources.get(resource_id, 1)
            if load > capacity:
                # Penalty for overloading
                energy += (load - capacity) ** 2 * 10
            else:
                # Reward for balanced utilization
                utilization = load / capacity if capacity > 0 else 0
                energy += (1 - utilization) ** 2
        
        # Task priority energy
        for task_id, resource_id in assignment.items():
            task = task_dict.get(task_id)
            if task:
                # Higher priority tasks should get better resources (lower energy)
                priority_weight = 1.0 - task.priority.weight
                energy += priority_weight * 2
        
        # Dependency satisfaction energy
        for task_id in assignment:
            task = task_dict.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in assignment:
                        # Penalty if dependent tasks are on same resource
                        if assignment[dep_id] == assignment[task_id]:
                            energy += 1.5
        
        # Conflict resolution energy
        for task_id in assignment:
            task = task_dict.get(task_id)
            if task:
                for conflict_id in task.conflicts:
                    if conflict_id in assignment:
                        # High penalty if conflicting tasks are on same resource
                        if assignment[conflict_id] == assignment[task_id]:
                            energy += 5.0
        
        # Resource requirement matching energy
        for task_id, resource_id in assignment.items():
            task = task_dict.get(task_id)
            if task:
                for req_resource, req_amount in task.required_resources.items():
                    available = resources.get(resource_id, 0)
                    if req_resource == resource_id:
                        if req_amount > available:
                            # Penalty for insufficient resources
                            energy += (req_amount - available) * 3
                        else:
                            # Small reward for resource match
                            energy -= 0.1
        
        # Quantum interference effects
        entangled_pairs = []
        for task_id in assignment:
            task = task_dict.get(task_id)
            if task:
                for entangled_id in task.entanglement_partners:
                    if entangled_id in assignment and (task_id, entangled_id) not in entangled_pairs:
                        entangled_pairs.append((task_id, entangled_id))
                        entangled_pairs.append((entangled_id, task_id))  # Symmetric
                        
                        # Entangled tasks benefit from being on different resources
                        if assignment[task_id] != assignment[entangled_id]:
                            energy -= 0.5
                        else:
                            energy += 0.2
        
        return energy
    
    def _quantum_tunneling(self, chains: List[Dict], tasks: List[Task], 
                          resources: Dict[str, int]) -> None:
        """Apply quantum tunneling to restart poor performing chains."""
        # Sort chains by energy
        chains.sort(key=lambda c: c['energy'])
        
        # Restart worst performing chains
        num_restart = max(1, len(chains) // 2)
        
        for i in range(-num_restart, 0):
            chain = chains[i]
            
            # Generate new random assignment with some bias toward good solutions
            if self.best_assignment:
                # 50% chance to use best assignment as starting point
                if np.random.random() < 0.5:
                    new_assignment = self.best_assignment.copy()
                    # Add some randomness
                    task_ids = list(new_assignment.keys())
                    num_changes = max(1, len(task_ids) // 4)
                    resource_ids = list(resources.keys())
                    
                    for _ in range(num_changes):
                        task_id = np.random.choice(task_ids)
                        new_assignment[task_id] = np.random.choice(resource_ids)
                else:
                    new_assignment = self._generate_initial_assignment(tasks, resources)
            else:
                new_assignment = self._generate_initial_assignment(tasks, resources)
            
            chain['assignment'] = new_assignment
            chain['energy'] = self._calculate_energy(new_assignment, tasks, resources)
            chain['temperature'] = self.config.initial_temperature * 0.5  # Warm restart
            
            logger.debug(f"Quantum tunneling: Restarted chain {chain['chain_id']}")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics and statistics."""
        if not self.energy_history:
            return {}
        
        return {
            'best_energy': self.best_energy,
            'initial_energy': self.energy_history[0] if self.energy_history else 0,
            'final_energy': self.energy_history[-1] if self.energy_history else 0,
            'energy_reduction': (self.energy_history[0] - self.energy_history[-1]) 
                              if len(self.energy_history) > 1 else 0,
            'convergence_iterations': len(self.energy_history),
            'energy_variance': np.var(self.energy_history[-100:]) if len(self.energy_history) >= 100 else 0,
            'improvement_ratio': (self.energy_history[0] - self.best_energy) / max(1, self.energy_history[0])
                               if self.energy_history and self.energy_history[0] > 0 else 0
        }
    
    def visualize_annealing_process(self) -> Dict[str, List[float]]:
        """Get data for visualizing the annealing process."""
        if not self.energy_history:
            return {'iterations': [], 'energies': []}
        
        return {
            'iterations': list(range(len(self.energy_history))),
            'energies': self.energy_history,
            'temperature_schedule': [self.config.initial_temperature * (self.config.cooling_rate ** i) 
                                   for i in range(len(self.energy_history))]
        }