"""Quantum computation engine for task scheduling optimization."""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
import math

from .task import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class QuantumGate(Enum):
    """Quantum gate operations for task scheduling."""
    HADAMARD = "H"  # Create superposition
    PAULI_X = "X"   # Flip task state
    PAULI_Y = "Y"   # Complex rotation
    PAULI_Z = "Z"   # Phase flip
    CNOT = "CNOT"   # Entanglement gate
    TOFFOLI = "TOF" # Three-qubit gate
    

@dataclass
class QuantumState:
    """Quantum state representation for task scheduling."""
    amplitude: complex
    phase: float
    probability: float
    task_assignments: Dict[str, str]  # task_id -> resource_id
    energy: float = 0.0
    
    def normalize(self):
        """Normalize quantum state amplitudes."""
        magnitude = abs(self.amplitude)
        if magnitude > 0:
            self.amplitude = self.amplitude / magnitude
            self.probability = magnitude ** 2
        else:
            self.probability = 0.0


class QuantumEngine:
    """Quantum computation engine for task scheduling optimization."""
    
    def __init__(self, 
                 num_qubits: int = 16,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6,
                 temperature: float = 1.0):
        """Initialize quantum engine."""
        self.num_qubits = num_qubits
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature
        
        # Quantum state space
        self.quantum_states: List[QuantumState] = []
        self.entanglement_matrix = np.zeros((num_qubits, num_qubits))
        
        # Task mapping
        self.task_qubit_map: Dict[str, int] = {}
        self.qubit_task_map: Dict[int, str] = {}
        
        # Optimization parameters
        self.hamiltonian_coefficients = {
            'priority_weight': 2.0,
            'deadline_weight': 3.0,
            'dependency_weight': 1.5,
            'resource_weight': 1.0,
            'conflict_weight': -2.0
        }
        
        logger.info(f"Initialized QuantumEngine with {num_qubits} qubits")
    
    def initialize_quantum_state(self, tasks: List[Task], resources: Dict[str, int]) -> None:
        """Initialize quantum state space for task scheduling."""
        if len(tasks) > self.num_qubits:
            logger.warning(f"Tasks ({len(tasks)}) exceed qubits ({self.num_qubits}). Using subset.")
            tasks = tasks[:self.num_qubits]
        
        # Map tasks to qubits
        self.task_qubit_map = {task.id: i for i, task in enumerate(tasks)}
        self.qubit_task_map = {i: task.id for i, task in enumerate(tasks)}
        
        # Create superposition of all possible task assignments
        num_states = min(1024, 2 ** len(tasks))  # Limit state space
        self.quantum_states = []
        
        for i in range(num_states):
            # Generate random task-resource assignment
            assignment = self._generate_random_assignment(tasks, resources)
            
            # Calculate initial amplitude and energy
            energy = self._calculate_hamiltonian_energy(assignment, tasks, resources)
            amplitude = complex(np.random.random(), np.random.random())
            
            state = QuantumState(
                amplitude=amplitude,
                phase=np.random.random() * 2 * np.pi,
                probability=0.0,
                task_assignments=assignment,
                energy=energy
            )
            state.normalize()
            self.quantum_states.append(state)
        
        # Normalize state probabilities
        self._normalize_state_probabilities()
        
        logger.info(f"Initialized {len(self.quantum_states)} quantum states")
    
    def apply_quantum_annealing(self, tasks: List[Task], resources: Dict[str, int]) -> Dict[str, str]:
        """Apply quantum annealing for optimal task scheduling."""
        if not self.quantum_states:
            self.initialize_quantum_state(tasks, resources)
        
        best_energy = float('inf')
        best_assignment = {}
        temperature = self.temperature
        
        for iteration in range(self.max_iterations):
            # Select random quantum state
            state_idx = np.random.choice(len(self.quantum_states), 
                                       p=[s.probability for s in self.quantum_states])
            current_state = self.quantum_states[state_idx]
            
            # Generate neighbor state through quantum tunneling
            neighbor_assignment = self._quantum_tunnel(current_state.task_assignments, tasks, resources)
            neighbor_energy = self._calculate_hamiltonian_energy(neighbor_assignment, tasks, resources)
            
            # Acceptance probability (quantum annealing)
            energy_diff = neighbor_energy - current_state.energy
            if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / temperature):
                # Accept new state
                current_state.task_assignments = neighbor_assignment
                current_state.energy = neighbor_energy
                
                # Update best solution
                if neighbor_energy < best_energy:
                    best_energy = neighbor_energy
                    best_assignment = neighbor_assignment.copy()
            
            # Cool down temperature
            temperature *= 0.995
            
            # Check convergence
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: Best energy = {best_energy:.4f}")
                
                if abs(energy_diff) < self.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        
        logger.info(f"Quantum annealing completed. Best energy: {best_energy:.4f}")
        return best_assignment
    
    def create_task_entanglement(self, task1_id: str, task2_id: str, strength: float = 1.0) -> None:
        """Create quantum entanglement between tasks."""
        if task1_id in self.task_qubit_map and task2_id in self.task_qubit_map:
            qubit1 = self.task_qubit_map[task1_id]
            qubit2 = self.task_qubit_map[task2_id]
            
            self.entanglement_matrix[qubit1, qubit2] = strength
            self.entanglement_matrix[qubit2, qubit1] = strength
            
            logger.debug(f"Created entanglement between tasks {task1_id} and {task2_id}")
    
    def apply_quantum_superposition(self, task_id: str) -> List[Dict[str, Any]]:
        """Apply quantum superposition to explore multiple task states."""
        if task_id not in self.task_qubit_map:
            return []
        
        qubit_idx = self.task_qubit_map[task_id]
        superposition_states = []
        
        # Generate superposition of different execution scenarios
        scenarios = [
            {'duration_factor': 0.8, 'success_probability': 0.95, 'resource_factor': 0.9},
            {'duration_factor': 1.0, 'success_probability': 0.90, 'resource_factor': 1.0},
            {'duration_factor': 1.2, 'success_probability': 0.85, 'resource_factor': 1.1},
            {'duration_factor': 1.5, 'success_probability': 0.75, 'resource_factor': 1.2}
        ]
        
        for scenario in scenarios:
            # Apply Hadamard gate to create superposition
            amplitude = complex(1 / np.sqrt(len(scenarios)), 0)
            
            state = {
                'amplitude': amplitude,
                'probability': abs(amplitude) ** 2,
                'scenario': scenario,
                'phase': np.random.random() * 2 * np.pi
            }
            superposition_states.append(state)
        
        return superposition_states
    
    def measure_quantum_state(self, collapse_probability: bool = True) -> Dict[str, str]:
        """Measure quantum state and collapse to classical assignment."""
        if not self.quantum_states:
            return {}
        
        # Select state based on probability distribution
        probabilities = [state.probability for state in self.quantum_states]
        if sum(probabilities) == 0:
            # Equal probability if all are zero
            probabilities = [1.0 / len(self.quantum_states)] * len(self.quantum_states)
        
        selected_idx = np.random.choice(len(self.quantum_states), p=probabilities)
        selected_state = self.quantum_states[selected_idx]
        
        if collapse_probability:
            # Collapse to single state (quantum measurement)
            self.quantum_states = [selected_state]
            selected_state.probability = 1.0
        
        logger.info(f"Quantum measurement: Energy = {selected_state.energy:.4f}")
        return selected_state.task_assignments
    
    def calculate_quantum_interference(self, task1_id: str, task2_id: str) -> float:
        """Calculate quantum interference between two tasks."""
        if task1_id not in self.task_qubit_map or task2_id not in self.task_qubit_map:
            return 0.0
        
        qubit1 = self.task_qubit_map[task1_id]
        qubit2 = self.task_qubit_map[task2_id]
        
        # Base interference from entanglement
        entanglement = self.entanglement_matrix[qubit1, qubit2]
        
        # Calculate interference from quantum states
        interference_sum = 0.0
        for state in self.quantum_states:
            if task1_id in state.task_assignments and task2_id in state.task_assignments:
                # Phase difference contributes to interference
                phase_diff = abs(state.phase - np.pi)
                interference = state.probability * np.cos(phase_diff) * entanglement
                interference_sum += interference
        
        return min(1.0, max(-1.0, interference_sum))
    
    def optimize_quantum_circuit(self, tasks: List[Task]) -> List[Tuple[QuantumGate, List[int]]]:
        """Optimize quantum circuit for task scheduling."""
        circuit = []
        
        # Apply Hadamard gates for superposition
        for i in range(min(len(tasks), self.num_qubits)):
            circuit.append((QuantumGate.HADAMARD, [i]))
        
        # Apply entanglement gates for dependent tasks
        for task in tasks:
            task_qubit = self.task_qubit_map.get(task.id)
            if task_qubit is not None:
                for dep_id in task.dependencies:
                    if dep_id in self.task_qubit_map:
                        dep_qubit = self.task_qubit_map[dep_id]
                        circuit.append((QuantumGate.CNOT, [dep_qubit, task_qubit]))
        
        # Apply phase gates based on task priorities
        for task in tasks:
            task_qubit = self.task_qubit_map.get(task.id)
            if task_qubit is not None:
                # Higher priority tasks get different phase
                if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                    circuit.append((QuantumGate.PAULI_Z, [task_qubit]))
        
        return circuit
    
    def _generate_random_assignment(self, tasks: List[Task], resources: Dict[str, int]) -> Dict[str, str]:
        """Generate random task-resource assignment."""
        assignment = {}
        resource_list = list(resources.keys())
        
        for task in tasks:
            if resource_list:
                # Consider task resource requirements
                preferred_resources = []
                for resource in resource_list:
                    if resource in task.required_resources:
                        preferred_resources.extend([resource] * task.required_resources[resource])
                
                if preferred_resources:
                    assignment[task.id] = np.random.choice(preferred_resources)
                else:
                    assignment[task.id] = np.random.choice(resource_list)
            else:
                assignment[task.id] = "default"
        
        return assignment
    
    def _calculate_hamiltonian_energy(self, assignment: Dict[str, str], 
                                    tasks: List[Task], resources: Dict[str, int]) -> float:
        """Calculate Hamiltonian energy for task assignment."""
        energy = 0.0
        
        # Create task lookup
        task_dict = {task.id: task for task in tasks}
        
        for task_id, resource_id in assignment.items():
            task = task_dict.get(task_id)
            if not task:
                continue
            
            # Priority energy (higher priority = lower energy)
            priority_energy = -self.hamiltonian_coefficients['priority_weight'] * task.priority.weight
            energy += priority_energy
            
            # Deadline urgency energy
            urgency_energy = self.hamiltonian_coefficients['deadline_weight'] * task.urgency_factor
            energy += urgency_energy
            
            # Dependency satisfaction energy
            for dep_id in task.dependencies:
                if dep_id in assignment:
                    # Penalty if dependency is assigned to same resource
                    if assignment[dep_id] == resource_id:
                        energy += self.hamiltonian_coefficients['dependency_weight']
            
            # Resource constraint energy
            resource_usage = sum(1 for t_id, r_id in assignment.items() if r_id == resource_id)
            max_capacity = resources.get(resource_id, 1)
            if resource_usage > max_capacity:
                energy += self.hamiltonian_coefficients['resource_weight'] * (resource_usage - max_capacity) ** 2
            
            # Conflict energy
            for other_task_id, other_resource_id in assignment.items():
                if task_id != other_task_id and resource_id == other_resource_id:
                    other_task = task_dict.get(other_task_id)
                    if other_task and other_task_id in task.conflicts:
                        energy += self.hamiltonian_coefficients['conflict_weight']
        
        return energy
    
    def _quantum_tunnel(self, current_assignment: Dict[str, str], 
                       tasks: List[Task], resources: Dict[str, int]) -> Dict[str, str]:
        """Perform quantum tunneling to neighboring state."""
        new_assignment = current_assignment.copy()
        resource_list = list(resources.keys())
        
        # Randomly select task to reassign
        if not new_assignment:
            return new_assignment
            
        task_id = np.random.choice(list(new_assignment.keys()))
        
        # Quantum tunneling: allow non-local moves
        if np.random.random() < 0.1:  # 10% chance of tunneling
            # Tunnel to completely random resource
            if resource_list:
                new_assignment[task_id] = np.random.choice(resource_list)
        else:
            # Local move to neighboring resource
            current_resource = new_assignment[task_id]
            if current_resource in resource_list:
                current_idx = resource_list.index(current_resource)
                # Move to adjacent resource
                if len(resource_list) > 1:
                    direction = np.random.choice([-1, 1])
                    new_idx = (current_idx + direction) % len(resource_list)
                    new_assignment[task_id] = resource_list[new_idx]
        
        return new_assignment
    
    def _normalize_state_probabilities(self) -> None:
        """Normalize probabilities across all quantum states."""
        total_prob = sum(state.probability for state in self.quantum_states)
        if total_prob > 0:
            for state in self.quantum_states:
                state.probability /= total_prob
    
    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get quantum computation metrics."""
        if not self.quantum_states:
            return {}
        
        energies = [state.energy for state in self.quantum_states]
        probabilities = [state.probability for state in self.quantum_states]
        
        return {
            'min_energy': min(energies),
            'max_energy': max(energies),
            'avg_energy': np.mean(energies),
            'energy_variance': np.var(energies),
            'entropy': -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0),
            'num_states': len(self.quantum_states),
            'entanglement_density': np.count_nonzero(self.entanglement_matrix) / (self.num_qubits ** 2)
        }