"""Task representation and management for quantum-inspired scheduling."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import uuid


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


@dataclass
class Task:
    """Quantum-inspired task representation with superposition capabilities."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing constraints
    estimated_duration: int = 60  # minutes
    deadline: Optional[datetime] = None
    earliest_start: Optional[datetime] = None
    
    # Dependencies and relationships
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    conflicts: Set[str] = field(default_factory=set)
    
    # Resource requirements
    required_resources: Dict[str, int] = field(default_factory=dict)
    preferred_resources: Dict[str, int] = field(default_factory=dict)
    
    # Quantum properties
    superposition_states: List[Dict[str, Any]] = field(default_factory=list)
    entanglement_partners: Set[str] = field(default_factory=set)
    quantum_weight: float = 1.0
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[int] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed properties."""
        if not self.name:
            self.name = f"Task-{self.id[:8]}"
            
        # Initialize quantum weight based on priority
        self.quantum_weight = self.priority.weight
        
        # Create default superposition state
        if not self.superposition_states:
            self.superposition_states = [{
                'probability': 1.0,
                'duration': self.estimated_duration,
                'resources': self.required_resources.copy(),
                'outcome': 'success'
            }]
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return (
            self.status == TaskStatus.PENDING and
            len(self.dependencies) == 0 and
            (self.earliest_start is None or datetime.now() >= self.earliest_start)
        )
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is past its deadline."""
        return (
            self.deadline is not None and 
            datetime.now() > self.deadline and
            self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        )
    
    @property
    def urgency_factor(self) -> float:
        """Calculate urgency based on deadline proximity."""
        if self.deadline is None:
            return 0.0
            
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return 1.0  # Maximum urgency for overdue tasks
            
        # Normalize by estimated duration
        duration_seconds = self.estimated_duration * 60
        return max(0.0, min(1.0, 1.0 - time_remaining / (duration_seconds * 2)))
    
    def add_dependency(self, task_id: str):
        """Add a task dependency."""
        self.dependencies.add(task_id)
    
    def remove_dependency(self, task_id: str):
        """Remove a task dependency."""
        self.dependencies.discard(task_id)
    
    def add_conflict(self, task_id: str):
        """Add a task conflict (cannot run simultaneously)."""
        self.conflicts.add(task_id)
    
    def entangle_with(self, task_id: str):
        """Create quantum entanglement with another task."""
        self.entanglement_partners.add(task_id)
    
    def add_superposition_state(self, probability: float, **state_kwargs):
        """Add a quantum superposition state."""
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
            
        # Normalize existing probabilities
        current_total = sum(state['probability'] for state in self.superposition_states)
        remaining = 1.0 - probability
        
        if current_total > 0:
            scale_factor = remaining / current_total
            for state in self.superposition_states:
                state['probability'] *= scale_factor
        
        # Add new state
        new_state = {
            'probability': probability,
            'duration': state_kwargs.get('duration', self.estimated_duration),
            'resources': state_kwargs.get('resources', self.required_resources.copy()),
            'outcome': state_kwargs.get('outcome', 'success')
        }
        
        self.superposition_states.append(new_state)
    
    def collapse_superposition(self, outcome_state: Dict[str, Any]):
        """Collapse quantum superposition to a specific state."""
        self.estimated_duration = outcome_state.get('duration', self.estimated_duration)
        self.required_resources = outcome_state.get('resources', self.required_resources)
        
        # Keep only the collapsed state
        self.superposition_states = [outcome_state]
    
    def calculate_quantum_interference(self, other_task: 'Task') -> float:
        """Calculate quantum interference with another task."""
        if not isinstance(other_task, Task):
            return 0.0
        
        # Base interference from conflicts
        conflict_interference = 0.5 if other_task.id in self.conflicts else 0.0
        
        # Resource competition interference
        resource_interference = 0.0
        for resource, amount in self.required_resources.items():
            if resource in other_task.required_resources:
                competition = min(amount, other_task.required_resources[resource])
                resource_interference += competition / max(amount, other_task.required_resources[resource])
        
        # Entanglement enhancement
        entanglement_boost = 0.3 if other_task.id in self.entanglement_partners else 0.0
        
        # Priority interference
        priority_diff = abs(self.priority.weight - other_task.priority.weight)
        priority_interference = priority_diff * 0.1
        
        return min(1.0, conflict_interference + resource_interference + priority_interference - entanglement_boost)
    
    def start_execution(self):
        """Mark task as started."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start task in {self.status} status")
            
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete_execution(self, actual_duration: Optional[int] = None):
        """Mark task as completed."""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task in {self.status} status")
            
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        
        if actual_duration is not None:
            self.actual_duration = actual_duration
        elif self.started_at is not None:
            duration_delta = datetime.now() - self.started_at
            self.actual_duration = int(duration_delta.total_seconds() / 60)
    
    def block_execution(self, reason: str = ""):
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        if reason:
            self.attributes['block_reason'] = reason
    
    def unblock_execution(self):
        """Remove block and return to pending status."""
        if self.status == TaskStatus.BLOCKED:
            self.status = TaskStatus.PENDING
            if 'block_reason' in self.attributes:
                del self.attributes['block_reason']
    
    def cancel_execution(self, reason: str = ""):
        """Cancel task execution."""
        self.status = TaskStatus.CANCELLED
        if reason:
            self.attributes['cancel_reason'] = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'priority': self.priority.name,
            'status': self.status.value,
            'estimated_duration': self.estimated_duration,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'earliest_start': self.earliest_start.isoformat() if self.earliest_start else None,
            'dependencies': list(self.dependencies),
            'dependents': list(self.dependents),
            'conflicts': list(self.conflicts),
            'required_resources': self.required_resources,
            'preferred_resources': self.preferred_resources,
            'superposition_states': self.superposition_states,
            'entanglement_partners': list(self.entanglement_partners),
            'quantum_weight': self.quantum_weight,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'actual_duration': self.actual_duration,
            'tags': list(self.tags),
            'attributes': self.attributes,
            'is_ready': self.is_ready,
            'is_overdue': self.is_overdue,
            'urgency_factor': self.urgency_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary representation."""
        task = cls()
        
        # Basic properties
        task.id = data.get('id', task.id)
        task.name = data.get('name', '')
        task.description = data.get('description', '')
        task.priority = TaskPriority[data.get('priority', 'MEDIUM')]
        task.status = TaskStatus(data.get('status', 'pending'))
        
        # Timing
        task.estimated_duration = data.get('estimated_duration', 60)
        if data.get('deadline'):
            task.deadline = datetime.fromisoformat(data['deadline'])
        if data.get('earliest_start'):
            task.earliest_start = datetime.fromisoformat(data['earliest_start'])
            
        # Dependencies and relationships
        task.dependencies = set(data.get('dependencies', []))
        task.dependents = set(data.get('dependents', []))
        task.conflicts = set(data.get('conflicts', []))
        
        # Resources
        task.required_resources = data.get('required_resources', {})
        task.preferred_resources = data.get('preferred_resources', {})
        
        # Quantum properties
        task.superposition_states = data.get('superposition_states', [])
        task.entanglement_partners = set(data.get('entanglement_partners', []))
        task.quantum_weight = data.get('quantum_weight', 1.0)
        
        # Execution tracking
        if data.get('created_at'):
            task.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            task.started_at = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            task.completed_at = datetime.fromisoformat(data['completed_at'])
        task.actual_duration = data.get('actual_duration')
        
        # Metadata
        task.tags = set(data.get('tags', []))
        task.attributes = data.get('attributes', {})
        
        return task
    
    def __str__(self) -> str:
        """String representation of task."""
        return f"Task({self.name}, {self.priority.name}, {self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Task(id={self.id[:8]}..., name='{self.name}', "
                f"priority={self.priority.name}, status={self.status.value}, "
                f"duration={self.estimated_duration}min)")