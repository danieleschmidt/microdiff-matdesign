"""Quantum-Inspired Task Planner: Advanced scheduling using quantum principles."""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

# Use simplified scheduler that doesn't require heavy dependencies
try:
    from .core.scheduler import QuantumInspiredScheduler
except ImportError:
    from .core.simple_scheduler import QuantumInspiredScheduler

from .core.task import Task, TaskPriority, TaskStatus

# Optional imports - use if available
try:
    from .core.quantum_engine import QuantumEngine
except ImportError:
    QuantumEngine = None

try:
    from .algorithms.quantum_annealing import QuantumAnnealingOptimizer
except ImportError:
    QuantumAnnealingOptimizer = None

try:
    from .algorithms.superposition import SuperpositionScheduler
except ImportError:
    SuperpositionScheduler = None

try:
    from .utils.visualization import ScheduleVisualizer
except ImportError:
    ScheduleVisualizer = None

__all__ = [
    "QuantumInspiredScheduler",
    "Task",
    "TaskPriority", 
    "TaskStatus",
    "QuantumEngine",
    "QuantumAnnealingOptimizer",
    "SuperpositionScheduler",
    "ScheduleVisualizer",
]