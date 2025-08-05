"""Auto-scaling and load balancing utilities for MicroDiff-MatDesign."""

import time
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .logging_config import get_logger
from .monitoring import system_monitor, performance_tracker
from .error_handling import handle_errors, MicroDiffError


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    RESPONSE_TIME_BASED = "response_time_based"
    HYBRID = "hybrid"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    policy: ScalingPolicy = ScalingPolicy.HYBRID
    min_workers: int = 1
    max_workers: int = 16
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_queue_size: int = 100
    target_response_time_ms: float = 1000.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_seconds: int = 300
    evaluation_interval: int = 60


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    node_id: str
    status: str = "idle"  # idle, busy, failed
    current_load: float = 0.0
    task_count: int = 0
    last_heartbeat: float = 0.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        self.last_heartbeat = time.time()


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: str = "round_robin"):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'weighted')
        """
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.current_index = 0
        self._lock = threading.RLock()
        self.logger = get_logger('scaling.load_balancer')
        
        # Performance tracking
        self.request_counts = {}
        self.response_times = {}
        self.error_counts = {}
    
    def register_worker(self, worker: WorkerNode) -> None:
        """Register a new worker node.
        
        Args:
            worker: Worker node to register
        """
        with self._lock:
            self.workers[worker.node_id] = worker
            self.request_counts[worker.node_id] = 0
            self.response_times[worker.node_id] = []
            self.error_counts[worker.node_id] = 0
            
        self.logger.info(f"Registered worker: {worker.node_id}")
    
    def unregister_worker(self, node_id: str) -> None:
        """Unregister a worker node.
        
        Args:
            node_id: ID of worker to unregister
        """
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                del self.request_counts[node_id]
                del self.response_times[node_id]
                del self.error_counts[node_id]
                
        self.logger.info(f"Unregistered worker: {node_id}")
    
    def select_worker(self, task_requirements: Optional[List[str]] = None) -> Optional[WorkerNode]:
        """Select optimal worker for a task.
        
        Args:
            task_requirements: Required capabilities for the task
            
        Returns:
            Selected worker node or None if no suitable worker
        """
        with self._lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.status in ["idle", "busy"] and self._worker_can_handle(worker, task_requirements)
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(available_workers)
            elif self.strategy == "least_connections":
                return self._least_connections_selection(available_workers)
            elif self.strategy == "weighted":
                return self._weighted_selection(available_workers)
            else:
                return available_workers[0]  # Fallback
    
    def _worker_can_handle(self, worker: WorkerNode, requirements: Optional[List[str]]) -> bool:
        """Check if worker can handle task requirements."""
        if not requirements:
            return True
        
        return all(req in worker.capabilities for req in requirements)
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        if not workers:
            return None
        
        self.current_index = (self.current_index + 1) % len(workers)
        return workers[self.current_index]
    
    def _least_connections_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.task_count)
    
    def _weighted_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on performance weights."""
        # Calculate weights based on inverse of average response time
        weights = []
        for worker in workers:
            avg_response_time = (
                sum(self.response_times[worker.node_id]) / 
                len(self.response_times[worker.node_id])
                if self.response_times[worker.node_id] else 1.0
            )
            
            # Lower response time = higher weight
            weight = 1.0 / (avg_response_time + 0.001)
            weights.append(weight)
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(workers)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return workers[i]
        
        return workers[-1]  # Fallback
    
    def update_worker_status(self, node_id: str, status: str, load: float = 0.0) -> None:
        """Update worker status and load.
        
        Args:
            node_id: Worker node ID
            status: New status
            load: Current load (0.0-1.0)
        """
        with self._lock:
            if node_id in self.workers:
                self.workers[node_id].status = status
                self.workers[node_id].current_load = load
                self.workers[node_id].last_heartbeat = time.time()
    
    def record_request(self, node_id: str, response_time: float, success: bool = True) -> None:
        """Record request performance metrics.
        
        Args:
            node_id: Worker node ID
            response_time: Response time in milliseconds
            success: Whether request was successful
        """
        with self._lock:
            if node_id in self.workers:
                self.request_counts[node_id] += 1
                self.response_times[node_id].append(response_time)
                
                # Keep only recent response times (last 100)
                if len(self.response_times[node_id]) > 100:
                    self.response_times[node_id].pop(0)
                
                if not success:
                    self.error_counts[node_id] += 1
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        with self._lock:
            stats = {}
            
            for node_id, worker in self.workers.items():
                avg_response_time = (
                    sum(self.response_times[node_id]) / len(self.response_times[node_id])
                    if self.response_times[node_id] else 0.0
                )
                
                error_rate = (
                    self.error_counts[node_id] / self.request_counts[node_id] * 100
                    if self.request_counts[node_id] > 0 else 0.0
                )
                
                stats[node_id] = {
                    'status': worker.status,
                    'current_load': worker.current_load,
                    'task_count': worker.task_count,
                    'request_count': self.request_counts[node_id],
                    'avg_response_time_ms': avg_response_time,
                    'error_rate_percent': error_rate,
                    'last_heartbeat_age': time.time() - worker.last_heartbeat,
                    'capabilities': worker.capabilities
                }
            
            return stats
    
    def cleanup_stale_workers(self, max_age_seconds: int = 300) -> None:
        """Remove workers that haven't sent heartbeats."""
        current_time = time.time()
        stale_workers = []
        
        with self._lock:
            for node_id, worker in self.workers.items():
                if current_time - worker.last_heartbeat > max_age_seconds:
                    stale_workers.append(node_id)
        
        for node_id in stale_workers:
            self.unregister_worker(node_id)
            self.logger.warning(f"Removed stale worker: {node_id}")


class AutoScaler:
    """Automatic scaling system based on various metrics."""
    
    def __init__(self, config: ScalingConfig, load_balancer: LoadBalancer):
        """Initialize auto-scaler.
        
        Args:
            config: Scaling configuration
            load_balancer: Load balancer instance
        """
        self.config = config
        self.load_balancer = load_balancer
        self.logger = get_logger('scaling.auto_scaler')
        
        self.current_workers = config.min_workers
        self.last_scale_time = 0.0
        self.scaling_history = []
        self.metrics_history = []
        
        self._running = False
        self._monitor_thread = None
    
    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(metrics)
                
                if scaling_decision != 0:
                    self._execute_scaling_decision(scaling_decision, metrics)
                
                time.sleep(self.config.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(self.config.evaluation_interval)
    
    def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics for scaling decisions."""
        try:
            # Get system metrics
            system_metrics = system_monitor.collect_system_metrics()
            
            # Get performance metrics
            perf_metrics = performance_tracker.get_performance_metrics()
            
            # Get load balancer stats
            worker_stats = self.load_balancer.get_worker_stats()
            
            # Calculate aggregated metrics
            active_workers = len([w for w in worker_stats.values() if w['status'] != 'failed'])
            avg_load = sum(w['current_load'] for w in worker_stats.values()) / max(1, len(worker_stats))
            total_requests = sum(w['request_count'] for w in worker_stats.values())
            
            return {
                'timestamp': time.time(),
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'active_workers': active_workers,
                'avg_worker_load': avg_load,
                'total_requests': total_requests,
                'avg_response_time': perf_metrics.average_response_time,
                'error_rate': perf_metrics.error_rate,
                'queue_size': perf_metrics.queue_size,
                'operations_per_second': perf_metrics.operations_per_second
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect scaling metrics: {e}")
            return {
                'timestamp': time.time(),
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'active_workers': self.current_workers,
                'avg_worker_load': 0.0,
                'total_requests': 0,
                'avg_response_time': 0.0,
                'error_rate': 0.0,
                'queue_size': 0,
                'operations_per_second': 0.0
            }
    
    def _make_scaling_decision(self, metrics: Dict[str, Any]) -> int:
        """Make scaling decision based on metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Scaling decision: positive for scale up, negative for scale down, 0 for no change
        """
        # Check cooldown period
        if time.time() - self.last_scale_time < self.config.cooldown_seconds:
            return 0
        
        scale_factors = []
        
        # CPU-based scaling
        if self.config.policy in [ScalingPolicy.CPU_BASED, ScalingPolicy.HYBRID]:
            cpu_factor = metrics['cpu_percent'] / self.config.target_cpu_percent
            scale_factors.append(('cpu', cpu_factor))
        
        # Memory-based scaling
        if self.config.policy in [ScalingPolicy.MEMORY_BASED, ScalingPolicy.HYBRID]:
            memory_factor = metrics['memory_percent'] / self.config.target_memory_percent
            scale_factors.append(('memory', memory_factor))
        
        # Queue-based scaling
        if self.config.policy in [ScalingPolicy.QUEUE_BASED, ScalingPolicy.HYBRID]:
            queue_factor = metrics['queue_size'] / self.config.target_queue_size
            scale_factors.append(('queue', queue_factor))
        
        # Response time-based scaling
        if self.config.policy in [ScalingPolicy.RESPONSE_TIME_BASED, ScalingPolicy.HYBRID]:
            if metrics['avg_response_time'] > 0:
                response_factor = metrics['avg_response_time'] / self.config.target_response_time_ms
                scale_factors.append(('response_time', response_factor))
        
        if not scale_factors:
            return 0
        
        # Calculate overall scaling need
        max_factor = max(factor for _, factor in scale_factors)
        min_factor = min(factor for _, factor in scale_factors)
        
        # Determine scaling direction
        if max_factor > self.config.scale_up_threshold:
            # Need to scale up
            if self.current_workers < self.config.max_workers:
                return 1
        elif max_factor < self.config.scale_down_threshold:
            # Can scale down
            if self.current_workers > self.config.min_workers:
                return -1
        
        return 0
    
    def _execute_scaling_decision(self, decision: int, metrics: Dict[str, Any]) -> None:
        """Execute scaling decision.
        
        Args:
            decision: Scaling decision (1 for up, -1 for down)
            metrics: Current metrics
        """
        if decision > 0:
            # Scale up
            new_worker_count = min(self.current_workers + 1, self.config.max_workers)
            action = "scale_up"
        else:
            # Scale down
            new_worker_count = max(self.current_workers - 1, self.config.min_workers)
            action = "scale_down"
        
        if new_worker_count != self.current_workers:
            self.logger.info(f"Scaling {action}: {self.current_workers} -> {new_worker_count} workers")
            
            # Record scaling event
            scaling_event = {
                'timestamp': time.time(),
                'action': action,
                'old_count': self.current_workers,
                'new_count': new_worker_count,
                'trigger_metrics': metrics.copy()
            }
            
            self.scaling_history.append(scaling_event)
            
            # Keep only recent history
            if len(self.scaling_history) > 50:
                self.scaling_history.pop(0)
            
            # Update worker count
            self.current_workers = new_worker_count
            self.last_scale_time = time.time()
            
            # Trigger actual worker scaling (would be implemented by specific deployment)
            self._trigger_worker_scaling(action, new_worker_count)
    
    def _trigger_worker_scaling(self, action: str, target_count: int) -> None:
        """Trigger actual worker scaling.
        
        This would be implemented differently depending on deployment:
        - Kubernetes: Update deployment replicas
        - Docker Swarm: Update service replicas
        - Cloud: Auto Scaling Groups
        - Local: Process management
        
        Args:
            action: Scaling action
            target_count: Target worker count
        """
        # Placeholder for deployment-specific scaling
        self.logger.info(f"Triggering {action} to {target_count} workers")
        
        # For now, just update load balancer expectations
        # In real implementation, this would create/destroy actual workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and history."""
        recent_events = self.scaling_history[-10:] if self.scaling_history else []
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'scaling_policy': self.config.policy.value,
            'last_scale_time': self.last_scale_time,
            'recent_scaling_events': recent_events,
            'recent_metrics': recent_metrics,
            'total_scaling_events': len(self.scaling_history)
        }


class ResourcePool:
    """Dynamic resource pool with auto-scaling capabilities."""
    
    def __init__(self, resource_factory: Callable[[], Any], 
                 initial_size: int = 2, max_size: int = 10):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        self.resource_factory = resource_factory
        self.max_size = max_size
        
        self.available_resources = queue.Queue()
        self.allocated_resources = set()
        self.total_created = 0
        self._lock = threading.Lock()
        
        self.logger = get_logger('scaling.resource_pool')
        
        # Pre-populate pool
        for _ in range(initial_size):
            resource = self._create_resource()
            self.available_resources.put(resource)
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        with self._lock:
            if self.total_created >= self.max_size:
                raise RuntimeError("Resource pool at maximum capacity")
            
            resource = self.resource_factory()
            self.total_created += 1
            
            self.logger.debug(f"Created new resource (total: {self.total_created})")
            return resource
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Acquire a resource from the pool.
        
        Args:
            timeout: Maximum time to wait for resource
            
        Returns:
            Resource or None if timeout
        """
        try:
            # Try to get existing resource
            resource = self.available_resources.get(timeout=0.1)
        except queue.Empty:
            # Create new resource if under limit
            try:
                resource = self._create_resource()
            except RuntimeError:
                # Pool at capacity, wait for available resource
                try:
                    resource = self.available_resources.get(timeout=timeout)
                except queue.Empty:
                    return None
        
        # Track allocated resource
        with self._lock:
            self.allocated_resources.add(id(resource))
        
        return resource
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool.
        
        Args:
            resource: Resource to release
        """
        resource_id = id(resource)
        
        with self._lock:
            if resource_id in self.allocated_resources:
                self.allocated_resources.remove(resource_id)
                self.available_resources.put(resource)
            else:
                self.logger.warning("Attempted to release untracked resource")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            return {
                'total_created': self.total_created,
                'available': self.available_resources.qsize(),
                'allocated': len(self.allocated_resources),
                'max_size': self.max_size,
                'utilization_percent': (len(self.allocated_resources) / self.total_created * 100) 
                                     if self.total_created > 0 else 0
            }


# Global instances
default_load_balancer = LoadBalancer()
default_scaling_config = ScalingConfig()
default_auto_scaler = AutoScaler(default_scaling_config, default_load_balancer)


@handle_errors("distributed_processing", reraise=True)
def distribute_work(tasks: List[Any], process_func: Callable,
                   load_balancer: Optional[LoadBalancer] = None) -> List[Any]:
    """Distribute work across available workers.
    
    Args:
        tasks: Tasks to distribute
        process_func: Function to process each task
        load_balancer: Load balancer to use
        
    Returns:
        List of results
    """
    if load_balancer is None:
        load_balancer = default_load_balancer
    
    results = []
    
    for task in tasks:
        worker = load_balancer.select_worker()
        
        if worker:
            start_time = time.time()
            try:
                result = process_func(task)
                success = True
            except Exception as e:
                result = None
                success = False
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            load_balancer.record_request(worker.node_id, response_time, success)
            
            results.append(result)
        else:
            # No workers available, process locally
            results.append(process_func(task))
    
    return results