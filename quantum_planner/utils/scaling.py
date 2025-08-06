"""Advanced scaling and optimization utilities for quantum task planner."""

import logging
import multiprocessing as mp
import concurrent.futures
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import time
import threading
from dataclasses import dataclass
import queue
import numpy as np
from functools import partial
import psutil

from ..core.task import Task, TaskStatus
from ..core.scheduler import Resource, SchedulingResult
from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for scaling operations."""
    max_workers: int = None  # Auto-detect based on CPU cores
    chunk_size: int = 10
    enable_multiprocessing: bool = True
    enable_async: bool = True
    memory_threshold_gb: float = 4.0
    cpu_threshold_percent: float = 80.0
    adaptive_scaling: bool = True
    load_balancing: bool = True


class WorkerPool:
    """Managed worker pool for parallel processing."""
    
    def __init__(self, config: ScalingConfig):
        """Initialize worker pool."""
        self.config = config
        self.max_workers = config.max_workers or min(32, mp.cpu_count() + 4)
        
        # Resource monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Worker management
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.active_workers = 0
        self.task_queue = queue.Queue()
        
        # Adaptive scaling
        self.scaling_history = []
        self.last_optimization = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized WorkerPool with {self.max_workers} max workers")
    
    def __enter__(self):
        """Context manager entry."""
        self._initialize_pools()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def _initialize_pools(self):
        """Initialize worker pools."""
        if self.config.enable_multiprocessing:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers // 2
            )
        
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        
        logger.debug("Worker pools initialized")
    
    def submit_cpu_intensive(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit CPU-intensive task to process pool."""
        if not self.process_pool:
            raise RuntimeError("Process pool not initialized")
        
        with self._lock:
            self.active_workers += 1
        
        future = self.process_pool.submit(func, *args, **kwargs)
        future.add_done_callback(self._task_completed)
        
        return future
    
    def submit_io_intensive(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit I/O-intensive task to thread pool."""
        if not self.thread_pool:
            raise RuntimeError("Thread pool not initialized")
        
        with self._lock:
            self.active_workers += 1
        
        future = self.thread_pool.submit(func, *args, **kwargs)
        future.add_done_callback(self._task_completed)
        
        return future
    
    def map_parallel(self, func: Callable, iterable: List[Any], 
                    cpu_intensive: bool = True, chunk_size: Optional[int] = None) -> List[Any]:
        """Map function over iterable in parallel."""
        chunk_size = chunk_size or self.config.chunk_size
        
        if cpu_intensive and self.process_pool:
            executor = self.process_pool
        else:
            executor = self.thread_pool
        
        if not executor:
            # Fallback to sequential processing
            logger.warning("No executor available, falling back to sequential processing")
            return [func(item) for item in iterable]
        
        # Process in chunks for better memory management
        results = []
        for i in range(0, len(iterable), chunk_size):
            chunk = iterable[i:i + chunk_size]
            chunk_futures = [executor.submit(func, item) for item in chunk]
            chunk_results = [future.result() for future in chunk_futures]
            results.extend(chunk_results)
        
        return results
    
    def get_optimal_chunk_size(self, total_items: int, task_complexity: str = "medium") -> int:
        """Calculate optimal chunk size based on system resources."""
        # Base chunk size on CPU cores and task complexity
        base_chunk = max(1, total_items // (self.max_workers * 2))
        
        complexity_multipliers = {
            "simple": 4,
            "medium": 2,
            "complex": 1,
            "very_complex": 1
        }
        
        multiplier = complexity_multipliers.get(task_complexity, 2)
        optimal_chunk = min(base_chunk * multiplier, total_items)
        
        # Adaptive adjustment based on memory
        system_metrics = self.performance_monitor.get_system_metrics()
        memory_usage = system_metrics.get('memory_percent', 0)
        
        if memory_usage > 80:
            optimal_chunk = max(1, optimal_chunk // 2)
        elif memory_usage < 50:
            optimal_chunk = min(total_items, optimal_chunk * 2)
        
        return max(1, optimal_chunk)
    
    def _task_completed(self, future: concurrent.futures.Future):
        """Handle task completion."""
        with self._lock:
            self.active_workers = max(0, self.active_workers - 1)
        
        # Check for adaptive scaling
        if self.config.adaptive_scaling:
            self._check_adaptive_scaling()
    
    def _check_adaptive_scaling(self):
        """Check if adaptive scaling adjustments are needed."""
        current_time = time.time()
        if current_time - self.last_optimization < 30:  # Don't optimize too frequently
            return
        
        system_metrics = self.performance_monitor.get_system_metrics()
        memory_percent = system_metrics.get('memory_percent', 0)
        cpu_percent = system_metrics.get('cpu_percent', 0)
        
        # Record scaling decision
        scaling_decision = {
            'timestamp': current_time,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'active_workers': self.active_workers,
            'action': 'none'
        }
        
        # Adaptive scaling logic
        if memory_percent > self.config.memory_threshold_gb * 25:  # Rough conversion
            # Reduce workers to save memory
            if self.max_workers > 2:
                self.max_workers = max(2, self.max_workers - 1)
                scaling_decision['action'] = 'scale_down_memory'
                logger.info(f"Scaled down workers to {self.max_workers} due to memory pressure")
        
        elif cpu_percent > self.config.cpu_threshold_percent:
            # CPU is busy, don't add more workers
            pass
        
        elif (memory_percent < 50 and cpu_percent < 60 and 
              self.active_workers >= self.max_workers * 0.8):
            # System has capacity, can scale up
            if self.max_workers < mp.cpu_count() * 2:
                self.max_workers += 1
                scaling_decision['action'] = 'scale_up'
                logger.info(f"Scaled up workers to {self.max_workers}")
        
        self.scaling_history.append(scaling_decision)
        self.last_optimization = current_time
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history.pop(0)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling and performance metrics."""
        system_metrics = self.performance_monitor.get_system_metrics()
        
        return {
            'max_workers': self.max_workers,
            'active_workers': self.active_workers,
            'utilization_percent': (self.active_workers / self.max_workers * 100) if self.max_workers > 0 else 0,
            'system_memory_percent': system_metrics.get('memory_percent', 0),
            'system_cpu_percent': system_metrics.get('cpu_percent', 0),
            'scaling_events': len(self.scaling_history),
            'last_scaling_decision': self.scaling_history[-1] if self.scaling_history else None
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pools."""
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)
            self.process_pool = None
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)
            self.thread_pool = None
        
        logger.info("Worker pools shut down")


class DistributedTaskProcessor:
    """Process tasks in a distributed manner for better scaling."""
    
    def __init__(self, config: ScalingConfig):
        """Initialize distributed processor."""
        self.config = config
        self.worker_pool = WorkerPool(config)
        
        # Load balancing
        self.task_queues = [queue.Queue() for _ in range(config.max_workers or 4)]
        self.queue_index = 0
        self.load_balancer_lock = threading.Lock()
        
        # Caching for frequently accessed data
        self.cache = {}
        self.cache_lock = threading.RLock()
        
        logger.info("Initialized DistributedTaskProcessor")
    
    def process_tasks_batch(self, tasks: List[Task], 
                           processing_func: Callable[[Task], Any]) -> List[Any]:
        """Process a batch of tasks efficiently."""
        if not tasks:
            return []
        
        # Determine optimal processing strategy
        strategy = self._determine_processing_strategy(tasks, processing_func)
        
        logger.info(f"Processing {len(tasks)} tasks using strategy: {strategy}")
        
        with self.worker_pool:
            if strategy == "sequential":
                return self._process_sequential(tasks, processing_func)
            elif strategy == "parallel_threads":
                return self._process_parallel_threads(tasks, processing_func)
            elif strategy == "parallel_processes":
                return self._process_parallel_processes(tasks, processing_func)
            elif strategy == "hybrid":
                return self._process_hybrid(tasks, processing_func)
            else:
                # Fallback
                return self._process_sequential(tasks, processing_func)
    
    def process_with_load_balancing(self, tasks: List[Task],
                                  processing_func: Callable[[Task], Any]) -> List[Any]:
        """Process tasks with intelligent load balancing."""
        if not tasks:
            return []
        
        # Distribute tasks across queues based on complexity/priority
        self._distribute_tasks(tasks)
        
        # Process queues in parallel
        futures = []
        with self.worker_pool:
            for i, task_queue in enumerate(self.task_queues):
                if not task_queue.empty():
                    queue_tasks = []
                    while not task_queue.empty():
                        try:
                            queue_tasks.append(task_queue.get_nowait())
                        except queue.Empty:
                            break
                    
                    if queue_tasks:
                        future = self.worker_pool.submit_cpu_intensive(
                            self._process_task_batch,
                            queue_tasks, processing_func
                        )
                        futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
        
        return results
    
    def _determine_processing_strategy(self, tasks: List[Task],
                                     processing_func: Callable) -> str:
        """Determine optimal processing strategy."""
        num_tasks = len(tasks)
        system_metrics = self.worker_pool.performance_monitor.get_system_metrics()
        
        memory_percent = system_metrics.get('memory_percent', 0)
        cpu_percent = system_metrics.get('cpu_percent', 0)
        cpu_count = system_metrics.get('cpu_count', 1)
        
        # Strategy decision logic
        if num_tasks < 5:
            return "sequential"
        
        if memory_percent > 85:
            return "sequential"  # Memory pressure
        
        if cpu_count <= 2 or num_tasks < 10:
            return "parallel_threads"
        
        if num_tasks > 100 and cpu_count > 4:
            return "hybrid"
        
        # Estimate task complexity
        avg_complexity = self._estimate_task_complexity(tasks)
        
        if avg_complexity > 0.7:  # High complexity
            return "parallel_processes"
        else:
            return "parallel_threads"
    
    def _estimate_task_complexity(self, tasks: List[Task]) -> float:
        """Estimate average task complexity."""
        if not tasks:
            return 0.0
        
        complexity_score = 0.0
        for task in tasks:
            score = 0.1  # Base score
            
            # Duration factor
            if task.estimated_duration > 60:
                score += 0.2
            if task.estimated_duration > 300:  # 5 hours
                score += 0.3
            
            # Dependency factor
            score += min(0.3, len(task.dependencies) * 0.05)
            
            # Resource factor
            score += min(0.2, len(task.required_resources) * 0.05)
            
            # Priority factor (higher priority = higher complexity for scheduling)
            score += task.priority.weight * 0.1
            
            complexity_score += score
        
        return complexity_score / len(tasks)
    
    def _process_sequential(self, tasks: List[Task], 
                           processing_func: Callable) -> List[Any]:
        """Process tasks sequentially."""
        results = []
        for task in tasks:
            result = processing_func(task)
            results.append(result)
        return results
    
    def _process_parallel_threads(self, tasks: List[Task],
                                 processing_func: Callable) -> List[Any]:
        """Process tasks using thread parallelism."""
        optimal_chunk = self.worker_pool.get_optimal_chunk_size(len(tasks))
        return self.worker_pool.map_parallel(processing_func, tasks, 
                                           cpu_intensive=False, chunk_size=optimal_chunk)
    
    def _process_parallel_processes(self, tasks: List[Task],
                                   processing_func: Callable) -> List[Any]:
        """Process tasks using process parallelism."""
        optimal_chunk = self.worker_pool.get_optimal_chunk_size(len(tasks), "complex")
        return self.worker_pool.map_parallel(processing_func, tasks,
                                           cpu_intensive=True, chunk_size=optimal_chunk)
    
    def _process_hybrid(self, tasks: List[Task], 
                       processing_func: Callable) -> List[Any]:
        """Process tasks using hybrid approach."""
        # Split tasks into CPU-intensive and I/O-intensive
        cpu_tasks = []
        io_tasks = []
        
        for task in tasks:
            if (task.estimated_duration > 60 or 
                len(task.dependencies) > 3 or
                len(task.required_resources) > 2):
                cpu_tasks.append(task)
            else:
                io_tasks.append(task)
        
        results = []
        
        # Process in parallel
        futures = []
        if cpu_tasks:
            future = self.worker_pool.submit_cpu_intensive(
                self._process_task_batch, cpu_tasks, processing_func
            )
            futures.append(future)
        
        if io_tasks:
            future = self.worker_pool.submit_io_intensive(
                self._process_task_batch, io_tasks, processing_func
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
        
        return results
    
    def _process_task_batch(self, tasks: List[Task], 
                           processing_func: Callable) -> List[Any]:
        """Process a batch of tasks."""
        results = []
        for task in tasks:
            try:
                result = processing_func(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing task {task.id}: {e}")
                results.append(None)  # Or appropriate error handling
        
        return results
    
    def _distribute_tasks(self, tasks: List[Task]):
        """Distribute tasks across queues for load balancing."""
        # Clear existing queues
        for q in self.task_queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        # Sort tasks by priority and complexity
        sorted_tasks = sorted(tasks, key=lambda t: (
            t.priority.weight,
            t.estimated_duration,
            len(t.dependencies)
        ), reverse=True)
        
        # Round-robin distribution with load balancing
        queue_loads = [0] * len(self.task_queues)
        
        for task in sorted_tasks:
            # Find queue with minimum load
            min_load_idx = min(range(len(queue_loads)), key=lambda i: queue_loads[i])
            
            self.task_queues[min_load_idx].put(task)
            queue_loads[min_load_idx] += task.estimated_duration
    
    def optimize_memory_usage(self):
        """Optimize memory usage for large-scale operations."""
        import gc
        
        # Clear cache if it's getting large
        with self.cache_lock:
            if len(self.cache) > 1000:
                # Keep only recent entries
                sorted_items = sorted(self.cache.items(), 
                                    key=lambda x: x[1].get('timestamp', 0), reverse=True)
                self.cache = dict(sorted_items[:500])
                logger.debug("Optimized cache size")
        
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        return collected
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        worker_metrics = self.worker_pool.get_scaling_metrics()
        
        # Queue status
        queue_sizes = [q.qsize() for q in self.task_queues]
        
        # Cache metrics
        with self.cache_lock:
            cache_size = len(self.cache)
        
        return {
            'worker_metrics': worker_metrics,
            'queue_sizes': queue_sizes,
            'total_queued_tasks': sum(queue_sizes),
            'cache_size': cache_size,
            'num_queues': len(self.task_queues)
        }


class AsyncTaskProcessor:
    """Asynchronous task processor for I/O-bound operations."""
    
    def __init__(self, config: ScalingConfig):
        """Initialize async processor."""
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_workers or 50)
        
        logger.info("Initialized AsyncTaskProcessor")
    
    async def process_tasks_async(self, tasks: List[Task],
                                 async_processing_func: Callable) -> List[Any]:
        """Process tasks asynchronously."""
        async def process_single_task(task):
            async with self.semaphore:
                try:
                    return await async_processing_func(task)
                except Exception as e:
                    logger.error(f"Error processing task {task.id} async: {e}")
                    return None
        
        # Create tasks and wait for completion
        task_coroutines = [process_single_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        return results
    
    async def process_with_batching(self, tasks: List[Task],
                                   async_processing_func: Callable,
                                   batch_size: int = 20) -> List[Any]:
        """Process tasks in batches asynchronously."""
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self.process_tasks_async(batch, async_processing_func)
            all_results.extend(batch_results)
            
            # Small delay between batches to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        return all_results


class ScalingOrchestrator:
    """Orchestrate scaling operations across different processing strategies."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        """Initialize scaling orchestrator."""
        self.config = config or ScalingConfig()
        
        # Initialize processors
        self.distributed_processor = DistributedTaskProcessor(self.config)
        self.async_processor = AsyncTaskProcessor(self.config)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.start_monitoring()
        
        logger.info("Initialized ScalingOrchestrator")
    
    def scale_task_processing(self, tasks: List[Task],
                            processing_func: Callable,
                            async_processing_func: Optional[Callable] = None) -> List[Any]:
        """Scale task processing using optimal strategy."""
        if not tasks:
            return []
        
        start_time = time.time()
        strategy = self._select_optimal_strategy(tasks, processing_func, async_processing_func)
        
        logger.info(f"Scaling {len(tasks)} tasks with strategy: {strategy}")
        
        try:
            if strategy == "async" and async_processing_func:
                # Use async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.async_processor.process_tasks_async(tasks, async_processing_func)
                    )
                finally:
                    loop.close()
            else:
                # Use distributed processing
                results = self.distributed_processor.process_tasks_batch(tasks, processing_func)
            
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_monitor.record_metric(
                "task_processing_time", processing_time,
                strategy=strategy, task_count=len(tasks)
            )
            
            logger.info(f"Processed {len(tasks)} tasks in {processing_time:.2f}s using {strategy}")
            return results
            
        except Exception as e:
            logger.error(f"Error in scaled processing: {e}")
            # Fallback to sequential processing
            return [processing_func(task) for task in tasks]
    
    def scale_schedule_optimization(self, tasks: List[Task], 
                                  resources: Dict[str, Any],
                                  optimization_func: Callable) -> SchedulingResult:
        """Scale schedule optimization operations."""
        # For very large problem sizes, use problem decomposition
        if len(tasks) > 1000:
            return self._decompose_and_optimize(tasks, resources, optimization_func)
        else:
            return optimization_func(tasks, resources)
    
    def _select_optimal_strategy(self, tasks: List[Task],
                               processing_func: Callable,
                               async_processing_func: Optional[Callable]) -> str:
        """Select optimal processing strategy."""
        system_metrics = self.performance_monitor.get_system_metrics()
        
        num_tasks = len(tasks)
        memory_percent = system_metrics.get('memory_percent', 0)
        cpu_percent = system_metrics.get('cpu_percent', 0)
        
        # I/O bound detection heuristics
        is_io_bound = (
            async_processing_func is not None and
            (cpu_percent < 50 or memory_percent < 60) and
            num_tasks > 20
        )
        
        if is_io_bound:
            return "async"
        else:
            return "distributed"
    
    def _decompose_and_optimize(self, tasks: List[Task],
                              resources: Dict[str, Any],
                              optimization_func: Callable) -> SchedulingResult:
        """Decompose large problems for scalable optimization."""
        # Group tasks by priority and dependencies
        task_groups = self._group_tasks_for_decomposition(tasks)
        
        # Optimize each group separately
        group_results = []
        
        with self.distributed_processor.worker_pool:
            futures = []
            for group in task_groups:
                if len(group) > 0:
                    future = self.distributed_processor.worker_pool.submit_cpu_intensive(
                        optimization_func, group, resources
                    )
                    futures.append(future)
            
            for future in futures:
                result = future.result()
                group_results.append(result)
        
        # Merge results
        return self._merge_optimization_results(group_results)
    
    def _group_tasks_for_decomposition(self, tasks: List[Task]) -> List[List[Task]]:
        """Group tasks for problem decomposition."""
        # Simple grouping by priority - could be more sophisticated
        groups = {}
        
        for task in tasks:
            priority_key = task.priority.name
            if priority_key not in groups:
                groups[priority_key] = []
            groups[priority_key].append(task)
        
        # Convert to list of groups
        task_groups = list(groups.values())
        
        # Ensure no group is too large
        final_groups = []
        max_group_size = 200
        
        for group in task_groups:
            if len(group) <= max_group_size:
                final_groups.append(group)
            else:
                # Split large groups
                for i in range(0, len(group), max_group_size):
                    final_groups.append(group[i:i + max_group_size])
        
        return final_groups
    
    def _merge_optimization_results(self, results: List[SchedulingResult]) -> SchedulingResult:
        """Merge multiple optimization results."""
        if not results:
            return SchedulingResult(
                schedule={}, total_completion_time=0, resource_utilization={},
                optimization_metrics={}, quantum_metrics={}, 
                conflicts_resolved=0, dependencies_satisfied=0,
                success=False, error_message="No results to merge"
            )
        
        # Merge schedules
        merged_schedule = {}
        total_completion_time = 0
        merged_resource_utilization = {}
        merged_optimization_metrics = {}
        merged_quantum_metrics = {}
        total_conflicts_resolved = 0
        total_dependencies_satisfied = 0
        
        for result in results:
            if result.success:
                merged_schedule.update(result.schedule)
                total_completion_time = max(total_completion_time, result.total_completion_time)
                
                # Merge resource utilization (average)
                for resource_id, utilization in result.resource_utilization.items():
                    if resource_id in merged_resource_utilization:
                        merged_resource_utilization[resource_id] = (
                            merged_resource_utilization[resource_id] + utilization
                        ) / 2
                    else:
                        merged_resource_utilization[resource_id] = utilization
                
                # Sum metrics
                total_conflicts_resolved += result.conflicts_resolved
                total_dependencies_satisfied += result.dependencies_satisfied
        
        return SchedulingResult(
            schedule=merged_schedule,
            total_completion_time=total_completion_time,
            resource_utilization=merged_resource_utilization,
            optimization_metrics=merged_optimization_metrics,
            quantum_metrics=merged_quantum_metrics,
            conflicts_resolved=total_conflicts_resolved,
            dependencies_satisfied=total_dependencies_satisfied,
            success=len(merged_schedule) > 0,
            error_message="" if len(merged_schedule) > 0 else "No successful optimizations"
        )
    
    def optimize_system_resources(self) -> Dict[str, Any]:
        """Optimize system resources for better scaling."""
        optimization_results = {}
        
        # Memory optimization
        memory_freed = self.distributed_processor.optimize_memory_usage()
        optimization_results['memory_freed_objects'] = memory_freed
        
        # Worker pool optimization
        worker_metrics = self.distributed_processor.worker_pool.get_scaling_metrics()
        optimization_results['worker_metrics'] = worker_metrics
        
        # System metrics
        system_metrics = self.performance_monitor.get_system_metrics()
        optimization_results['system_metrics'] = system_metrics
        
        return optimization_results
    
    def get_scaling_recommendations(self) -> List[str]:
        """Get recommendations for improving scaling performance."""
        recommendations = []
        
        system_metrics = self.performance_monitor.get_system_metrics()
        worker_metrics = self.distributed_processor.worker_pool.get_scaling_metrics()
        
        memory_percent = system_metrics.get('memory_percent', 0)
        cpu_percent = system_metrics.get('cpu_percent', 0)
        worker_utilization = worker_metrics.get('utilization_percent', 0)
        
        if memory_percent > 85:
            recommendations.append("Consider reducing batch sizes or implementing memory streaming")
        
        if cpu_percent > 90:
            recommendations.append("CPU is at capacity - consider horizontal scaling or optimize algorithms")
        
        if worker_utilization < 30:
            recommendations.append("Worker pool is under-utilized - consider reducing max workers")
        
        if worker_utilization > 90:
            recommendations.append("Worker pool is over-utilized - consider increasing max workers")
        
        if len(recommendations) == 0:
            recommendations.append("System scaling appears optimal")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown scaling orchestrator."""
        self.performance_monitor.stop_monitoring()
        self.distributed_processor.worker_pool.shutdown()
        logger.info("Scaling orchestrator shut down")