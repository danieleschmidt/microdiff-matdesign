"""Performance optimization utilities for MicroDiff-MatDesign."""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from queue import Queue, Empty
import gc

from .logging_config import get_logger, with_logging
from .error_handling import handle_errors, safe_execute
from .monitoring import performance_tracker


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_multiprocessing: bool = True
    enable_multithreading: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 100
    enable_gc_optimization: bool = True
    memory_limit_mb: Optional[float] = None
    cpu_threshold_percent: float = 80.0


class ResourceManager:
    """Manage system resources and prevent overload."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = get_logger('performance.resources')
        self._active_workers = 0
        self._lock = threading.Lock()
        
        # Determine optimal worker counts
        cpu_count = multiprocessing.cpu_count()
        self.optimal_thread_workers = min(cpu_count * 2, 32)
        self.optimal_process_workers = cpu_count
        
        if config.max_workers:
            self.optimal_thread_workers = min(self.optimal_thread_workers, config.max_workers)
            self.optimal_process_workers = min(self.optimal_process_workers, config.max_workers)
    
    def get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_mb': memory_mb,
                'active_workers': self._active_workers,
                'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            }
        except ImportError:
            # Fallback without psutil
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_mb': 0.0,
                'active_workers': self._active_workers,
                'disk_read_mb': 0.0,
                'disk_write_mb': 0.0
            }
    
    def should_throttle(self) -> bool:
        """Check if processing should be throttled."""
        load = self.get_system_load()
        
        # Check CPU threshold
        if load['cpu_percent'] > self.config.cpu_threshold_percent:
            return True
        
        # Check memory limit
        if (self.config.memory_limit_mb and 
            load['memory_mb'] > self.config.memory_limit_mb):
            return True
        
        return False
    
    def acquire_worker(self) -> bool:
        """Try to acquire a worker slot."""
        with self._lock:
            if self.should_throttle():
                return False
            
            self._active_workers += 1
            return True
    
    def release_worker(self) -> None:
        """Release a worker slot."""
        with self._lock:
            self._active_workers = max(0, self._active_workers - 1)
    
    def get_optimal_workers(self, workload_type: str = "cpu") -> int:
        """Get optimal number of workers for workload type."""
        if workload_type == "io":
            return self.optimal_thread_workers
        elif workload_type == "cpu":
            return self.optimal_process_workers
        else:
            return max(1, min(self.optimal_thread_workers, self.optimal_process_workers))


class BatchProcessor:
    """Efficient batch processing with resource management."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.resource_manager = ResourceManager(config)
        self.logger = get_logger('performance.batch')
    
    @with_logging("batch_process_parallel")
    def process_batch_parallel(self, 
                             items: List[Any],
                             process_func: Callable[[Any], Any],
                             workload_type: str = "cpu",
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Process items in parallel batches.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            workload_type: Type of workload ('cpu', 'io', 'mixed')
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        # Determine execution strategy
        num_workers = self.resource_manager.get_optimal_workers(workload_type)
        use_processes = (workload_type == "cpu" and 
                        self.config.enable_multiprocessing and 
                        len(items) > 50)
        
        self.logger.info(f"Processing {len(items)} items with {num_workers} workers "
                        f"({'processes' if use_processes else 'threads'})")
        
        # Split items into chunks
        chunk_size = max(1, min(self.config.chunk_size, len(items) // num_workers))
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        results = []
        completed = 0
        
        try:
            if use_processes:
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor
            
            with executor_class(max_workers=num_workers) as executor:
                # Submit chunks
                future_to_chunk = {}
                for chunk in chunks:
                    if self.resource_manager.acquire_worker():
                        future = executor.submit(self._process_chunk, chunk, process_func)
                        future_to_chunk[future] = chunk
                    else:
                        # Process synchronously if throttled
                        chunk_results = self._process_chunk(chunk, process_func)
                        results.extend(chunk_results)
                        completed += len(chunk)
                
                # Collect results
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        completed += len(future_to_chunk[future])
                        
                        if progress_callback:
                            progress_callback(completed, len(items))
                    
                    except Exception as e:
                        self.logger.error(f"Chunk processing failed: {e}")
                        # Add None results for failed chunk
                        chunk_size = len(future_to_chunk[future])
                        results.extend([None] * chunk_size)
                        completed += chunk_size
                    
                    finally:
                        self.resource_manager.release_worker()
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
        
        # Garbage collection optimization
        if self.config.enable_gc_optimization:
            gc.collect()
        
        return results
    
    def _process_chunk(self, chunk: List[Any], process_func: Callable[[Any], Any]) -> List[Any]:
        """Process a chunk of items."""
        results = []
        for item in chunk:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Item processing failed: {e}")
                results.append(None)
        
        return results
    
    @with_logging("stream_process")
    def stream_process(self,
                      items: List[Any],
                      process_func: Callable[[Any], Any],
                      batch_size: int = 10,
                      yield_results: bool = True) -> Union[List[Any], None]:
        """Process items in streaming fashion with memory efficiency.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            batch_size: Size of processing batches
            yield_results: Whether to yield results or process in-place
            
        Returns:
            Results if yield_results=True, None otherwise
        """
        results = [] if yield_results else None
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Check if we should throttle
            if self.resource_manager.should_throttle():
                self.logger.info("Throttling due to high system load")
                time.sleep(0.1)
            
            # Process batch
            batch_results = []
            for item in batch:
                try:
                    result = process_func(item)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Streaming item processing failed: {e}")
                    batch_results.append(None)
            
            if yield_results:
                results.extend(batch_results)
            
            # Periodic garbage collection
            if i % (batch_size * 10) == 0 and self.config.enable_gc_optimization:
                gc.collect()
        
        return results


class AsyncTaskQueue:
    """Asynchronous task queue with priority and rate limiting."""
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        self.task_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        self.workers = []
        self.running = False
        
        self.logger = get_logger('performance.task_queue')
    
    def start(self) -> None:
        """Start the task queue workers."""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started task queue with {self.max_workers} workers")
    
    def stop(self) -> None:
        """Stop the task queue."""
        self.running = False
        
        # Add stop signals
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        self.logger.info("Task queue stopped")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the queue.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        import uuid
        
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        try:
            self.task_queue.put(task, timeout=1.0)
            return task_id
        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            raise
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a result from the queue.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Result dictionary or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Stop signal
                    break
                
                # Execute task
                start_time = time.time()
                
                try:
                    result = task['func'](*task['args'], **task['kwargs'])
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                duration = time.time() - start_time
                
                # Put result
                result_data = {
                    'task_id': task['id'],
                    'success': success,
                    'result': result,
                    'error': error,
                    'duration': duration,
                    'completed_at': time.time()
                }
                
                self.result_queue.put(result_data)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")


class PerformanceProfiler:
    """Profile and optimize performance bottlenecks."""
    
    def __init__(self):
        self.profiles = {}
        self.logger = get_logger('performance.profiler')
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profiling results
        """
        import cProfile
        import pstats
        import io
        
        # Profile execution
        profiler = cProfile.Profile()
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        profiler.enable()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            profiler.disable()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_text = s.getvalue()
        
        return {
            'success': success,
            'result': result,
            'error': error,
            'duration': end_time - start_time,
            'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
            'start_memory_mb': start_memory / (1024 * 1024),
            'end_memory_mb': end_memory / (1024 * 1024),
            'profile_text': profile_text
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0
    
    def benchmark_function(self, func: Callable, iterations: int = 100, 
                          *args, **kwargs) -> Dict[str, Any]:
        """Benchmark function performance.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        durations = []
        memory_usages = []
        
        for i in range(iterations):
            start_memory = self._get_memory_usage()
            start_time = time.time()
            
            try:
                func(*args, **kwargs)
                success = True
            except Exception:
                success = False
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            if success:
                durations.append(end_time - start_time)
                memory_usages.append(end_memory - start_memory)
        
        if durations:
            return {
                'iterations': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations),
                'operations_per_second': len(durations) / sum(durations),
                'avg_memory_delta_mb': sum(memory_usages) / len(memory_usages) / (1024 * 1024),
                'success_rate': len(durations) / iterations
            }
        else:
            return {'error': 'All iterations failed'}


# Global instances
default_config = PerformanceConfig()
default_batch_processor = BatchProcessor(default_config)
default_task_queue = AsyncTaskQueue()
default_profiler = PerformanceProfiler()


@handle_errors("parallel_map", reraise=True)
def parallel_map(func: Callable, items: List[Any], 
                workers: Optional[int] = None,
                workload_type: str = "cpu") -> List[Any]:
    """Parallel map operation with automatic optimization.
    
    Args:
        func: Function to apply to each item
        items: Items to process
        workers: Number of workers (auto-detected if None)
        workload_type: Type of workload for optimization
        
    Returns:
        List of results
    """
    return default_batch_processor.process_batch_parallel(
        items, func, workload_type
    )


def optimize_memory_usage():
    """Optimize memory usage by triggering garbage collection."""
    import gc
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory statistics
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'objects_collected': collected,
            'memory_rss_mb': memory_info.rss / (1024 * 1024),
            'memory_vms_mb': memory_info.vms / (1024 * 1024)
        }
    except ImportError:
        return {
            'objects_collected': collected
        }


def adaptive_batch_size(base_size: int, system_load: float, memory_usage: float) -> int:
    """Calculate adaptive batch size based on system conditions.
    
    Args:
        base_size: Base batch size
        system_load: Current system load (0.0-1.0)
        memory_usage: Current memory usage (0.0-1.0)
        
    Returns:
        Optimized batch size
    """
    # Reduce batch size under high load or memory pressure
    load_factor = max(0.1, 1.0 - system_load)
    memory_factor = max(0.1, 1.0 - memory_usage)
    
    adaptive_factor = min(load_factor, memory_factor)
    
    return max(1, int(base_size * adaptive_factor))