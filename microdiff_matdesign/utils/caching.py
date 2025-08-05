"""Advanced caching system for MicroDiff-MatDesign."""

import time
import threading
import hashlib
import pickle
from typing import Any, Dict, Optional, Callable, Tuple, Union
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger
from .error_handling import MicroDiffError, handle_errors


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


class MemoryCache:
    """High-performance in-memory cache with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0,
                 policy: CachePolicy = CachePolicy.LRU, default_ttl: Optional[float] = None):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            policy: Cache eviction policy
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl = default_ttl
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_bytes': 0
        }
        
        self.logger = get_logger('cache.memory')
        self.logger.info(f"Initialized memory cache: {max_size} entries, {max_memory_mb}MB, {policy.value} policy")
    
    def _compute_key(self, key: Union[str, bytes, tuple]) -> str:
        """Compute cache key hash."""
        if isinstance(key, str):
            return key
        elif isinstance(key, bytes):
            return hashlib.sha256(key).hexdigest()
        else:
            # Hash complex objects
            key_str = str(key)
            return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            return len(str(obj)) * 2
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return (len(self._cache) >= self.max_size or 
                self._stats['memory_bytes'] >= self.max_memory_bytes)
    
    def _evict_entries(self) -> None:
        """Evict entries based on policy."""
        if not self._cache:
            return
        
        current_time = time.time()
        
        # First, remove expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self._stats['evictions'] += 1
        
        # If still need to evict, use policy
        while self._should_evict() and self._cache:
            if self.policy == CachePolicy.LRU:
                # Remove least recently used
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].last_accessed)
            elif self.policy == CachePolicy.LFU:
                # Remove least frequently used
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k].access_count)
            elif self.policy == CachePolicy.TTL:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k].created_at)
            else:  # FIFO
                # Remove first inserted (oldest)
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k].created_at)
            
            self._remove_entry(oldest_key)
            self._stats['evictions'] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats['memory_bytes'] -= entry.size_bytes
            del self._cache[key]
    
    def get(self, key: Union[str, bytes, tuple], default: Any = None) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_key = self._compute_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(cache_key)
                    self._stats['misses'] += 1
                    return default
                
                # Update access statistics
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                self._stats['hits'] += 1
                return entry.value
            else:
                self._stats['misses'] += 1
                return default
    
    def put(self, key: Union[str, bytes, tuple], value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live override
        """
        cache_key = self._compute_key(key)
        current_time = time.time()
        
        # Use provided TTL or default
        entry_ttl = ttl if ttl is not None else self.default_ttl
        
        with self._lock:
            # Create entry
            size_bytes = self._estimate_size(value)
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=entry_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if cache_key in self._cache:
                self._remove_entry(cache_key)
            
            # Check if we need to evict
            if self._should_evict():
                self._evict_entries()
            
            # Add new entry
            self._cache[cache_key] = entry
            self._stats['memory_bytes'] += size_bytes
    
    def delete(self, key: Union[str, bytes, tuple]) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        cache_key = self._compute_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats['memory_bytes'] = 0
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'max_entries': self.max_size,
                'memory_bytes': self._stats['memory_bytes'],
                'memory_mb': self._stats['memory_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate_percent': hit_rate,
                'evictions': self._stats['evictions'],
                'policy': self.policy.value
            }
    
    def get_size_distribution(self) -> Dict[str, Any]:
        """Get size distribution of cached entries."""
        with self._lock:
            if not self._cache:
                return {}
            
            sizes = [entry.size_bytes for entry in self._cache.values()]
            
            return {
                'total_entries': len(sizes),
                'min_size_bytes': min(sizes),
                'max_size_bytes': max(sizes),
                'avg_size_bytes': sum(sizes) / len(sizes),
                'total_size_bytes': sum(sizes)
            }


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str = ".cache", max_size_gb: float = 1.0):
        """Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum disk usage in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        self._lock = threading.RLock()
        self.logger = get_logger('cache.disk')
        self.logger.info(f"Initialized disk cache: {cache_dir}, {max_size_gb}GB")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_current_size(self) -> int:
        """Get current cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                total_size += cache_file.stat().st_size
            except (OSError, FileNotFoundError):
                pass
        return total_size
    
    def _cleanup_old_files(self) -> None:
        """Remove oldest files if over size limit."""
        current_size = self._get_current_size()
        
        if current_size <= self.max_size_bytes:
            return
        
        # Get all cache files with modification times
        cache_files = []
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                stat_info = cache_file.stat()
                cache_files.append((cache_file, stat_info.st_mtime, stat_info.st_size))
            except (OSError, FileNotFoundError):
                pass
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove files until under limit
        for cache_file, _, file_size in cache_files:
            try:
                cache_file.unlink()
                current_size -= file_size
                self.logger.debug(f"Removed old cache file: {cache_file.name}")
                
                if current_size <= self.max_size_bytes:
                    break
            except (OSError, FileNotFoundError):
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from disk cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Update access time
                    cache_path.touch()
                    
                    return data
                else:
                    return default
            except Exception as e:
                self.logger.warning(f"Failed to read cache file {cache_path}: {e}")
                return default
    
    def put(self, key: str, value: Any) -> None:
        """Put value in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            try:
                # Clean up if needed
                self._cleanup_old_files()
                
                # Write to temporary file first
                temp_path = cache_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Atomic move
                temp_path.rename(cache_path)
                
            except Exception as e:
                self.logger.error(f"Failed to write cache file {cache_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            try:
                if cache_path.exists():
                    cache_path.unlink()
                    return True
                else:
                    return False
            except Exception as e:
                self.logger.warning(f"Failed to delete cache file {cache_path}: {e}")
                return False
    
    def clear(self) -> None:
        """Clear all disk cache entries."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                self.logger.info("Disk cache cleared")
            except Exception as e:
                self.logger.error(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        with self._lock:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            return {
                'entries': len(cache_files),
                'size_bytes': total_size,
                'size_mb': total_size / (1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }


class MultiLevelCache:
    """Multi-level cache with memory and disk tiers."""
    
    def __init__(self, memory_cache: Optional[MemoryCache] = None,
                 disk_cache: Optional[DiskCache] = None):
        """Initialize multi-level cache.
        
        Args:
            memory_cache: L1 memory cache
            disk_cache: L2 disk cache
        """
        self.memory_cache = memory_cache or MemoryCache()
        self.disk_cache = disk_cache or DiskCache()
        
        self.logger = get_logger('cache.multilevel')
        self.logger.info("Initialized multi-level cache")
    
    def get(self, key: Union[str, bytes, tuple], default: Any = None) -> Any:
        """Get value from multi-level cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_key = str(key)
        
        # Try L1 cache (memory) first
        value = self.memory_cache.get(key, None)
        if value is not None:
            return value
        
        # Try L2 cache (disk)
        value = self.disk_cache.get(cache_key, None)
        if value is not None:
            # Promote to L1 cache
            self.memory_cache.put(key, value)
            return value
        
        return default
    
    def put(self, key: Union[str, bytes, tuple], value: Any, ttl: Optional[float] = None) -> None:
        """Put value in multi-level cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live
        """
        cache_key = str(key)
        
        # Store in both levels
        self.memory_cache.put(key, value, ttl)
        self.disk_cache.put(cache_key, value)
    
    def delete(self, key: Union[str, bytes, tuple]) -> bool:
        """Delete from both cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted from any level
        """
        cache_key = str(key)
        
        deleted_memory = self.memory_cache.delete(key)
        deleted_disk = self.disk_cache.delete(cache_key)
        
        return deleted_memory or deleted_disk
    
    def clear(self) -> None:
        """Clear all cache levels."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'disk_cache': self.disk_cache.get_stats()
        }


# Global cache instances
default_memory_cache = MemoryCache()
default_disk_cache = DiskCache()
default_multilevel_cache = MultiLevelCache(default_memory_cache, default_disk_cache)


def cached(cache: Optional[Union[MemoryCache, DiskCache, MultiLevelCache]] = None,
          ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        cache: Cache instance to use
        ttl: Time-to-live for cached results
        key_func: Function to generate cache key
    """
    if cache is None:
        cache = default_multilevel_cache
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


@handle_errors("cache_warming", reraise=False)
def warm_cache(cache_func: Callable, param_sets: list, 
               cache: Optional[Union[MemoryCache, DiskCache, MultiLevelCache]] = None,
               max_workers: int = 4) -> None:
    """Warm cache with pre-computed results.
    
    Args:
        cache_func: Function to cache results for
        param_sets: List of parameter sets to pre-compute
        cache: Cache instance
        max_workers: Maximum number of worker threads
    """
    import concurrent.futures
    
    if cache is None:
        cache = default_multilevel_cache
    
    logger = get_logger('cache.warming')
    
    @cached(cache=cache)
    def cached_func(*args, **kwargs):
        return cache_func(*args, **kwargs)
    
    def compute_and_cache(params):
        try:
            if isinstance(params, dict):
                return cached_func(**params)
            else:
                return cached_func(*params)
        except Exception as e:
            logger.warning(f"Cache warming failed for params {params}: {e}")
            return None
    
    logger.info(f"Starting cache warming with {len(param_sets)} parameter sets")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_and_cache, params) for params in param_sets]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                logger.info(f"Cache warming progress: {completed}/{len(param_sets)}")
    
    logger.info("Cache warming completed")