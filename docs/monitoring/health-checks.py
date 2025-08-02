"""
Health check endpoints for MicroDiff-MatDesign monitoring.

This module provides comprehensive health checking functionality
for application components, dependencies, and system resources.
"""

import json
import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class HealthStatus:
    """Health check status representation."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    details: Dict[str, Any]
    duration_ms: float


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
    
    def check_application_health(self) -> HealthStatus:
        """Check overall application health."""
        start = time.time()
        
        try:
            # Core application checks
            details = {
                "uptime_seconds": time.time() - self.start_time,
                "version": self._get_application_version(),
                "environment": self._get_environment(),
                "pid": self._get_process_id()
            }
            
            status = "healthy"
            
        except Exception as e:
            self.logger.error(f"Application health check failed: {e}")
            details = {"error": str(e)}
            status = "unhealthy"
        
        duration = (time.time() - start) * 1000
        return HealthStatus(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
            duration_ms=duration
        )
    
    def check_database_health(self) -> HealthStatus:
        """Check database connectivity and performance."""
        start = time.time()
        
        try:
            # Simulate database connection check
            # In real implementation, use your actual database client
            details = {
                "connection_pool_active": self._get_db_pool_stats(),
                "query_response_time_ms": self._measure_db_query_time(),
                "database_version": "PostgreSQL 15.2"
            }
            
            # Determine status based on response time
            response_time = details["query_response_time_ms"]
            if response_time < 100:
                status = "healthy"
            elif response_time < 500:
                status = "degraded"
            else:
                status = "unhealthy"
                
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            details = {"error": str(e)}
            status = "unhealthy"
        
        duration = (time.time() - start) * 1000
        return HealthStatus(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
            duration_ms=duration
        )
    
    def check_model_health(self) -> HealthStatus:
        """Check ML model loading and inference health."""
        start = time.time()
        
        try:
            # Simulate model health check
            details = {
                "models_loaded": self._get_loaded_models(),
                "gpu_available": self._check_gpu_availability(),
                "memory_usage_mb": self._get_model_memory_usage(),
                "last_inference_time": self._get_last_inference_time()
            }
            
            # Check if critical models are loaded
            if details["models_loaded"]["diffusion_model"]:
                status = "healthy"
            else:
                status = "unhealthy"
                
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            details = {"error": str(e)}
            status = "unhealthy"
        
        duration = (time.time() - start) * 1000
        return HealthStatus(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
            duration_ms=duration
        )
    
    def check_system_resources(self) -> HealthStatus:
        """Check system resource utilization."""
        start = time.time()
        
        try:
            # Get system resource information
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            details = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "load_average": list(psutil.getloadavg())
            }
            
            # Determine status based on resource usage
            if (cpu_percent < 80 and memory.percent < 80 and disk.percent < 90):
                status = "healthy"
            elif (cpu_percent < 95 and memory.percent < 95 and disk.percent < 95):
                status = "degraded"
            else:
                status = "unhealthy"
                
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            details = {"error": str(e)}
            status = "unhealthy"
        
        duration = (time.time() - start) * 1000
        return HealthStatus(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
            duration_ms=duration
        )
    
    def comprehensive_health_check(self) -> Dict[str, HealthStatus]:
        """Run all health checks and return combined results."""
        return {
            "application": self.check_application_health(),
            "database": self.check_database_health(),
            "models": self.check_model_health(),
            "system": self.check_system_resources()
        }
    
    def get_overall_status(self, health_results: Dict[str, HealthStatus]) -> str:
        """Determine overall system health from individual checks."""
        statuses = [check.status for check in health_results.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    # Helper methods (implement based on your actual infrastructure)
    
    def _get_application_version(self) -> str:
        """Get application version."""
        return "1.0.0"  # Replace with actual version retrieval
    
    def _get_environment(self) -> str:
        """Get deployment environment."""
        import os
        return os.getenv("ENVIRONMENT", "development")
    
    def _get_process_id(self) -> int:
        """Get current process ID."""
        import os
        return os.getpid()
    
    def _get_db_pool_stats(self) -> Dict[str, int]:
        """Get database connection pool statistics."""
        # Simulate connection pool stats
        return {
            "active_connections": 5,
            "idle_connections": 10,
            "total_connections": 15
        }
    
    def _measure_db_query_time(self) -> float:
        """Measure database query response time."""
        # Simulate database query timing
        import random
        return random.uniform(10, 150)  # milliseconds
    
    def _get_loaded_models(self) -> Dict[str, bool]:
        """Check which models are loaded."""
        # Simulate model loading status
        return {
            "diffusion_model": True,
            "encoder_model": True,
            "decoder_model": True
        }
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and status."""
        try:
            import torch
            return {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
            }
        except ImportError:
            return {"available": False, "error": "PyTorch not installed"}
    
    def _get_model_memory_usage(self) -> float:
        """Get model memory usage in MB."""
        # Simulate model memory usage
        return 2048.5  # MB
    
    def _get_last_inference_time(self) -> Optional[str]:
        """Get timestamp of last successful inference."""
        # Simulate last inference time
        return datetime.now(timezone.utc).isoformat()


# Flask/FastAPI endpoint examples

def create_health_endpoints():
    """Example health check endpoints for web frameworks."""
    
    health_checker = HealthChecker()
    
    # Flask example
    def flask_health_endpoint():
        """Flask health check endpoint."""
        try:
            health_results = health_checker.comprehensive_health_check()
            overall_status = health_checker.get_overall_status(health_results)
            
            response_data = {
                "status": overall_status,
                "checks": {name: asdict(check) for name, check in health_results.items()}
            }
            
            status_code = 200 if overall_status == "healthy" else 503
            return json.dumps(response_data), status_code, {'Content-Type': 'application/json'}
            
        except Exception as e:
            return json.dumps({
                "status": "unhealthy",
                "error": str(e)
            }), 503, {'Content-Type': 'application/json'}
    
    # FastAPI example
    async def fastapi_health_endpoint():
        """FastAPI health check endpoint."""
        try:
            health_results = health_checker.comprehensive_health_check()
            overall_status = health_checker.get_overall_status(health_results)
            
            response_data = {
                "status": overall_status,
                "checks": {name: asdict(check) for name, check in health_results.items()}
            }
            
            from fastapi import HTTPException
            if overall_status != "healthy":
                raise HTTPException(status_code=503, detail=response_data)
            
            return response_data
            
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail={
                "status": "unhealthy",
                "error": str(e)
            })
    
    return flask_health_endpoint, fastapi_health_endpoint


if __name__ == "__main__":
    # Example usage
    checker = HealthChecker()
    
    # Run comprehensive health check
    health_results = checker.comprehensive_health_check()
    overall_status = checker.get_overall_status(health_results)
    
    print(f"Overall Status: {overall_status}")
    print("\nDetailed Results:")
    for name, result in health_results.items():
        print(f"\n{name.upper()}:")
        print(f"  Status: {result.status}")
        print(f"  Duration: {result.duration_ms:.2f}ms")
        print(f"  Details: {json.dumps(result.details, indent=4)}")