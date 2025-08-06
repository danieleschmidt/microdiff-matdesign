# Quantum Task Planner - Deployment Guide

## Overview

The Quantum-Inspired Task Planner is a production-ready scheduling system that uses quantum principles for advanced task optimization. This guide covers deployment options and configuration.

## Architecture

```
quantum-task-planner/
├── quantum_planner/           # Core package
│   ├── core/                 # Core components (Task, Scheduler, Quantum Engine)
│   ├── algorithms/           # Quantum algorithms (Annealing, Superposition)
│   ├── utils/               # Utilities (Validation, Performance, Scaling)
│   └── main.py              # CLI interface
├── tests/                   # Comprehensive test suite
├── examples/                # Usage examples
└── docs/                   # Documentation
```

## Installation Options

### Basic Installation (Lightweight)
```bash
pip install quantum-task-planner
```
- Includes core functionality with minimal dependencies
- Task management, basic scheduling, validation
- Works with Python standard library only

### Full Installation (All Features)
```bash
pip install quantum-task-planner[full]
```
- Includes all advanced features
- Visualization, performance monitoring, advanced algorithms
- Requires numpy, matplotlib, psutil, etc.

### Custom Installation
```bash
# Visualization only
pip install quantum-task-planner[visualization]

# Performance monitoring only  
pip install quantum-task-planner[performance]

# Quantum algorithms only
pip install quantum-task-planner[quantum]
```

## Deployment Options

### 1. Standalone Application

Deploy as a standalone CLI application:

```bash
# Install
pip install quantum-task-planner[full]

# Run
quantum-planner schedule tasks.json resources.json -o results.json
```

### 2. Library Integration

Use as a Python library in your application:

```python
from quantum_planner import QuantumInspiredScheduler, Task, Resource

scheduler = QuantumInspiredScheduler()
# Add tasks and resources...
result = scheduler.create_optimal_schedule()
```

### 3. Microservice Deployment

Deploy as a REST API microservice:

```python
# app.py
from flask import Flask, request, jsonify
from quantum_planner import QuantumInspiredScheduler, Task, Resource

app = Flask(__name__)

@app.route('/schedule', methods=['POST'])
def create_schedule():
    data = request.json
    scheduler = QuantumInspiredScheduler()
    
    # Process tasks and resources from data
    for task_data in data['tasks']:
        task = Task.from_dict(task_data)
        scheduler.add_task(task)
    
    for resource_data in data['resources']:
        resource = Resource(**resource_data)
        scheduler.add_resource(resource)
    
    result = scheduler.create_optimal_schedule()
    return jsonify(result.schedule)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 4. Container Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_quantum.txt .
RUN pip install -r requirements_quantum.txt
RUN pip install quantum-task-planner[full]

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t quantum-planner .
docker run -p 8080:8080 quantum-planner
```

### 5. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-planner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-planner
  template:
    metadata:
      labels:
        app: quantum-planner
    spec:
      containers:
      - name: quantum-planner
        image: quantum-planner:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: MAX_WORKERS
          value: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-planner-service
spec:
  selector:
    app: quantum-planner
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

## Configuration

### Environment Variables

```bash
# Logging
export LOG_LEVEL=INFO
export LOG_DIR=/var/log/quantum-planner

# Performance
export MAX_WORKERS=8
export ENABLE_PERFORMANCE_MONITORING=true
export MEMORY_THRESHOLD_GB=4.0

# Scaling
export ENABLE_SCALING=true
export ADAPTIVE_SCALING=true

# Quantum Settings
export OPTIMIZATION_METHOD=hybrid
export ENABLE_SUPERPOSITION=true
export ENABLE_ENTANGLEMENT=true
```

### Configuration File

```yaml
# config.yaml
quantum_planner:
  logging:
    level: INFO
    dir: /var/log/quantum-planner
    enable_structured: true
  
  performance:
    monitoring_enabled: true
    max_workers: 8
    memory_threshold_gb: 4.0
  
  scaling:
    enabled: true
    adaptive: true
    max_parallel_tasks: 20
  
  quantum:
    optimization_method: hybrid
    enable_superposition: true
    enable_entanglement: true
    max_iterations: 1000
```

## Security Considerations

### 1. Input Validation
- All task and resource data is validated
- Dependency cycles are detected
- Resource constraints are enforced

### 2. Error Handling
- Comprehensive error handling with recovery
- Circuit breaker pattern for cascade failure prevention
- Secure error reporting (no sensitive data in logs)

### 3. Resource Management
- Memory usage monitoring and optimization
- CPU usage limits and adaptive scaling
- Resource leak detection and cleanup

### 4. Access Control
```python
# Add authentication middleware
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not validate_token(token):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/schedule', methods=['POST'])
@require_auth
def create_schedule():
    # ... scheduling logic
```

## Monitoring and Observability

### 1. Health Checks

```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/ready')
def readiness_check():
    # Check dependencies, resources, etc.
    return jsonify({'ready': True})
```

### 2. Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
schedule_requests = Counter('schedule_requests_total', 'Total schedule requests')
schedule_duration = Histogram('schedule_duration_seconds', 'Time spent creating schedules')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### 3. Logging

```python
import structlog

logger = structlog.get_logger()

@app.before_request
def log_request():
    logger.info("request_started", 
                method=request.method, 
                path=request.path,
                user_id=get_user_id())
```

## Performance Tuning

### 1. CPU Optimization
```bash
# Set CPU affinity
taskset -c 0-3 python app.py

# Use multiple workers
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### 2. Memory Optimization
```python
# Configure garbage collection
import gc
gc.set_threshold(700, 10, 10)

# Use memory profiling
from memory_profiler import profile

@profile
def create_schedule():
    # ... scheduling logic
```

### 3. Database Optimization
```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

## Scaling Patterns

### 1. Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use stateless design for easy scaling
- Implement distributed task queues

### 2. Vertical Scaling
- Increase CPU/memory for complex schedules
- Use adaptive worker pool sizing
- Monitor and adjust resource limits

### 3. Auto-scaling
```yaml
# HPA for Kubernetes
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-planner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-planner
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Disaster Recovery

### 1. Backup Strategy
- Regular backups of configuration
- Task and schedule data persistence
- Recovery procedures documentation

### 2. High Availability
- Multi-region deployment
- Database replication
- Graceful degradation

### 3. Monitoring and Alerts
```yaml
# Alertmanager rules
groups:
- name: quantum-planner
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, schedule_duration_seconds) > 30
    for: 10m
    annotations:
      summary: "High scheduling latency"
```

## Best Practices

1. **Start Simple**: Begin with basic installation and gradually add features
2. **Monitor Everything**: Set up comprehensive monitoring from day one
3. **Test Thoroughly**: Use provided test suites and add custom tests
4. **Secure by Default**: Implement authentication and input validation
5. **Scale Gradually**: Start with single instance, scale based on demand
6. **Plan for Failure**: Implement error handling and recovery procedures

## Support and Maintenance

- Regular updates and security patches
- Performance monitoring and optimization
- Capacity planning and scaling decisions
- Documentation updates and training

## Troubleshooting

### Common Issues

1. **Memory Issues**: 
   - Enable memory monitoring
   - Adjust worker pool sizes
   - Implement memory cleanup

2. **Performance Problems**:
   - Use performance profiling
   - Optimize quantum algorithms
   - Adjust scheduling parameters

3. **Scaling Issues**:
   - Check resource constraints
   - Monitor worker utilization
   - Review scaling configuration

For detailed troubleshooting, see the full documentation and log analysis tools.