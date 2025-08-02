# Monitoring & Observability

This directory contains monitoring and observability configuration for the MicroDiff-MatDesign project.

## Health Check Endpoints

### Application Health Check

```python
from flask import Flask, jsonify
from microdiff_matdesign.core import MicrostructureDiffusion

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': microdiff_matdesign.__version__
    })

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with component status"""
    checks = {
        'database': check_database_connection(),
        'gpu': check_gpu_availability(),
        'model': check_model_loading(),
        'storage': check_storage_access()
    }
    
    overall_status = 'healthy' if all(checks.values()) else 'degraded'
    
    return jsonify({
        'status': overall_status,
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    })

def check_gpu_availability():
    """Check if GPU is available for model inference"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def check_model_loading():
    """Check if pre-trained models can be loaded"""
    try:
        model = MicrostructureDiffusion(alloy="Ti-6Al-4V", pretrained=True)
        return True
    except Exception:
        return False
```

### Kubernetes Health Probes

```yaml
# k8s-health-probes.yaml
apiVersion: v1
kind: Service
metadata:
  name: microdiff-service
spec:
  selector:
    app: microdiff
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: microdiff-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: microdiff
  template:
    metadata:
      labels:
        app: microdiff
    spec:
      containers:
      - name: microdiff
        image: microdiff-matdesign:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Structured Logging Configuration

### Python Logging Setup

```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'model_type'):
            log_entry['model_type'] = record.model_type
            
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration"""
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler
    file_handler = logging.FileHandler('microdiff.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    loggers = [
        'microdiff_matdesign',
        'microdiff_matdesign.models',
        'microdiff_matdesign.imaging',
        'microdiff_matdesign.optimization'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

# Usage example
def log_model_inference(model_type: str, duration_ms: float, success: bool):
    """Log model inference metrics"""
    logger = logging.getLogger('microdiff_matdesign.models')
    
    extra = {
        'model_type': model_type,
        'duration_ms': duration_ms,
        'success': success
    }
    
    if success:
        logger.info(f"Model inference completed successfully", extra=extra)
    else:
        logger.error(f"Model inference failed", extra=extra)
```

### Log Aggregation Configuration

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*microdiff*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_key timestamp
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.monitoring.svc.cluster.local
      port 9200
      index_name microdiff-logs
      type_name _doc
    </match>
```

## Prometheus Metrics

### Application Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter(
    'microdiff_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'microdiff_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

MODEL_INFERENCE_DURATION = Histogram(
    'microdiff_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type', 'alloy']
)

ACTIVE_CONNECTIONS = Gauge(
    'microdiff_active_connections',
    'Number of active connections'
)

GPU_MEMORY_USAGE = Gauge(
    'microdiff_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device']
)

MODEL_ACCURACY = Gauge(
    'microdiff_model_accuracy',
    'Model prediction accuracy',
    ['model_type', 'validation_set']
)

def track_request_metrics(f):
    """Decorator to track request metrics"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status = 'success'
        
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            status = 'error'
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=getattr(f, 'method', 'unknown'),
                endpoint=getattr(f, 'endpoint', 'unknown')
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=getattr(f, 'method', 'unknown'),
                endpoint=getattr(f, 'endpoint', 'unknown'),
                status=status
            ).inc()
    
    return wrapper

def track_model_inference(model_type: str, alloy: str):
    """Decorator to track model inference metrics"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                MODEL_INFERENCE_DURATION.labels(
                    model_type=model_type,
                    alloy=alloy
                ).observe(duration)
        
        return wrapper
    return decorator

def update_gpu_metrics():
    """Update GPU usage metrics"""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(device=f'cuda:{i}').set(memory_used)
    except ImportError:
        pass

# Start metrics server
def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "microdiff_rules.yml"

scrape_configs:
  - job_name: 'microdiff-app'
    static_configs:
      - targets: ['microdiff-service:8000']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'microdiff-gpu'
    static_configs:
      - targets: ['microdiff-service:8001']
    scrape_interval: 10s
    metrics_path: /gpu-metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alerting Rules

```yaml
# microdiff_rules.yml
groups:
- name: microdiff.rules
  rules:
  - alert: HighErrorRate
    expr: rate(microdiff_requests_total{status="error"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: ModelInferenceLatency
    expr: histogram_quantile(0.95, rate(microdiff_model_inference_duration_seconds_bucket[5m])) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Model inference latency too high"
      description: "95th percentile latency is {{ $value }} seconds"

  - alert: GPUMemoryHigh
    expr: microdiff_gpu_memory_usage_bytes / (1024^3) > 8
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory usage high"
      description: "GPU memory usage is {{ $value }}GB"

  - alert: ServiceDown
    expr: up{job="microdiff-app"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MicroDiff service is down"
      description: "MicroDiff service has been down for more than 1 minute"
```

## Grafana Dashboards

### Main Application Dashboard

```json
{
  "dashboard": {
    "title": "MicroDiff-MatDesign Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(microdiff_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(microdiff_requests_total{status=\"error\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(microdiff_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(microdiff_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "microdiff_gpu_memory_usage_bytes / (1024^3)",
            "legendFormat": "GPU {{device}}"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "microdiff_model_accuracy",
            "legendFormat": "{{model_type}}"
          }
        ]
      }
    ]
  }
}
```

## Tracing Configuration

### OpenTelemetry Setup

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.torch import TorchInstrumentor

def setup_tracing(service_name: str = "microdiff-matdesign"):
    """Setup distributed tracing with Jaeger"""
    
    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument libraries
    RequestsInstrumentor().instrument()
    TorchInstrumentor().instrument()
    
    return tracer

# Usage in application code
tracer = setup_tracing()

def process_microstructure(image_path: str):
    """Process microstructure with tracing"""
    with tracer.start_as_current_span("process_microstructure") as span:
        span.set_attribute("image_path", image_path)
        span.set_attribute("operation", "microstructure_processing")
        
        # Your processing logic here
        result = perform_processing(image_path)
        
        span.set_attribute("result_size", len(result))
        return result
```

## Monitoring Best Practices

### 1. Golden Signals
- **Latency**: Track request and model inference response times
- **Traffic**: Monitor request rates and throughput
- **Errors**: Track error rates and types
- **Saturation**: Monitor GPU utilization and memory usage

### 2. Alerting Strategy
- **Critical**: Service unavailable, GPU memory exhaustion
- **Warning**: High latency, elevated error rates
- **Info**: Performance degradation, resource usage trends

### 3. Dashboard Organization
- **Overview**: High-level system health
- **Application**: Detailed application metrics
- **Infrastructure**: System resources and performance
- **Business**: Model accuracy and usage statistics

### 4. Log Aggregation
- **Centralized**: All logs sent to Elasticsearch
- **Structured**: JSON format for easy parsing
- **Retention**: 30 days for debugging, 1 year for compliance
- **Security**: Sensitive data filtering and access controls