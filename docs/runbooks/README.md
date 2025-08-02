# Operational Runbooks

This directory contains operational procedures and runbooks for the MicroDiff-MatDesign project.

## Common Operational Scenarios

### 1. Service Deployment

#### Standard Deployment Procedure

```bash
#!/bin/bash
# deploy.sh - Standard deployment script

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "Deploying MicroDiff-MatDesign v${VERSION} to ${ENVIRONMENT}"

# Pre-deployment checks
echo "Running pre-deployment checks..."
./scripts/pre-deployment-checks.sh ${ENVIRONMENT}

# Database migrations
echo "Running database migrations..."
python manage.py migrate --database=${ENVIRONMENT}

# Deploy application
echo "Deploying application..."
kubectl set image deployment/microdiff-deployment \
  microdiff=microdiff-matdesign:${VERSION} \
  -n ${ENVIRONMENT}

# Wait for rollout
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/microdiff-deployment -n ${ENVIRONMENT}

# Post-deployment verification
echo "Running post-deployment verification..."
./scripts/post-deployment-checks.sh ${ENVIRONMENT}

echo "Deployment completed successfully!"
```

#### Blue-Green Deployment

```bash
#!/bin/bash
# blue-green-deploy.sh

CURRENT_COLOR=$(kubectl get service microdiff-service -o jsonpath='{.spec.selector.version}')
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "Current deployment: $CURRENT_COLOR"
echo "Deploying to: $NEW_COLOR"

# Deploy new version
kubectl apply -f k8s/deployment-${NEW_COLOR}.yaml

# Wait for readiness
kubectl wait --for=condition=available deployment/microdiff-${NEW_COLOR} --timeout=300s

# Run health checks
if ./scripts/health-check.sh ${NEW_COLOR}; then
    # Switch traffic
    kubectl patch service microdiff-service -p '{"spec":{"selector":{"version":"'$NEW_COLOR'"}}}'
    echo "Traffic switched to $NEW_COLOR"
    
    # Scale down old version
    kubectl scale deployment microdiff-${CURRENT_COLOR} --replicas=0
    echo "Scaled down $CURRENT_COLOR deployment"
else
    echo "Health checks failed, rolling back..."
    kubectl delete deployment microdiff-${NEW_COLOR}
    exit 1
fi
```

### 2. Incident Response

#### High CPU Usage

**Symptoms:**
- Response times > 5 seconds
- CPU usage > 80% sustained
- Queue backlog growing

**Investigation Steps:**

```bash
# Check current resource usage
kubectl top pods -n production

# Check application logs
kubectl logs -f deployment/microdiff-deployment -n production

# Check for memory leaks
kubectl exec -it <pod-name> -- ps aux | head -20

# Check GPU utilization
kubectl exec -it <pod-name> -- nvidia-smi
```

**Resolution:**

```bash
# Scale up replicas temporarily
kubectl scale deployment microdiff-deployment --replicas=6 -n production

# Check if scaling helps
watch kubectl get pods -n production

# If scaling doesn't help, restart deployment
kubectl rollout restart deployment/microdiff-deployment -n production
```

#### Model Loading Failures

**Symptoms:**
- 500 errors on model inference endpoints
- "Model not found" errors in logs
- Health check failures

**Investigation:**

```bash
# Check model files exist
kubectl exec -it <pod-name> -- ls -la /app/models/

# Check model loading logs
kubectl logs <pod-name> | grep -i "model"

# Verify model file integrity
kubectl exec -it <pod-name> -- md5sum /app/models/ti64_lpbf.pth
```

**Resolution:**

```bash
# Re-download models if corrupted
kubectl exec -it <pod-name> -- /app/scripts/download-models.sh

# Restart pods to reload models
kubectl delete pods -l app=microdiff -n production

# Verify model loading
kubectl exec -it <pod-name> -- python -c "
from microdiff_matdesign import MicrostructureDiffusion
model = MicrostructureDiffusion(alloy='Ti-6Al-4V', pretrained=True)
print('Model loaded successfully')
"
```

#### Database Connection Issues

**Symptoms:**
- Database connection timeouts
- "Connection refused" errors
- Transaction rollbacks

**Investigation:**

```bash
# Check database connectivity
kubectl exec -it <pod-name> -- nc -zv postgres-service 5432

# Check connection pool status
kubectl exec -it <pod-name> -- python -c "
import psycopg2
from django.db import connection
print('Active connections:', len(connection.queries))
"

# Check database logs
kubectl logs postgres-pod -n database
```

**Resolution:**

```bash
# Restart database connections
kubectl exec -it <pod-name> -- python manage.py shell -c "
from django.db import connections
for conn in connections.all():
    conn.close()
"

# Scale up database if needed
kubectl scale statefulset postgres --replicas=2 -n database

# Check for long-running queries
kubectl exec -it postgres-pod -- psql -U postgres -c "
SELECT pid, query, state, query_start 
FROM pg_stat_activity 
WHERE state = 'active' AND query_start < now() - interval '5 minutes';
"
```

### 3. Performance Optimization

#### GPU Memory Optimization

```bash
#!/bin/bash
# gpu-optimization.sh

# Check current GPU usage
nvidia-smi

# Clear GPU cache
python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"

# Optimize model batch size
python scripts/optimize-batch-size.py --target-memory=0.8

# Monitor GPU memory during inference
python scripts/memory-monitor.py --duration=300
```

#### Model Inference Optimization

```python
# optimize_inference.py
import torch
import time
from microdiff_matdesign import MicrostructureDiffusion

def benchmark_model(model, num_runs=100):
    """Benchmark model inference performance"""
    
    # Warm up
    dummy_input = torch.randn(1, 1, 128, 128, 128)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def optimize_model():
    """Apply optimization techniques"""
    
    model = MicrostructureDiffusion(alloy="Ti-6Al-4V", pretrained=True)
    
    # Convert to half precision
    model = model.half()
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model

if __name__ == "__main__":
    model = optimize_model()
    avg_time = benchmark_model(model)
    print(f"Average inference time: {avg_time:.4f} seconds")
```

### 4. Backup and Recovery

#### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p ${BACKUP_DIR}

# Create database backup
kubectl exec postgres-pod -n database -- pg_dump -U postgres microdiff > \
  ${BACKUP_DIR}/microdiff_db_$(date +%Y%m%d_%H%M%S).sql

# Backup model files
kubectl cp microdiff-pod:/app/models ${BACKUP_DIR}/models

# Compress backup
tar -czf ${BACKUP_DIR}.tar.gz ${BACKUP_DIR}

# Upload to cloud storage
aws s3 cp ${BACKUP_DIR}.tar.gz s3://microdiff-backups/

echo "Backup completed: ${BACKUP_DIR}.tar.gz"
```

#### Disaster Recovery

```bash
#!/bin/bash
# disaster-recovery.sh

BACKUP_FILE=${1}
RECOVERY_ENV=${2:-staging}

echo "Starting disaster recovery from ${BACKUP_FILE}"

# Restore database
kubectl exec -i postgres-pod -n ${RECOVERY_ENV} -- psql -U postgres < \
  $(tar -xzOf ${BACKUP_FILE} --wildcards "*/microdiff_db_*.sql")

# Restore model files
tar -xzf ${BACKUP_FILE} --strip-components=1 -C /tmp/recovery
kubectl cp /tmp/recovery/models microdiff-pod:/app/models

# Restart application
kubectl rollout restart deployment/microdiff-deployment -n ${RECOVERY_ENV}

# Verify recovery
./scripts/health-check.sh ${RECOVERY_ENV}

echo "Disaster recovery completed"
```

### 5. Maintenance Procedures

#### Scheduled Maintenance

```bash
#!/bin/bash
# scheduled-maintenance.sh

echo "Starting scheduled maintenance..."

# Put application in maintenance mode
kubectl patch configmap microdiff-config -p '{"data":{"MAINTENANCE_MODE":"true"}}'

# Scale down to single replica
kubectl scale deployment microdiff-deployment --replicas=1

# Perform maintenance tasks
echo "Running database maintenance..."
kubectl exec postgres-pod -- vacuumdb -U postgres -z microdiff

echo "Cleaning up old logs..."
kubectl exec microdiff-pod -- find /var/log -name "*.log" -mtime +7 -delete

echo "Updating model files..."
kubectl exec microdiff-pod -- /app/scripts/update-models.sh

# Scale back up
kubectl scale deployment microdiff-deployment --replicas=3

# Exit maintenance mode
kubectl patch configmap microdiff-config -p '{"data":{"MAINTENANCE_MODE":"false"}}'

echo "Scheduled maintenance completed"
```

#### Certificate Renewal

```bash
#!/bin/bash
# renew-certificates.sh

# Check certificate expiration
openssl x509 -in /etc/ssl/certs/microdiff.crt -noout -dates

# Renew Let's Encrypt certificate
certbot renew --cert-name microdiff.example.com

# Update Kubernetes secret
kubectl create secret tls microdiff-tls \
  --cert=/etc/letsencrypt/live/microdiff.example.com/fullchain.pem \
  --key=/etc/letsencrypt/live/microdiff.example.com/privkey.pem \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart ingress to pick up new certificate
kubectl rollout restart deployment/nginx-ingress-controller
```

## Emergency Contacts

### On-Call Rotation
- **Primary**: devops-oncall@company.com
- **Secondary**: engineering-lead@company.com
- **Escalation**: cto@company.com

### Service Dependencies
- **Database**: dba-team@company.com
- **Infrastructure**: platform-team@company.com
- **Security**: security-team@company.com

### Vendor Contacts
- **Cloud Provider**: support@cloudprovider.com
- **Monitoring**: support@monitoringvendor.com
- **CDN**: support@cdnprovider.com

## Escalation Procedures

### Severity Levels

**P0 - Critical**
- Service completely down
- Data loss or corruption
- Security breach
- **Response Time**: 15 minutes
- **Escalation**: Immediate

**P1 - High**
- Major functionality impaired
- Performance severely degraded
- **Response Time**: 1 hour
- **Escalation**: 2 hours

**P2 - Medium**
- Minor functionality issues
- Performance degradation
- **Response Time**: 4 hours
- **Escalation**: 8 hours

**P3 - Low**
- Cosmetic issues
- Enhancement requests
- **Response Time**: 24 hours
- **Escalation**: 48 hours

### Communication Channels

1. **Slack**: #microdiff-alerts for real-time coordination
2. **Email**: For formal incident reports
3. **Phone**: For P0/P1 incidents only
4. **Status Page**: For customer communication

### Post-Incident Review

After any P0 or P1 incident:

1. **Immediate**: Update status page
2. **Within 24h**: Draft incident report
3. **Within 72h**: Post-mortem meeting
4. **Within 1 week**: Action items implemented

## Monitoring and Alerting Contacts

### Alert Routing

```yaml
# alertmanager.yml excerpt
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty'
  - match:
      severity: warning
    receiver: 'slack'

receivers:
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: '<pagerduty-key>'
    
- name: 'slack'
  slack_configs:
  - api_url: '<slack-webhook>'
    channel: '#microdiff-alerts'
```