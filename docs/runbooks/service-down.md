# Service Down Runbook

**Alert**: ServiceDown
**Severity**: Critical
**Team**: Platform

## Symptoms
- Service health check returning 0 (down)
- HTTP requests failing with connection errors
- Users unable to access the application
- Monitoring dashboard shows service as unavailable

## Immediate Actions (0-5 minutes)

### 1. Verify the Alert
```bash
# Check if service is actually down
curl -f http://localhost:8080/health
echo $?  # Should return 0 if healthy, non-zero if down

# Check service status
systemctl status microdiff-matdesign
```

### 2. Check Recent Changes
```bash
# Check recent deployments
git log --oneline -10

# Check system logs for recent service restarts
journalctl -u microdiff-matdesign --since "1 hour ago"
```

### 3. Quick Service Restart (if appropriate)
```bash
# Only if no obvious cause is found and service appears hung
systemctl restart microdiff-matdesign

# Wait 30 seconds and check status
sleep 30
systemctl status microdiff-matdesign
curl -f http://localhost:8080/health
```

## Investigation (5-15 minutes)

### 1. Check Application Logs
```bash
# Check recent application logs
journalctl -u microdiff-matdesign -f --lines=100

# Look for specific error patterns
journalctl -u microdiff-matdesign --since "1 hour ago" | grep -i "error\|fatal\|exception"
```

### 2. Check System Resources
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check CPU usage
top -o %CPU

# Check disk space
df -h
du -sh /var/log/* | sort -hr | head -5

# Check file descriptors
lsof | wc -l
ulimit -n
```

### 3. Check Dependencies
```bash
# Database connectivity
pg_isready -h localhost -p 5432
psql -h localhost -p 5432 -U microdiff -c "SELECT 1;"

# Redis (if applicable)
redis-cli ping

# External API dependencies
curl -f https://api.external-service.com/health
```

### 4. Check Network Issues
```bash
# Check listening ports
netstat -tlnp | grep :8080

# Check firewall rules
iptables -L

# Check DNS resolution
nslookup api.external-service.com
```

## Common Causes and Solutions

### 1. Out of Memory
**Symptoms**: OOMKiller messages in logs, high memory usage

**Solution**:
```bash
# Check for OOM killer activity
dmesg | grep -i "killed process"

# If OOM killed the service, restart it
systemctl restart microdiff-matdesign

# Consider scaling up memory or optimizing memory usage
```

### 2. Database Connection Issues
**Symptoms**: Database connection errors in logs

**Solution**:
```bash
# Check database status
systemctl status postgresql

# Check database connections
psql -h localhost -p 5432 -U microdiff -c "SELECT count(*) FROM pg_stat_activity;"

# If database is down, restart it
systemctl restart postgresql
sleep 10
systemctl restart microdiff-matdesign
```

### 3. Disk Space Full
**Symptoms**: "No space left on device" errors

**Solution**:
```bash
# Find large files
find /var/log -name "*.log" -size +100M

# Clean up old logs
journalctl --vacuum-time=7d

# Rotate application logs
logrotate /etc/logrotate.d/microdiff-matdesign

# Restart service
systemctl restart microdiff-matdesign
```

### 4. Configuration Issues
**Symptoms**: Service fails to start, configuration errors in logs

**Solution**:
```bash
# Validate configuration
python -m microdiff_matdesign.config --validate

# Check configuration file syntax
python -c "import yaml; yaml.safe_load(open('/etc/microdiff/config.yaml'))"

# Restore from backup if needed
cp /etc/microdiff/config.yaml.backup /etc/microdiff/config.yaml
systemctl restart microdiff-matdesign
```

### 5. Port Already in Use
**Symptoms**: "Address already in use" errors

**Solution**:
```bash
# Find process using the port
lsof -i :8080

# Kill the process if it's a stale instance
kill -9 <PID>

# Restart the service
systemctl restart microdiff-matdesign
```

## Escalation (15+ minutes)

### If Service Still Down After 15 Minutes:

1. **Page the Team Lead**
   - Send alert to team lead
   - Provide summary of actions taken
   - Share relevant log snippets

2. **Consider Rollback**
   ```bash
   # If recent deployment is suspected
   git checkout <previous-working-commit>
   
   # Rebuild and deploy
   make build
   systemctl restart microdiff-matdesign
   ```

3. **Enable Maintenance Mode**
   ```bash
   # Enable maintenance page
   cp /var/www/maintenance.html /var/www/html/index.html
   
   # Update load balancer to show maintenance page
   ```

## Advanced Diagnostics

### 1. Core Dump Analysis
```bash
# Check for core dumps
ls -la /var/crash/
ls -la /tmp/core.*

# Analyze core dump (if available)
gdb python /tmp/core.<pid>
```

### 2. Strace Analysis
```bash
# If process is running but unresponsive
strace -p <pid> -o /tmp/strace.out

# Check the output for stuck system calls
tail -f /tmp/strace.out
```

### 3. Memory Analysis
```bash
# Check memory maps
cat /proc/<pid>/maps

# Check memory limits
cat /proc/<pid>/limits
```

## Recovery Verification

### 1. Health Check
```bash
# Verify service is responding
curl -f http://localhost:8080/health

# Check detailed health endpoint
curl http://localhost:8080/health/detailed | jq
```

### 2. Functional Test
```bash
# Test core functionality
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

### 3. Monitor for 10 Minutes
- Watch error rates in Grafana
- Monitor response times
- Check for any recurring issues

## Post-Incident Actions

### 1. Document the Incident
- Root cause
- Actions taken
- Time to resolution
- Impact assessment

### 2. Update Monitoring
- Add new alerts if gaps were identified
- Improve health checks if needed
- Update dashboards

### 3. Preventive Measures
- Code fixes if bugs were found
- Infrastructure improvements
- Process improvements
- Runbook updates

## Prevention Strategies

1. **Automated Health Checks**: Ensure comprehensive health endpoints
2. **Resource Monitoring**: Set up alerts for resource exhaustion
3. **Dependency Monitoring**: Monitor all external dependencies
4. **Graceful Degradation**: Implement circuit breakers and fallbacks
5. **Blue-Green Deployments**: Reduce deployment-related outages
6. **Regular Testing**: Conduct disaster recovery drills

## Related Runbooks
- [High Error Rate](high-error-rate.md)
- [Performance Degradation](performance-degradation.md)
- [Database Issues](database-issues.md)
- [Resource Exhaustion](resource-exhaustion.md)