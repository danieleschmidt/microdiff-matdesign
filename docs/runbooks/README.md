# Operational Runbooks

This directory contains runbooks for common operational scenarios, incident response procedures, and troubleshooting guides for MicroDiff-MatDesign.

## Structure

### Incident Response
- `incident-response.md` - General incident response procedures
- `security-incidents.md` - Security incident handling
- `data-loss-recovery.md` - Data recovery procedures

### System Issues
- `service-down.md` - Service outage resolution
- `high-error-rate.md` - Debugging high error rates
- `performance-degradation.md` - Performance issue resolution
- `resource-exhaustion.md` - CPU, memory, disk space issues

### Application-Specific
- `model-inference-issues.md` - ML model troubleshooting
- `image-processing-failures.md` - Image processing pipeline issues
- `database-issues.md` - Database connectivity and performance
- `queue-management.md` - Processing queue issues

### Maintenance
- `deployment-procedures.md` - Safe deployment practices
- `backup-restore.md` - Backup and restore procedures
- `scaling-procedures.md` - Horizontal and vertical scaling
- `maintenance-windows.md` - Scheduled maintenance procedures

## Quick Reference

### Emergency Contacts
- **Platform Team**: platform-oncall@company.com
- **Security Team**: security@company.com
- **ML Team**: ml-team@company.com
- **Management**: incidents@company.com

### Key Dashboards
- [Service Overview](https://grafana.company.com/d/service-overview)
- [System Resources](https://grafana.company.com/d/system-resources)
- [Model Performance](https://grafana.company.com/d/model-performance)
- [Business Metrics](https://grafana.company.com/d/business-metrics)

### Common Commands

```bash
# Check service status
systemctl status microdiff-matdesign

# View recent logs
journalctl -u microdiff-matdesign -f --lines=50

# Check resource usage
htop
df -h
nvidia-smi

# Database status
pg_isready -h localhost -p 5432

# Application health check
curl http://localhost:8080/health

# Restart services (use with caution)
systemctl restart microdiff-matdesign
```

### Escalation Matrix

| Severity | Initial Response | Escalation (15 min) | Manager Escalation (30 min) |
|----------|------------------|---------------------|------------------------------|
| P0 (Critical) | On-call engineer | Team lead | VP Engineering |
| P1 (High) | On-call engineer | Team lead | Engineering Manager |
| P2 (Medium) | Assigned engineer | Team lead | - |
| P3 (Low) | Assigned engineer | - | - |

## Using These Runbooks

1. **Identify the Issue**: Use monitoring alerts and symptoms to determine the appropriate runbook
2. **Follow the Steps**: Execute the troubleshooting steps in order
3. **Document Actions**: Record all actions taken during incident resolution
4. **Post-Incident**: Complete post-mortem if required
5. **Update Runbooks**: Improve runbooks based on lessons learned

## Contributing

When updating runbooks:
- Keep procedures clear and step-by-step
- Include expected outputs for commands
- Add troubleshooting for common failure scenarios
- Update after any significant changes to the system
- Test procedures during maintenance windows