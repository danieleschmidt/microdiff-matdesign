# 🤖 Automation Scripts

This directory contains automation scripts for maintaining and monitoring the MicroDiff-MatDesign project.

## Available Scripts

### 📊 Metrics Collection

#### `collect-metrics.py`
Comprehensive metrics collection script that gathers data from various sources.

**Usage:**
```bash
# Basic metrics collection
python scripts/collect-metrics.py

# Update metrics configuration with collected values
python scripts/collect-metrics.py --update-config

# Generate markdown report
python scripts/collect-metrics.py --output markdown --report-file metrics-report.md
```

**Features:**
- Git repository metrics (commits, contributors, LOC)
- Test coverage and execution time
- Security vulnerabilities
- Dependency freshness
- Code quality indicators
- Build performance

### 📦 Dependency Management

#### `dependency-updates.sh`
Automated dependency update script with comprehensive testing.

**Usage:**
```bash
# Minor version updates (default)
./scripts/automation/dependency-updates.sh

# Security updates only
./scripts/automation/dependency-updates.sh security

# Major version updates
./scripts/automation/dependency-updates.sh major

# Patch updates only
./scripts/automation/dependency-updates.sh patch
```

**Features:**
- Intelligent update categorization
- Comprehensive testing before PR creation
- Security vulnerability prioritization
- Automatic rollback on test failures
- Detailed update summaries

### 🔧 Repository Maintenance

#### `repository-maintenance.py`
Comprehensive repository maintenance and health monitoring.

**Usage:**
```bash
# Full maintenance (default operations)
python scripts/automation/repository-maintenance.py

# Specific operations
python scripts/automation/repository-maintenance.py \
  --operations clean_branches check_stale health_check

# Dry run to see what would be done
python scripts/automation/repository-maintenance.py --dry-run

# Generate maintenance report
python scripts/automation/repository-maintenance.py \
  --report-file maintenance-report.md
```

**Operations:**
- `clean_branches`: Remove old merged branches
- `check_stale`: Identify stale issues and PRs
- `label_stale`: Add stale labels to old items
- `update_metadata`: Update repository description and topics
- `health_check`: Comprehensive repository health assessment

### 📈 Code Quality Monitoring

#### `code-quality-monitor.py`
Continuous code quality monitoring with trend analysis.

**Usage:**
```bash
# Run quality check
python scripts/automation/code-quality-monitor.py

# Generate trend report for last 60 days
python scripts/automation/code-quality-monitor.py --trend-days 60

# Output markdown report
python scripts/automation/code-quality-monitor.py \
  --output-format markdown \
  --report-file quality-report.md
```

**Metrics Tracked:**
- Test coverage percentage
- Code complexity (cyclomatic)
- Maintainability index
- Code duplication
- Documentation coverage
- Security issues

## Configuration

### Environment Variables

```bash
# GitHub API access (for repository-maintenance.py)
export GITHUB_TOKEN="your_github_token"

# Metrics collection configuration
export METRICS_CONFIG_PATH=".github/project-metrics.json"

# Quality monitoring database
export QUALITY_DB_PATH=".quality_metrics.db"
```

### Metrics Configuration

The metrics collection is configured via `.github/project-metrics.json`:

```json
{
  "metrics": {
    "development": {
      "code_quality": {
        "test_coverage": {
          "target": 85,
          "current": 0,
          "trend": "stable"
        }
      }
    }
  }
}
```

## Automation Integration

### GitHub Actions Integration

These scripts can be integrated into GitHub Actions workflows:

```yaml
name: Automated Maintenance

on:
  schedule:
    - cron: '0 3 * * 1'  # Weekly Monday 3 AM

jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run metrics collection
      run: |
        python scripts/collect-metrics.py --update-config
    
    - name: Run repository maintenance
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        python scripts/automation/repository-maintenance.py
    
    - name: Run quality monitoring
      run: |
        python scripts/automation/code-quality-monitor.py
```

### Cron Job Setup

For local or server-based automation:

```bash
# Weekly dependency updates
0 3 * * 1 cd /path/to/project && ./scripts/automation/dependency-updates.sh minor

# Daily metrics collection
0 6 * * * cd /path/to/project && python scripts/collect-metrics.py --update-config

# Weekly repository maintenance
0 4 * * 1 cd /path/to/project && python scripts/automation/repository-maintenance.py

# Daily quality monitoring
0 7 * * * cd /path/to/project && python scripts/automation/code-quality-monitor.py
```

## Monitoring and Alerting

### Metric Thresholds

Configure alerts based on metric thresholds:

```bash
# Example: Alert if test coverage drops below 80%
if [ $(python scripts/collect-metrics.py | jq '.collected_metrics.test_coverage') -lt 80 ]; then
  echo "❌ Test coverage below threshold!"
  # Send alert
fi
```

### Integration with Monitoring Systems

Export metrics to external monitoring systems:

```bash
# Export to Prometheus
python scripts/collect-metrics.py --output json | \
  prometheus-push-gateway --job microdiff-metrics

# Export to DataDog
python scripts/collect-metrics.py --output json | \
  datadog-metrics-sender
```

## Best Practices

### Script Execution

1. **Always test in dry-run mode first**
   ```bash
   ./scripts/automation/dependency-updates.sh --dry-run
   ```

2. **Run scripts in project root directory**
   ```bash
   cd /path/to/microdiff-matdesign
   ./scripts/automation/script-name.sh
   ```

3. **Check script output and logs**
   ```bash
   ./script.sh 2>&1 | tee script-output.log
   ```

### Error Handling

Scripts include comprehensive error handling:
- Automatic rollback on failures
- Detailed error messages
- Exit codes for automation integration

### Security Considerations

- Scripts never commit sensitive information
- GitHub tokens are handled securely
- Local files are cleaned up automatically
- Dry-run mode available for testing

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x scripts/automation/*.sh scripts/automation/*.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Git Authentication**
   ```bash
   git config --global credential.helper store
   ```

4. **GitHub API Rate Limits**
   - Use personal access token with appropriate permissions
   - Monitor rate limit headers in script output

### Debug Mode

Enable debug output for troubleshooting:

```bash
# Enable verbose output
export DEBUG=1
./scripts/automation/script-name.sh

# Trace script execution
bash -x ./scripts/automation/script-name.sh
```

## Contributing

When adding new automation scripts:

1. Follow the existing script structure
2. Include comprehensive help text
3. Add error handling and cleanup
4. Test in dry-run mode
5. Update this README
6. Add example usage to documentation

### Script Template

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    args = parser.parse_args()
    
    try:
        # Script logic here
        pass
    except KeyboardInterrupt:
        print("\n❌ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

## Support

For issues with automation scripts:

1. Check script logs and error messages
2. Verify prerequisites and permissions
3. Test in dry-run mode
4. Review GitHub repository settings
5. Check GitHub Actions workflow logs

For feature requests or improvements, please open an issue in the repository.