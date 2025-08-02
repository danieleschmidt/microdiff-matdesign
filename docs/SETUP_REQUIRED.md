# Manual Setup Requirements

This document outlines manual setup steps required to complete the SDLC implementation due to GitHub App permission limitations.

## Required GitHub Actions Workflows

The following workflows need to be manually created in `.github/workflows/` directory:

### 1. CI/CD Pipeline (`ci.yml`)

Create `.github/workflows/ci.yml` with the content from [`docs/workflows/README.md`](workflows/README.md#1-ci/cd-pipeline-githubworkflowsciyml).

**Required Secrets:**
- `CODECOV_TOKEN` (for coverage reporting)

### 2. Security Scanning (`security.yml`)

Create `.github/workflows/security.yml` with the content from [`docs/workflows/README.md`](workflows/README.md#2-security-scanning-githubworkflowssecurityyml).

**Setup Required:**
- Enable CodeQL analysis in repository settings
- Configure Dependabot in repository settings

### 3. Dependency Updates (`dependencies.yml`)

Create `.github/workflows/dependencies.yml` with the content from [`docs/workflows/README.md`](workflows/README.md#3-dependency-updates-githubworkflowsdependenciesyml).

**Setup Required:**
- Enable GitHub Actions to create pull requests

## Repository Settings Configuration

### Branch Protection Rules

Configure the following branch protection rules for the `main` branch:

1. **Go to Repository Settings → Branches → Add rule**
2. **Branch name pattern:** `main`
3. **Protection settings:**
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
     - Select: `test (3.8)`, `test (3.9)`, `test (3.10)`, `test (3.11)`
     - Select: `CodeQL`
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Restrict pushes that create files larger than 100MB
   - ✅ Do not allow bypassing the above settings

### Security Settings

1. **Enable Dependabot:**
   - Go to Settings → Security & analysis
   - Enable "Dependabot alerts"
   - Enable "Dependabot security updates"

2. **Enable CodeQL Analysis:**
   - Go to Settings → Security & analysis
   - Enable "Code scanning alerts"
   - Set up CodeQL analysis

3. **Configure Secret Scanning:**
   - Go to Settings → Security & analysis
   - Enable "Secret scanning alerts"

### Repository Topics

Add the following topics to improve discoverability:

```
materials-science, machine-learning, diffusion-models, additive-manufacturing, 
inverse-design, microstructure, python, pytorch, research, open-source
```

### Repository Description

Update the repository description:

```
Diffusion model framework for inverse material design that transforms micro-CT images into printable alloy process parameters
```

### Repository Homepage

Set homepage URL to documentation site (when available):

```
https://microdiff-matdesign.readthedocs.io/
```

## Secrets Configuration

Configure the following secrets in repository settings:

### Required Secrets

1. **`CODECOV_TOKEN`**
   - Purpose: Coverage reporting
   - Obtain from: [Codecov.io](https://codecov.io/)
   - Path: Settings → Secrets and variables → Actions → New repository secret

2. **`PYPI_API_TOKEN`** (for package publishing)
   - Purpose: Automated package publishing
   - Obtain from: [PyPI Account Settings](https://pypi.org/manage/account/)
   - Scope: This project only

### Optional Secrets

1. **`SLACK_WEBHOOK_URL`**
   - Purpose: Slack notifications for alerts
   - Setup: Create Slack app with incoming webhook

2. **`DATADOG_API_KEY`**
   - Purpose: Metrics and monitoring integration
   - Obtain from: Datadog account settings

## Monitoring Setup

### Prometheus Integration

1. **Install Prometheus** (if not using managed service):
   ```bash
   # Using Docker
   docker run -d -p 9090:9090 \
     -v $(pwd)/docs/monitoring/prometheus-config.yml:/etc/prometheus/prometheus.yml \
     prom/prometheus
   ```

2. **Configure Application Metrics Endpoint:**
   - Implement `/metrics` endpoint in application
   - Use configuration from `docs/monitoring/prometheus-config.yml`

### Grafana Dashboards

1. **Install Grafana** (if not using managed service):
   ```bash
   docker run -d -p 3000:3000 grafana/grafana
   ```

2. **Import Dashboards:**
   - Create dashboards based on metrics defined in `docs/monitoring/prometheus-config.yml`
   - Configure alerts based on thresholds in `docs/monitoring/alerts.yaml`

## External Integrations

### Documentation Hosting

1. **ReadTheDocs Setup:**
   - Connect repository to [ReadTheDocs](https://readthedocs.org/)
   - Configure build settings for MkDocs/Sphinx
   - Set up automatic builds on documentation changes

2. **GitHub Pages** (alternative):
   - Enable GitHub Pages in repository settings
   - Configure to build from `docs/` directory

### Package Registry

1. **PyPI Publishing:**
   - Create account on [PyPI](https://pypi.org/)
   - Configure API token
   - Test package publishing workflow

### Monitoring Services

1. **Codecov Integration:**
   - Sign up at [Codecov.io](https://codecov.io/)
   - Connect repository
   - Configure coverage thresholds

2. **Dependabot Configuration:**
   - Create `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       open-pull-requests-limit: 5
   ```

## Verification Checklist

After completing manual setup, verify:

- [ ] All GitHub Actions workflows are created and passing
- [ ] Branch protection rules are configured
- [ ] Security scanning is enabled and running
- [ ] Dependabot is configured and creating PRs
- [ ] Secrets are properly configured
- [ ] Repository metadata (description, topics, homepage) is set
- [ ] CODEOWNERS file is recognized by GitHub
- [ ] Documentation builds successfully
- [ ] Monitoring endpoints are accessible
- [ ] All automated scripts are executable

## Troubleshooting

### Common Issues

1. **Workflow Permissions:**
   - Ensure GitHub Actions has required permissions in repository settings
   - Check token permissions for external services

2. **Secret Access:**
   - Verify secrets are accessible in workflow runs
   - Check secret names match exactly (case-sensitive)

3. **Branch Protection:**
   - Ensure status check names match workflow job names
   - Verify all required checks are configured

### Support Contacts

- **Platform Issues:** platform-team@company.com
- **Security Setup:** security-team@company.com
- **CI/CD Problems:** devops-team@company.com
- **General Support:** support@company.com

## Next Steps

Once manual setup is complete:

1. Run the metrics collection script: `./scripts/collect-metrics.py --summary`
2. Test dependency update automation: `./scripts/automation/dependency-updates.sh --dry-run`
3. Verify code quality monitoring: `./scripts/automation/code-quality-monitor.py --verbose`
4. Review monitoring dashboards and alerts
5. Schedule regular review of quality metrics and trends

This completes the SDLC implementation for MicroDiff-MatDesign!