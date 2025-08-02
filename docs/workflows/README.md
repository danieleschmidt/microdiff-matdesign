# GitHub Actions Workflows

This directory contains comprehensive GitHub Actions workflow templates and documentation for the MicroDiff-MatDesign project.

## 🚀 Available Workflow Templates

### Core CI/CD Workflows

| Workflow | Purpose | Trigger | Duration |
|----------|---------|---------|----------|
| **[ci.yml](examples/ci.yml)** | Continuous Integration | Push, PR | ~15-30 min |
| **[cd.yml](examples/cd.yml)** | Continuous Deployment | Main push, Tags | ~20-45 min |
| **[security-scan.yml](examples/security-scan.yml)** | Security Scanning | Push, PR, Schedule | ~20-30 min |
| **[dependency-update.yml](examples/dependency-update.yml)** | Dependency Updates | Schedule, Manual | ~10-20 min |

### Workflow Features

#### 📋 Continuous Integration (ci.yml)
- **Multi-Python testing** (3.8, 3.9, 3.10, 3.11)
- **Code quality checks** (Black, Ruff, MyPy)
- **Security scanning** (Bandit, Safety, Semgrep)
- **Comprehensive testing** (Unit, Integration, E2E)
- **Performance benchmarks**
- **Docker image building and testing**
- **Documentation verification**
- **License compliance checking**
- **Parallel execution** for faster feedback

#### 🚀 Continuous Deployment (cd.yml)
- **Environment-specific deployments** (Staging, Production)
- **Blue-green deployment strategy**
- **Container security scanning**
- **Automated rollback capability**
- **Health checks and smoke tests**
- **Image signing with Cosign**
- **SBOM generation**
- **Deployment notifications**

#### 🔒 Security Scanning (security-scan.yml)
- **Static Application Security Testing (SAST)**
  - CodeQL analysis
  - Bandit security linting
  - Semgrep static analysis
  - Secret detection with TruffleHog
- **Software Composition Analysis (SCA)**
  - Safety vulnerability checking
  - Pip-audit dependency scanning
  - Snyk vulnerability scanning
  - License compliance verification
- **Container Security**
  - Trivy vulnerability scanning
  - Anchore Grype scanning
  - Docker Scout analysis
  - Hadolint Dockerfile linting
- **Infrastructure as Code (IaC)**
  - Checkov security scanning
  - Terrascan analysis
- **Dynamic Application Security Testing (DAST)**
  - OWASP ZAP scanning

#### 📦 Dependency Updates (dependency-update.yml)
- **Automated dependency updates**
  - Patch, minor, major version updates
  - Security-focused updates
  - Smart scheduling (weekly/monthly)
- **Comprehensive testing**
  - Multi-Python version validation
  - Security vulnerability verification
  - Performance impact assessment
- **Pull request automation**
  - Detailed change summaries
  - Security impact analysis
  - Automatic reviewer assignment
- **Rollback capabilities**

## 🔧 Setup Instructions

### Prerequisites

1. **Repository Structure**
   ```
   .github/
   └── workflows/
       ├── ci.yml
       ├── cd.yml
       ├── security-scan.yml
       └── dependency-update.yml
   ```

2. **Required Secrets**
   ```bash
   # Container Registry
   GITHUB_TOKEN                    # Automatic (GitHub provides)
   
   # Deployment
   STAGING_KUBECONFIG             # Base64 encoded kubeconfig
   PRODUCTION_KUBECONFIG          # Base64 encoded kubeconfig
   
   # Monitoring & Notifications
   SLACK_WEBHOOK_URL              # Slack notifications
   SECURITY_SLACK_WEBHOOK_URL     # Security-specific notifications
   GRAFANA_API_KEY               # Deployment annotations
   
   # External Services
   CODECOV_TOKEN                 # Code coverage reporting
   SNYK_TOKEN                    # Snyk security scanning
   ```

3. **Repository Settings**
   - Enable GitHub Actions
   - Configure branch protection rules
   - Set up required status checks
   - Enable security alerts
   - Configure Dependabot (optional, complements automated updates)

### Manual Setup Steps

1. **Copy Workflow Files**
   ```bash
   mkdir -p .github/workflows
   cp docs/workflows/examples/*.yml .github/workflows/
   ```

2. **Configure Secrets**
   - Go to Repository Settings → Secrets and variables → Actions
   - Add all required secrets listed above

3. **Set Up Environments**
   - Create `staging` and `production` environments
   - Configure protection rules for production
   - Add environment-specific secrets

4. **Configure Branch Protection**
   ```json
   {
     "required_status_checks": {
       "strict": true,
       "contexts": [
         "CI Success",
         "Security Scan",
         "Code Quality"
       ]
     },
     "enforce_admins": true,
     "required_pull_request_reviews": {
       "required_approving_review_count": 2,
       "dismiss_stale_reviews": true,
       "require_code_owner_reviews": true
     },
     "restrictions": null
   }
   ```

## 📊 Workflow Monitoring

### Success Metrics
- **CI Success Rate**: > 95%
- **Deployment Success Rate**: > 99%
- **Security Scan Coverage**: 100%
- **Mean Time to Deployment**: < 30 minutes
- **Mean Time to Recovery**: < 15 minutes

### Key Performance Indicators
- Build duration trends
- Test coverage percentage
- Security vulnerability count
- Deployment frequency
- Failed deployment count

### Alerting Setup
Configure alerts for:
- Failed main branch builds
- Security vulnerabilities discovered
- Deployment failures
- Long-running workflows
- Dependency update conflicts

## 🛡️ Security Considerations

### Security Scanning Coverage
- **Code**: Static analysis, secret detection
- **Dependencies**: Vulnerability scanning, license compliance
- **Containers**: Image security, best practices
- **Infrastructure**: IaC security scanning
- **Runtime**: Dynamic application security testing

### Security Policies
- All workflows run with minimal required permissions
- Secrets are properly scoped and rotated
- Container images are signed and verified
- Deployment requires security scan approval
- Emergency procedures for security incidents

## 🔄 Rollback Procedures

### Workflow Failure Recovery

1. **Disable Problematic Workflow**
   ```bash
   # Via GitHub CLI
   gh workflow disable "workflow-name.yml"
   ```

2. **Emergency Hotfix**
   ```bash
   git checkout main
   git checkout -b hotfix/emergency-fix
   # Make minimal required changes
   git commit -m "hotfix: emergency fix"
   git push origin hotfix/emergency-fix
   # Create PR with bypass reviews (if configured)
   ```

3. **Deployment Rollback**
   ```bash
   # Automatic rollback triggered by health check failures
   # Manual rollback via workflow_dispatch
   gh workflow run cd.yml -f environment=production -f rollback=true
   ```

### Recovery Checklist
- [ ] Identify root cause
- [ ] Implement immediate fix
- [ ] Verify fix in staging
- [ ] Deploy fix to production
- [ ] Monitor for stability
- [ ] Document incident
- [ ] Update procedures

## 📈 Optimization Tips

### Performance Optimization
- Use matrix strategies for parallel execution
- Cache dependencies and build artifacts
- Optimize Docker layer caching
- Use conditional job execution
- Implement smart test selection

### Cost Optimization
- Right-size runner instances
- Use self-hosted runners for large workloads
- Implement workflow timeout limits
- Cache external dependencies
- Optimize container image sizes

### Developer Experience
- Provide clear workflow status feedback
- Include detailed failure messages
- Generate comprehensive test reports
- Automatic PR comments with results
- Integration with IDE/development tools

## 🤝 Contributing to Workflows

### Modification Guidelines
1. Test changes in feature branches
2. Validate with realistic scenarios
3. Update documentation
4. Consider backward compatibility
5. Review security implications

### Best Practices
- Use semantic versioning for action references
- Include timeout limits for all jobs
- Implement proper error handling
- Use descriptive job and step names
- Include comprehensive logging

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/learn-github-actions/workflow-syntax-for-github-actions)
- [Action Marketplace](https://github.com/marketplace?type=actions)

---

**Note**: These workflows require manual setup due to GitHub App permission limitations. Repository maintainers must create the actual workflow files from these templates.