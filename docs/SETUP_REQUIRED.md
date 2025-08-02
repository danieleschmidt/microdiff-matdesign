# 🔧 Required Manual Setup

This document outlines the manual setup steps required after the automated SDLC implementation due to GitHub App permission limitations.

## 🚨 Critical Setup Steps

### 1. GitHub Actions Workflows

**Status**: ⚠️ MANUAL ACTION REQUIRED

The automated SDLC implementation has created comprehensive workflow templates in `docs/workflows/examples/`, but these need to be manually copied to `.github/workflows/` due to GitHub App permissions.

**Action Required**:
```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/*.yml .github/workflows/

# Commit and push
git add .github/workflows/
git commit -m "feat: add CI/CD workflows from templates"
git push
```

**Required Workflows**:
- [ ] `ci.yml` - Continuous Integration
- [ ] `cd.yml` - Continuous Deployment  
- [ ] `security-scan.yml` - Security Scanning
- [ ] `dependency-update.yml` - Dependency Updates

### 2. GitHub Repository Settings

**Status**: ⚠️ MANUAL ACTION REQUIRED

Configure repository settings that cannot be automated:

#### Branch Protection Rules
Navigate to: `Settings > Branches > Add rule`

**Main Branch Protection**:
- [ ] Branch name pattern: `main`
- [ ] Require pull request reviews before merging
  - [ ] Required approving reviews: 2
  - [ ] Dismiss stale PR approvals when new commits are pushed
  - [ ] Require review from code owners
- [ ] Require status checks to pass before merging
  - [ ] Require branches to be up to date before merging
  - [ ] Required status checks:
    - `CI Success`
    - `Security Scan`
    - `Code Quality`
- [ ] Require signed commits
- [ ] Require linear history
- [ ] Include administrators

#### Repository Secrets

Navigate to: `Settings > Secrets and variables > Actions`

**Required Secrets**:
```
# Container Registry (GitHub provides automatically)
GITHUB_TOKEN                    # ✅ Automatic

# Deployment
STAGING_KUBECONFIG             # ⚠️ Base64 encoded kubeconfig
PRODUCTION_KUBECONFIG          # ⚠️ Base64 encoded kubeconfig

# Monitoring & Notifications
SLACK_WEBHOOK_URL              # ⚠️ Team notifications
SECURITY_SLACK_WEBHOOK_URL     # ⚠️ Security alerts
GRAFANA_API_KEY               # ⚠️ Deployment annotations

# External Services
CODECOV_TOKEN                 # ⚠️ Code coverage reporting
SNYK_TOKEN                    # ⚠️ Snyk security scanning
```

#### Environments

Navigate to: `Settings > Environments`

**Create Environments**:
- [ ] `staging`
  - [ ] Required reviewers: 1
  - [ ] Wait timer: 0 minutes
  - [ ] Environment secrets: `STAGING_KUBECONFIG`
- [ ] `production`
  - [ ] Required reviewers: 2
  - [ ] Wait timer: 5 minutes
  - [ ] Environment secrets: `PRODUCTION_KUBECONFIG`

#### Repository Features

Navigate to: `Settings > General`

**Enable Features**:
- [ ] Issues
- [ ] Sponsorships  
- [ ] Discussions
- [ ] Projects

**Security & Analysis**:
- [ ] Dependency graph
- [ ] Dependabot alerts
- [ ] Dependabot security updates
- [ ] Code scanning (CodeQL)
- [ ] Secret scanning

### 3. GitHub Discussions

**Status**: ⚠️ MANUAL ACTION REQUIRED

Enable and configure GitHub Discussions:

1. Go to `Settings > Features > Discussions`
2. Click "Set up discussions"
3. Create categories:
   - [ ] 🎯 Project Planning
   - [ ] 🐛 Bug Reports
   - [ ] 💡 Feature Requests
   - [ ] ❓ Questions & Help
   - [ ] 📢 Announcements
   - [ ] 🔬 Research & Development

### 4. Issue and PR Templates

**Status**: ✅ COMPLETED

Issue and PR templates have been created automatically. Verify they appear correctly:

- [ ] Bug report template
- [ ] Feature request template  
- [ ] Pull request template

### 5. Repository Topics and Description

**Status**: ⚠️ MANUAL ACTION REQUIRED

Navigate to: Repository main page

**Update Repository Information**:
- [ ] Description: "Diffusion model framework for inverse material design that transforms micro-CT images into printable alloy process parameters"
- [ ] Website: Your project homepage (if any)
- [ ] Topics: `diffusion-models`, `materials-science`, `inverse-design`, `pytorch`, `python`, `machine-learning`, `scientific-computing`

### 6. License Configuration

**Status**: ✅ COMPLETED

MIT License has been configured. GitHub should automatically detect it.

### 7. Security Configuration

**Status**: ⚠️ MANUAL ACTION REQUIRED

#### Security Policy
- [ ] Verify `SECURITY.md` is recognized by GitHub
- [ ] Navigate to `Security > Security advisories` and review settings

#### Dependency Scanning
- [ ] Enable Dependabot version updates in `Settings > Security & analysis`
- [ ] Review and customize `.github/dependabot.yml` if created

### 8. Monitoring Integration

**Status**: ⚠️ MANUAL ACTION REQUIRED

#### External Monitoring Services

**Codecov Setup**:
1. Visit [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Copy the upload token
4. Add as `CODECOV_TOKEN` secret

**Slack Integration**:
1. Create Slack app or incoming webhook
2. Add webhook URLs as secrets:
   - `SLACK_WEBHOOK_URL`
   - `SECURITY_SLACK_WEBHOOK_URL`

**Monitoring Services** (Optional):
- [ ] Sentry for error tracking
- [ ] DataDog for application monitoring
- [ ] New Relic for performance monitoring

### 9. Documentation Review

**Status**: ✅ COMPLETED

Review automatically generated documentation:

- [ ] `README.md` - Project overview and quick start
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `docs/ARCHITECTURE.md` - System architecture
- [ ] `docs/ROADMAP.md` - Project roadmap
- [ ] Documentation in `docs/` directory

## 🔄 Validation Checklist

After completing manual setup, validate the implementation:

### Repository Structure
- [ ] All essential files present
- [ ] Documentation is comprehensive
- [ ] License is properly configured

### CI/CD Pipeline
- [ ] Workflows execute successfully
- [ ] Branch protection is working
- [ ] Required status checks pass
- [ ] Deployments work correctly

### Security
- [ ] Security scanning is enabled
- [ ] Secrets are properly configured
- [ ] Vulnerability alerts are working
- [ ] Branch protection prevents direct pushes

### Monitoring
- [ ] Metrics collection is working
- [ ] Alerts are configured
- [ ] Dashboards are accessible
- [ ] Notifications are delivered

### Team Collaboration
- [ ] Code review process is enforced
- [ ] Issue templates work correctly
- [ ] PR templates are helpful
- [ ] CODEOWNERS is respected

## 📞 Support and Troubleshooting

### Common Issues

**1. Workflow Failures**
```bash
# Check workflow syntax
gh workflow list
gh workflow view ci.yml

# Check secrets
gh secret list
```

**2. Branch Protection Issues**
- Verify status check names match workflow job names
- Ensure admin bypass is configured correctly
- Check that required reviewers are available

**3. Permission Problems**
- Review GitHub App permissions
- Check repository collaborator access
- Verify team memberships

### Getting Help

1. **GitHub Documentation**: [docs.github.com](https://docs.github.com)
2. **Workflow Troubleshooting**: Check Actions tab for detailed logs
3. **Security Issues**: Contact security team immediately
4. **Repository Issues**: Create issue with `setup` label

## 📋 Completion Status

Track your progress through the setup:

- [ ] **Phase 1**: GitHub Actions workflows copied and working
- [ ] **Phase 2**: Repository settings configured
- [ ] **Phase 3**: Security features enabled
- [ ] **Phase 4**: Monitoring integrated
- [ ] **Phase 5**: Team access configured
- [ ] **Phase 6**: Documentation reviewed and updated

**Estimated Setup Time**: 2-4 hours depending on team size and complexity

---

**Important**: This setup is crucial for the SDLC to function properly. Do not skip any required steps as they ensure security, quality, and operational excellence for your project.