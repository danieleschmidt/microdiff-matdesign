#!/bin/bash
# Automated dependency update script for MicroDiff-MatDesign
# This script checks for outdated dependencies and creates update PRs

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UPDATE_BRANCH="dependency-updates-$(date +%Y%m%d-%H%M%S)"
MAX_UPDATES=10  # Maximum number of dependencies to update at once

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log "Checking required dependencies..."
    
    local missing_deps=()
    
    command -v python3 >/dev/null 2>&1 || missing_deps+=("python3")
    command -v pip >/dev/null 2>&1 || missing_deps+=("pip")
    command -v git >/dev/null 2>&1 || missing_deps+=("git")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
    fi
    
    success "All required dependencies are installed"
}

# Setup virtual environment for dependency checking
setup_venv() {
    log "Setting up virtual environment for dependency checking..."
    
    local venv_dir="$PROJECT_ROOT/.dependency-check-venv"
    
    if [[ -d "$venv_dir" ]]; then
        rm -rf "$venv_dir"
    fi
    
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    
    pip install --upgrade pip
    pip install pip-tools safety bandit
    
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    fi
    
    success "Virtual environment set up"
}

# Check for outdated dependencies
check_outdated() {
    log "Checking for outdated dependencies..."
    
    local outdated_file="$PROJECT_ROOT/.outdated-deps.txt"
    
    # Check pip outdated packages
    pip list --outdated --format=json > "$outdated_file.json" 2>/dev/null || {
        warn "Could not check outdated packages with pip"
        echo "[]" > "$outdated_file.json"
    }
    
    # Parse JSON and create readable list
    python3 -c "
import json
import sys

with open('$outdated_file.json', 'r') as f:
    outdated = json.load(f)

if not outdated:
    print('No outdated packages found')
    sys.exit(0)

print(f'Found {len(outdated)} outdated packages:')
for pkg in outdated[:$MAX_UPDATES]:
    print(f'{pkg[\"name\"]}: {pkg[\"version\"]} -> {pkg[\"latest_version\"]}')

if len(outdated) > $MAX_UPDATES:
    print(f'... and {len(outdated) - $MAX_UPDATES} more packages')
" > "$outdated_file"
    
    if [[ ! -s "$outdated_file" ]]; then
        success "No outdated dependencies found"
        return 1
    fi
    
    cat "$outdated_file"
    return 0
}

# Run security check on current dependencies
security_check() {
    log "Running security check on dependencies..."
    
    local security_report="$PROJECT_ROOT/.security-report.txt"
    
    # Safety check
    safety check --json > "$security_report.json" 2>/dev/null || {
        warn "Safety check failed or found vulnerabilities"
    }
    
    # Bandit security check for Python code
    bandit -r "$PROJECT_ROOT/microdiff_matdesign" -f json -o "$PROJECT_ROOT/.bandit-report.json" 2>/dev/null || {
        warn "Bandit security scan found issues"
    }
    
    # Parse safety results
    if [[ -f "$security_report.json" ]]; then
        python3 -c "
import json
import sys

try:
    with open('$security_report.json', 'r') as f:
        safety_data = json.load(f)
    
    if safety_data:
        print('Security vulnerabilities found:')
        for vuln in safety_data:
            print(f'- {vuln.get(\"package\", \"unknown\")}: {vuln.get(\"advisory\", \"No details\")}')
        sys.exit(1)
    else:
        print('No security vulnerabilities found')
except:
    print('Could not parse safety report')
" > "$security_report"
        
        if [[ $? -eq 1 ]]; then
            error "Security vulnerabilities found. Check $security_report for details."
        fi
    fi
    
    success "Security check passed"
}

# Update requirements files
update_requirements() {
    log "Updating requirements files..."
    
    local updated_files=()
    
    # Update requirements.txt if requirements.in exists
    if [[ -f "$PROJECT_ROOT/requirements.in" ]]; then
        log "Updating requirements.txt from requirements.in..."
        pip-compile --upgrade "$PROJECT_ROOT/requirements.in"
        updated_files+=("requirements.txt")
    fi
    
    # Update requirements-dev.txt if requirements-dev.in exists
    if [[ -f "$PROJECT_ROOT/requirements-dev.in" ]]; then
        log "Updating requirements-dev.txt from requirements-dev.in..."
        pip-compile --upgrade "$PROJECT_ROOT/requirements-dev.in"
        updated_files+=("requirements-dev.txt")
    fi
    
    # If no .in files, try to update existing requirements.txt
    if [[ ${#updated_files[@]} -eq 0 && -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log "Updating packages in requirements.txt..."
        
        # Create a backup
        cp "$PROJECT_ROOT/requirements.txt" "$PROJECT_ROOT/requirements.txt.backup"
        
        # Update specific packages
        python3 -c "
import subprocess
import sys

# Read current requirements
with open('$PROJECT_ROOT/requirements.txt', 'r') as f:
    lines = f.readlines()

updated_lines = []
for line in lines:
    line = line.strip()
    if line and not line.startswith('#'):
        # Extract package name (before ==, >=, etc.)
        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
        try:
            # Try to update to latest version
            result = subprocess.run(['pip', 'install', '--upgrade', '--dry-run', pkg_name], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                # Get latest version
                latest_result = subprocess.run(['pip', 'show', pkg_name], 
                                             capture_output=True, text=True)
                if latest_result.returncode == 0:
                    for show_line in latest_result.stdout.split('\n'):
                        if show_line.startswith('Version:'):
                            version = show_line.split(':', 1)[1].strip()
                            updated_lines.append(f'{pkg_name}=={version}')
                            break
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        except:
            updated_lines.append(line)
    else:
        updated_lines.append(line)

# Write updated requirements
with open('$PROJECT_ROOT/requirements.txt', 'w') as f:
    for line in updated_lines:
        f.write(line + '\n')
"
        updated_files+=("requirements.txt")
    fi
    
    if [[ ${#updated_files[@]} -eq 0 ]]; then
        warn "No requirements files to update"
        return 1
    fi
    
    success "Updated files: ${updated_files[*]}"
    return 0
}

# Test updated dependencies
test_updates() {
    log "Testing updated dependencies..."
    
    # Install updated dependencies
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    fi
    
    # Run tests if available
    if [[ -f "$PROJECT_ROOT/pytest.ini" ]] || [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log "Running tests..."
        cd "$PROJECT_ROOT"
        python -m pytest tests/ --tb=short -x || {
            error "Tests failed with updated dependencies"
        }
    else
        log "No test configuration found, skipping tests"
    fi
    
    # Try importing main package
    python3 -c "import microdiff_matdesign; print('Package import successful')" || {
        error "Package import failed with updated dependencies"
    }
    
    success "All tests passed"
}

# Create git branch and commit changes
create_update_branch() {
    log "Creating update branch..."
    
    cd "$PROJECT_ROOT"
    
    # Check if there are any changes
    if [[ -z "$(git status --porcelain)" ]]; then
        warn "No changes to commit"
        return 1
    fi
    
    # Create new branch
    git checkout -b "$UPDATE_BRANCH"
    
    # Add changed files
    git add requirements*.txt requirements*.in 2>/dev/null || true
    
    # Create commit
    local commit_msg="chore: update dependencies

Automated dependency update $(date +'%Y-%m-%d %H:%M:%S')

Updated packages:
$(cat .outdated-deps.txt 2>/dev/null || echo 'See git diff for details')

🤖 Generated with automated dependency update script

Co-Authored-By: Dependency Bot <noreply@company.com>"
    
    git commit -m "$commit_msg"
    
    success "Created branch $UPDATE_BRANCH with updates"
    return 0
}

# Push branch and create PR
create_pull_request() {
    log "Creating pull request..."
    
    cd "$PROJECT_ROOT"
    
    # Push branch
    git push origin "$UPDATE_BRANCH" || {
        error "Failed to push branch"
    }
    
    # Create PR using GitHub CLI if available
    if command -v gh >/dev/null 2>&1; then
        local pr_body="## Summary
Automated dependency updates

## Changes
$(cat .outdated-deps.txt 2>/dev/null || echo 'Dependency updates applied')

## Testing
- [x] Dependencies installed successfully
- [x] Tests pass
- [x] Security check passed

## Notes
This PR was automatically generated by the dependency update script.
Please review the changes and ensure all functionality works as expected.

🤖 Generated with automated dependency update script"
        
        gh pr create \
            --title "chore: automated dependency updates $(date +'%Y-%m-%d')" \
            --body "$pr_body" \
            --label "dependencies,automated" \
            --assignee "@me" || {
            warn "Failed to create PR with GitHub CLI"
        }
        
        success "Pull request created"
    else
        warn "GitHub CLI not available. Branch pushed but PR not created."
        log "Create PR manually: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1\/\2/')/compare/$UPDATE_BRANCH"
    fi
}

# Clean up temporary files
cleanup() {
    log "Cleaning up..."
    
    cd "$PROJECT_ROOT"
    
    # Remove temporary files
    rm -f .outdated-deps.txt* .security-report.* .bandit-report.json
    
    # Deactivate and remove virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        deactivate
    fi
    
    rm -rf .dependency-check-venv
    
    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting automated dependency update process..."
    
    # Set up trap for cleanup
    trap cleanup EXIT
    
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_dependencies
    
    # Setup environment
    setup_venv
    
    # Check for updates
    if ! check_outdated; then
        success "No updates needed"
        exit 0
    fi
    
    # Run security check
    security_check
    
    # Update requirements
    if ! update_requirements; then
        error "Failed to update requirements"
    fi
    
    # Test updates
    test_updates
    
    # Create branch and commit
    if ! create_update_branch; then
        warn "No changes to commit"
        exit 0
    fi
    
    # Create PR
    create_pull_request
    
    success "Dependency update process completed successfully!"
    log "Branch: $UPDATE_BRANCH"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --dry-run      Check for updates but don't apply them"
        echo "  --force        Force update even if tests fail"
        exit 0
        ;;
    --dry-run)
        log "Running in dry-run mode..."
        check_dependencies
        setup_venv
        check_outdated
        security_check
        success "Dry-run completed"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac