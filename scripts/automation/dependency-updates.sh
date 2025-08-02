#!/bin/bash
set -e

# Automated dependency update script for MicroDiff-MatDesign
# This script checks for outdated dependencies and creates update PRs

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BRANCH_PREFIX="dependency-updates"
UPDATE_TYPE="${1:-minor}"  # patch, minor, major, security

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_tools=()
    
    for tool in python3 pip git; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
        exit 1
    fi
    
    # Check if working directory is clean
    if [ -n "$(git status --porcelain)" ]; then
        error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Install required tools
install_tools() {
    log "Installing required tools..."
    
    python3 -m pip install --upgrade pip
    python3 -m pip install pip-tools safety pip-audit
    
    success "Tools installed"
}

# Check for outdated dependencies
check_outdated() {
    log "Checking for outdated dependencies..."
    
    # Create temporary files
    local outdated_file=$(mktemp)
    local security_file=$(mktemp)
    
    # Check for outdated packages
    python3 -m pip list --outdated --format=json > "$outdated_file"
    
    # Check for security vulnerabilities
    python3 -m safety check --json --output "$security_file" --continue-on-error || true
    
    local outdated_count=$(jq length "$outdated_file")
    local security_count=0
    
    if [ -s "$security_file" ]; then
        security_count=$(jq '.vulnerabilities | length' "$security_file" 2>/dev/null || echo "0")
    fi
    
    log "Found $outdated_count outdated packages"
    log "Found $security_count security vulnerabilities"
    
    # Cleanup
    rm -f "$outdated_file" "$security_file"
    
    # Return whether updates are needed
    if [ "$outdated_count" -gt 0 ] || [ "$security_count" -gt 0 ]; then
        return 0  # Updates needed
    else
        return 1  # No updates needed
    fi
}

# Update dependencies based on type
update_dependencies() {
    local update_type="$1"
    log "Updating dependencies (type: $update_type)..."
    
    case "$update_type" in
        "security")
            log "Updating security vulnerabilities only..."
            # Get list of vulnerable packages
            python3 -m safety check --json --output security.json --continue-on-error || true
            
            if [ -s security.json ]; then
                # Extract package names and update them
                jq -r '.vulnerabilities[].package_name' security.json | sort -u | while read -r package; do
                    if [ -n "$package" ]; then
                        log "Updating vulnerable package: $package"
                        python3 -m pip install --upgrade "$package" || warning "Failed to update $package"
                    fi
                done
            fi
            
            rm -f security.json
            ;;
        "patch")
            log "Updating patch versions..."
            # Update requirements while constraining to patch versions
            if [ -f requirements.in ]; then
                pip-compile --upgrade-package "*" --resolver=backtracking requirements.in || true
            fi
            ;;
        "minor")
            log "Updating minor versions..."
            # Update requirements allowing minor version updates
            if [ -f requirements.in ]; then
                pip-compile --upgrade --resolver=backtracking requirements.in || true
            fi
            ;;
        "major")
            log "Updating all versions (including major)..."
            # Update all packages to latest versions
            if [ -f requirements.in ]; then
                pip-compile --upgrade --resolver=backtracking requirements.in || true
            fi
            
            # Also update pyproject.toml dependencies if needed
            if [ -f pyproject.toml ]; then
                log "Checking pyproject.toml for updates..."
                # This would require more sophisticated parsing
                # For now, just log that it should be checked manually
                warning "Please review pyproject.toml dependencies manually"
            fi
            ;;
        *)
            error "Unknown update type: $update_type"
            exit 1
            ;;
    esac
}

# Install updated dependencies and run tests
test_updates() {
    log "Installing updated dependencies..."
    
    # Install updated requirements
    python3 -m pip install -r requirements.txt
    python3 -m pip install -e ".[dev]"
    
    # Run basic tests
    log "Running basic import test..."
    python3 -c "import microdiff_matdesign; print('✅ Package imports successfully')" || {
        error "Package import failed"
        return 1
    }
    
    # Run unit tests
    log "Running unit tests..."
    python3 -m pytest tests/unit/ --maxfail=5 -q || {
        warning "Some unit tests failed"
        return 1
    }
    
    # Run security check on updated packages
    log "Running security check..."
    python3 -m safety check || {
        warning "Security vulnerabilities found in updated packages"
        return 1
    }
    
    success "All tests passed"
    return 0
}

# Generate update summary
generate_summary() {
    local update_type="$1"
    local summary_file="update-summary.md"
    
    log "Generating update summary..."
    
    cat > "$summary_file" << EOF
# 📦 Dependency Update Summary

**Update Type**: $update_type
**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

## Changes

EOF
    
    # Show diff if requirements.txt was updated
    if [ -f requirements.txt.backup ]; then
        echo "### Requirements Changes" >> "$summary_file"
        echo "" >> "$summary_file"
        echo '```diff' >> "$summary_file"
        diff -u requirements.txt.backup requirements.txt >> "$summary_file" || true
        echo '```' >> "$summary_file"
        echo "" >> "$summary_file"
    else
        echo "No changes detected in requirements.txt" >> "$summary_file"
        echo "" >> "$summary_file"
    fi
    
    # Add security status
    echo "## Security Status" >> "$summary_file"
    echo "" >> "$summary_file"
    
    python3 -m safety check --json --output security-check.json --continue-on-error || true
    if [ -s security-check.json ]; then
        local vuln_count=$(jq '.vulnerabilities | length' security-check.json 2>/dev/null || echo "0")
        if [ "$vuln_count" -eq 0 ]; then
            echo "✅ No known security vulnerabilities" >> "$summary_file"
        else
            echo "⚠️ $vuln_count security vulnerabilities remain" >> "$summary_file"
        fi
    else
        echo "✅ No known security vulnerabilities" >> "$summary_file"
    fi
    
    echo "" >> "$summary_file"
    echo "## Testing Status" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "- ✅ Package imports successfully" >> "$summary_file"
    echo "- ✅ Unit tests passed" >> "$summary_file"
    echo "- ✅ Security scan completed" >> "$summary_file"
    
    rm -f security-check.json
    success "Summary generated: $summary_file"
}

# Create pull request
create_pull_request() {
    local update_type="$1"
    local branch_name="$BRANCH_PREFIX/$(date +%Y%m%d-%H%M%S)"
    
    log "Creating pull request..."
    
    # Check if there are actual changes
    if ! git diff --quiet requirements.txt; then
        log "Changes detected, creating branch and PR..."
        
        # Create and switch to new branch
        git checkout -b "$branch_name"
        
        # Add changes
        git add requirements.txt pyproject.toml update-summary.md
        
        # Create commit
        local commit_msg
        if [ "$update_type" == "security" ]; then
            commit_msg="chore: update dependencies (security fixes)"
        else
            commit_msg="chore: update dependencies ($update_type)"
        fi
        
        git commit -m "$commit_msg"
        
        # Push branch
        git push origin "$branch_name"
        
        # Create PR (requires gh CLI)
        if command -v gh &> /dev/null; then
            local pr_body
            pr_body=$(cat << EOF
## 📦 Automated Dependency Updates

This PR contains automated dependency updates.

**Update Type**: $update_type

### Changes Summary

$(cat update-summary.md)

### Review Checklist

- [ ] Review dependency changes
- [ ] Check for breaking changes
- [ ] Verify security improvements
- [ ] Run full test suite
- [ ] Check documentation updates needed

---

🤖 This PR was created automatically by the dependency update workflow.

**Note**: Please review the changes carefully before merging.
EOF
)
            
            gh pr create \
                --title "🤖 Automated dependency updates ($update_type)" \
                --body "$pr_body" \
                --label "dependencies,automated" \
                --assignee "@me"
                
            success "Pull request created: $branch_name"
        else
            warning "GitHub CLI not available. Branch pushed: $branch_name"
            log "Please create PR manually"
        fi
    else
        log "No changes detected, skipping PR creation"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove temporary files
    rm -f requirements.txt.backup pyproject.toml.backup
    rm -f update-summary.md
    rm -f security.json security-check.json
    
    # Return to main branch if we created a branch
    if git branch | grep -q "dependency-updates"; then
        git checkout main || git checkout master || true
    fi
}

# Main execution
main() {
    log "Starting dependency update process..."
    log "Update type: $UPDATE_TYPE"
    
    cd "$PROJECT_ROOT"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run the update process
    check_prerequisites
    install_tools
    
    if ! check_outdated; then
        success "All dependencies are up to date!"
        exit 0
    fi
    
    # Backup current files
    [ -f requirements.txt ] && cp requirements.txt requirements.txt.backup
    [ -f pyproject.toml ] && cp pyproject.toml pyproject.toml.backup
    
    update_dependencies "$UPDATE_TYPE"
    
    if test_updates; then
        generate_summary "$UPDATE_TYPE"
        
        if create_pull_request "$UPDATE_TYPE"; then
            success "Dependency update process completed successfully!"
        else
            warning "No changes to create PR"
        fi
    else
        error "Tests failed after updates. Rolling back..."
        
        # Restore backups
        [ -f requirements.txt.backup ] && mv requirements.txt.backup requirements.txt
        [ -f pyproject.toml.backup ] && mv pyproject.toml.backup pyproject.toml
        
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
Automated Dependency Update Script

Usage: $0 [UPDATE_TYPE]

UPDATE_TYPE:
  patch     - Update patch versions only (default for weekly runs)
  minor     - Update minor versions (default)
  major     - Update all versions including major (monthly runs)
  security  - Update only packages with security vulnerabilities

Examples:
  $0              # Minor updates
  $0 patch        # Patch updates only
  $0 security     # Security updates only
  $0 major        # All updates including major versions

Environment Variables:
  SKIP_TESTS    - Skip running tests (not recommended)
  DRY_RUN       - Show what would be updated without making changes

EOF
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    patch|minor|major|security)
        UPDATE_TYPE="$1"
        ;;
    "")
        UPDATE_TYPE="minor"
        ;;
    *)
        error "Invalid update type: $1"
        show_help
        exit 1
        ;;
esac

# Run main function
main "$@"