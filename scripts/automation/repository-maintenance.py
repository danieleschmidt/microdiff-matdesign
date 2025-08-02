#!/usr/bin/env python3
"""
Repository maintenance automation script for MicroDiff-MatDesign.

This script performs various repository maintenance tasks including:
- Cleaning up old branches
- Updating documentation
- Checking for stale issues and PRs
- Repository health monitoring
"""

import os
import sys
import json
import subprocess
import requests
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import time

class RepositoryMaintenance:
    """Automated repository maintenance operations."""
    
    def __init__(self, repo_path: str = ".", github_token: Optional[str] = None):
        self.repo_path = Path(repo_path).resolve()
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.repo_info = self._get_repo_info()
        
    def _get_repo_info(self) -> Dict[str, str]:
        """Get repository information from git."""
        try:
            # Get remote URL
            result = subprocess.run([
                'git', 'config', '--get', 'remote.origin.url'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                print("❌ Could not get repository URL")
                return {}
            
            url = result.stdout.strip()
            
            # Parse GitHub repository from URL
            if 'github.com' in url:
                if url.startswith('git@'):
                    # SSH format: git@github.com:owner/repo.git
                    parts = url.split(':')[1].replace('.git', '').split('/')
                elif url.startswith('https://'):
                    # HTTPS format: https://github.com/owner/repo.git
                    parts = url.replace('https://github.com/', '').replace('.git', '').split('/')
                else:
                    return {}
                
                if len(parts) >= 2:
                    return {
                        'owner': parts[0],
                        'repo': parts[1],
                        'full_name': f"{parts[0]}/{parts[1]}"
                    }
            
            return {}
        except Exception as e:
            print(f"❌ Error getting repository info: {e}")
            return {}
    
    def _github_api_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to the GitHub API."""
        if not self.github_token or not self.repo_info:
            return None
        
        url = f"https://api.github.com/repos/{self.repo_info['full_name']}/{endpoint}"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method == 'PATCH':
                response = requests.patch(url, headers=headers, json=data)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ GitHub API request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"⚠️ Error making GitHub API request: {e}")
            return None
    
    def clean_old_branches(self, days_old: int = 30, dry_run: bool = False) -> List[str]:
        """Clean up old merged branches."""
        print(f"🧹 Cleaning branches older than {days_old} days...")
        
        deleted_branches = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        
        try:
            # Get list of merged branches
            result = subprocess.run([
                'git', 'branch', '--merged', 'main'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                # Try with master
                result = subprocess.run([
                    'git', 'branch', '--merged', 'master'
                ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                branches = [line.strip().replace('*', '').strip() 
                           for line in result.stdout.split('\n') 
                           if line.strip() and not line.strip().startswith('*')]
                
                for branch in branches:
                    if branch in ['main', 'master', 'develop']:
                        continue
                    
                    # Get last commit date for branch
                    result = subprocess.run([
                        'git', 'log', '-1', '--format=%ct', branch
                    ], capture_output=True, text=True, cwd=self.repo_path)
                    
                    if result.returncode == 0:
                        try:
                            commit_timestamp = int(result.stdout.strip())
                            commit_date = datetime.datetime.fromtimestamp(commit_timestamp)
                            
                            if commit_date < cutoff_date:
                                if dry_run:
                                    print(f"  Would delete: {branch} (last commit: {commit_date.date()})")
                                    deleted_branches.append(branch)
                                else:
                                    # Delete local branch
                                    delete_result = subprocess.run([
                                        'git', 'branch', '-d', branch
                                    ], capture_output=True, text=True, cwd=self.repo_path)
                                    
                                    if delete_result.returncode == 0:
                                        print(f"  ✅ Deleted branch: {branch}")
                                        deleted_branches.append(branch)
                                    else:
                                        print(f"  ⚠️ Could not delete branch: {branch}")
                        except ValueError:
                            pass
            
        except Exception as e:
            print(f"❌ Error cleaning branches: {e}")
        
        return deleted_branches
    
    def check_stale_issues(self, days_stale: int = 90) -> List[Dict]:
        """Check for stale issues and PRs."""
        print(f"🕸️ Checking for stale issues (older than {days_stale} days)...")
        
        stale_items = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_stale)
        
        # Check issues
        issues = self._github_api_request('issues?state=open&per_page=100')
        if issues:
            for issue in issues:
                updated_at = datetime.datetime.strptime(
                    issue['updated_at'], '%Y-%m-%dT%H:%M:%SZ'
                )
                
                if updated_at < cutoff_date:
                    stale_items.append({
                        'type': 'issue',
                        'number': issue['number'],
                        'title': issue['title'],
                        'updated_at': updated_at,
                        'url': issue['html_url']
                    })
        
        # Check pull requests
        prs = self._github_api_request('pulls?state=open&per_page=100')
        if prs:
            for pr in prs:
                updated_at = datetime.datetime.strptime(
                    pr['updated_at'], '%Y-%m-%dT%H:%M:%SZ'
                )
                
                if updated_at < cutoff_date:
                    stale_items.append({
                        'type': 'pull_request',
                        'number': pr['number'],
                        'title': pr['title'],
                        'updated_at': updated_at,
                        'url': pr['html_url']
                    })
        
        if stale_items:
            print(f"  Found {len(stale_items)} stale items:")
            for item in stale_items:
                print(f"    {item['type'].upper()} #{item['number']}: {item['title']}")
        else:
            print("  ✅ No stale issues or PRs found")
        
        return stale_items
    
    def add_stale_labels(self, stale_items: List[Dict], dry_run: bool = False) -> int:
        """Add stale labels to old issues and PRs."""
        print("🏷️ Adding stale labels...")
        
        labeled_count = 0
        
        for item in stale_items:
            if dry_run:
                print(f"  Would label {item['type']} #{item['number']}: {item['title']}")
                labeled_count += 1
            else:
                # Add stale label
                endpoint = f"issues/{item['number']}/labels"
                data = {'labels': ['stale']}
                
                result = self._github_api_request(endpoint, method='POST', data=data)
                if result:
                    print(f"  ✅ Labeled {item['type']} #{item['number']}")
                    labeled_count += 1
                else:
                    print(f"  ⚠️ Could not label {item['type']} #{item['number']}")
        
        return labeled_count
    
    def update_repository_metadata(self) -> bool:
        """Update repository description, topics, and other metadata."""
        print("📝 Updating repository metadata...")
        
        try:
            # Read project configuration
            pyproject_path = self.repo_path / 'pyproject.toml'
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                
                # Extract description and keywords from pyproject.toml
                description = None
                topics = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('description ='):
                        description = line.split('=', 1)[1].strip().strip('"\'')
                    elif 'classifiers' in line:
                        # Extract topics from classifiers
                        if 'Machine Learning' in content or 'Artificial Intelligence' in content:
                            topics.extend(['machine-learning', 'ai'])
                        if 'Scientific' in content:
                            topics.extend(['scientific-computing', 'research'])
                
                # Add project-specific topics
                topics.extend([
                    'diffusion-models',
                    'materials-science',
                    'inverse-design',
                    'pytorch',
                    'python'
                ])
                
                # Remove duplicates
                topics = list(set(topics))
                
                # Update repository via API
                if self.github_token and description:
                    data = {
                        'description': description,
                        'topics': topics[:20]  # GitHub limits to 20 topics
                    }
                    
                    result = self._github_api_request('', method='PATCH', data=data)
                    if result:
                        print(f"  ✅ Updated description: {description}")
                        print(f"  ✅ Updated topics: {', '.join(topics[:10])}")
                        return True
                    else:
                        print("  ⚠️ Could not update repository metadata")
        
        except Exception as e:
            print(f"❌ Error updating repository metadata: {e}")
        
        return False
    
    def check_repository_health(self) -> Dict[str, Any]:
        """Check overall repository health."""
        print("🏥 Checking repository health...")
        
        health_report = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check README exists and is recent
        readme_path = self.repo_path / 'README.md'
        if readme_path.exists():
            stat = readme_path.stat()
            age_days = (time.time() - stat.st_mtime) / (24 * 3600)
            health_report['checks']['readme'] = {
                'exists': True,
                'age_days': round(age_days, 1),
                'status': 'good' if age_days < 180 else 'stale'
            }
        else:
            health_report['checks']['readme'] = {
                'exists': False,
                'status': 'missing'
            }
        
        # Check for essential files
        essential_files = [
            'LICENSE',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'SECURITY.md',
            'pyproject.toml',
            'requirements.txt'
        ]
        
        missing_files = []
        for file_name in essential_files:
            if not (self.repo_path / file_name).exists():
                missing_files.append(file_name)
        
        health_report['checks']['essential_files'] = {
            'total': len(essential_files),
            'present': len(essential_files) - len(missing_files),
            'missing': missing_files,
            'status': 'good' if len(missing_files) == 0 else 'incomplete'
        }
        
        # Check git configuration
        try:
            # Check if there are any commits
            result = subprocess.run([
                'git', 'rev-list', '--count', 'HEAD'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Check recent activity
            result = subprocess.run([
                'git', 'log', '--since=30.days.ago', '--oneline'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            health_report['checks']['git_activity'] = {
                'total_commits': commit_count,
                'recent_commits': recent_commits,
                'status': 'active' if recent_commits > 0 else 'inactive'
            }
            
        except Exception:
            health_report['checks']['git_activity'] = {
                'status': 'error'
            }
        
        # Check branch protection (if GitHub token is available)
        if self.github_token and self.repo_info:
            branch_protection = self._github_api_request('branches/main/protection')
            health_report['checks']['branch_protection'] = {
                'enabled': branch_protection is not None,
                'status': 'protected' if branch_protection else 'unprotected'
            }
        
        # Generate overall health score
        scores = []
        for check in health_report['checks'].values():
            if check['status'] in ['good', 'active', 'protected']:
                scores.append(1)
            elif check['status'] in ['incomplete', 'inactive', 'unprotected']:
                scores.append(0.5)
            else:
                scores.append(0)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        health_report['overall_score'] = round(overall_score * 100, 1)
        health_report['overall_status'] = (
            'excellent' if overall_score >= 0.9 else
            'good' if overall_score >= 0.7 else
            'fair' if overall_score >= 0.5 else
            'poor'
        )
        
        print(f"  Overall health score: {health_report['overall_score']}% ({health_report['overall_status']})")
        
        return health_report
    
    def generate_maintenance_report(self, operations_performed: Dict[str, Any]) -> str:
        """Generate a maintenance report."""
        print("📋 Generating maintenance report...")
        
        report_lines = [
            "# 🔧 Repository Maintenance Report",
            "",
            f"**Generated**: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Repository**: {self.repo_info.get('full_name', 'Unknown')}",
            "",
            "## Operations Performed",
            ""
        ]
        
        for operation, details in operations_performed.items():
            report_lines.append(f"### {operation.replace('_', ' ').title()}")
            report_lines.append("")
            
            if isinstance(details, list):
                if details:
                    for item in details:
                        report_lines.append(f"- {item}")
                else:
                    report_lines.append("- No items found")
            elif isinstance(details, dict):
                for key, value in details.items():
                    report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            else:
                report_lines.append(f"- {details}")
            
            report_lines.append("")
        
        return '\n'.join(report_lines)
    
    def run_maintenance(self, operations: List[str], dry_run: bool = False) -> Dict[str, Any]:
        """Run the complete maintenance process."""
        print("🚀 Starting repository maintenance...")
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        results = {}
        
        if 'clean_branches' in operations:
            results['clean_branches'] = self.clean_old_branches(dry_run=dry_run)
        
        if 'check_stale' in operations:
            stale_items = self.check_stale_issues()
            results['stale_items'] = stale_items
            
            if stale_items and 'label_stale' in operations:
                results['labeled_stale'] = self.add_stale_labels(stale_items, dry_run=dry_run)
        
        if 'update_metadata' in operations:
            results['metadata_updated'] = self.update_repository_metadata()
        
        if 'health_check' in operations:
            results['health_report'] = self.check_repository_health()
        
        return results


def main():
    """Main function for the repository maintenance script."""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--repo-path', default='.', 
                      help='Path to repository')
    parser.add_argument('--github-token',
                      help='GitHub API token (or use GITHUB_TOKEN env var)')
    parser.add_argument('--operations', nargs='+', 
                      choices=['clean_branches', 'check_stale', 'label_stale', 
                              'update_metadata', 'health_check'],
                      default=['clean_branches', 'check_stale', 'health_check'],
                      help='Operations to perform')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    parser.add_argument('--report-file',
                      help='Save maintenance report to file')
    
    args = parser.parse_args()
    
    try:
        maintenance = RepositoryMaintenance(
            repo_path=args.repo_path,
            github_token=args.github_token
        )
        
        results = maintenance.run_maintenance(args.operations, dry_run=args.dry_run)
        
        # Generate report
        report = maintenance.generate_maintenance_report(results)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"📄 Maintenance report saved to {args.report_file}")
        else:
            print("\n" + "="*60)
            print("MAINTENANCE REPORT")
            print("="*60)
            print(report)
        
        print("✅ Repository maintenance completed!")
        
    except KeyboardInterrupt:
        print("\n❌ Maintenance interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()