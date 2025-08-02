#!/usr/bin/env python3
"""
Automated metrics collection script for MicroDiff-MatDesign.

This script collects comprehensive metrics from various sources including
GitHub API, code analysis tools, CI/CD systems, and application monitoring.
"""

import json
import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import requests
import yaml


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    github_token: Optional[str] = None
    github_repo: str = "danieleschmidt/microdiff-matdesign"
    output_format: str = "json"
    output_file: Optional[str] = None
    include_git_stats: bool = True
    include_code_quality: bool = True
    include_dependencies: bool = True
    verbose: bool = False


class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.repo_path = Path.cwd()
        self.metrics = self._load_base_metrics()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG if self.config.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_base_metrics(self) -> Dict[str, Any]:
        """Load base metrics structure."""
        metrics_file = self.repo_path / ".github" / "project-metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        else:
            self.logger.warning("Base metrics file not found, creating empty structure")
            return {"metrics": {}, "targets": {}, "tracking": {}}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        self.logger.info("Starting comprehensive metrics collection")
        
        if self.config.include_git_stats:
            self.collect_git_metrics()
            
        if self.config.github_token:
            self.collect_github_metrics()
            
        if self.config.include_code_quality:
            self.collect_code_quality_metrics()
            
        if self.config.include_dependencies:
            self.collect_dependency_metrics()
            
        self.collect_file_metrics()
        self.collect_test_metrics()
        
        # Update tracking information
        self.metrics["tracking"]["last_collection"] = datetime.now(timezone.utc).isoformat()
        
        self.logger.info("Metrics collection completed")
        return self.metrics
    
    def collect_git_metrics(self) -> None:
        """Collect Git repository metrics."""
        self.logger.info("Collecting Git metrics")
        
        try:
            # Total commits
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            total_commits = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Commits in last month
            result = subprocess.run(
                ["git", "rev-list", "--count", "--since=1.month", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            monthly_commits = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            contributors = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            contributors.append({
                                "name": parts[1],
                                "commits": int(parts[0])
                            })
            
            # Branches
            result = subprocess.run(
                ["git", "branch", "-r"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            total_branches = len(result.stdout.strip().split('\n')) if result.returncode == 0 else 0
            
            # Update metrics
            self.metrics.setdefault("metrics", {}).setdefault("development", {}).update({
                "commits": {
                    "total": total_commits,
                    "last_month": monthly_commits,
                    "avg_per_week": monthly_commits / 4 if monthly_commits > 0 else 0,
                    "contributors": contributors
                },
                "branches": {
                    "total": total_branches,
                    "active": len([c for c in contributors if c["commits"] > 0]),
                    "stale": max(0, total_branches - len(contributors))
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting Git metrics: {e}")
    
    def collect_github_metrics(self) -> None:
        """Collect GitHub API metrics."""
        self.logger.info("Collecting GitHub metrics")
        
        if not self.config.github_token:
            self.logger.warning("GitHub token not provided, skipping GitHub metrics")
            return
            
        headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Repository information
            repo_url = f"https://api.github.com/repos/{self.config.github_repo}"
            response = requests.get(repo_url, headers=headers)
            
            if response.status_code == 200:
                repo_data = response.json()
                
                # Basic repository metrics
                self.metrics.setdefault("metrics", {}).setdefault("community", {}).setdefault("github", {}).update({
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0)
                })
                
                # Pull requests
                pr_url = f"{repo_url}/pulls?state=all&per_page=100"
                pr_response = requests.get(pr_url, headers=headers)
                
                if pr_response.status_code == 200:
                    prs = pr_response.json()
                    open_prs = len([pr for pr in prs if pr["state"] == "open"])
                    merged_prs = len([pr for pr in prs if pr["merged_at"] is not None])
                    closed_prs = len([pr for pr in prs if pr["state"] == "closed" and pr["merged_at"] is None])
                    
                    self.metrics["metrics"]["development"]["pull_requests"] = {
                        "total": len(prs),
                        "open": open_prs,
                        "merged": merged_prs,
                        "closed": closed_prs,
                        "avg_time_to_merge_hours": self._calculate_avg_merge_time(prs)
                    }
                
                # Issues
                issues_url = f"{repo_url}/issues?state=all&per_page=100"
                issues_response = requests.get(issues_url, headers=headers)
                
                if issues_response.status_code == 200:
                    issues = issues_response.json()
                    # Filter out pull requests (they appear in issues endpoint)
                    actual_issues = [issue for issue in issues if "pull_request" not in issue]
                    
                    open_issues = len([issue for issue in actual_issues if issue["state"] == "open"])
                    closed_issues = len([issue for issue in actual_issues if issue["state"] == "closed"])
                    
                    # Label counts
                    label_counts = {}
                    for issue in actual_issues:
                        for label in issue.get("labels", []):
                            label_name = label["name"]
                            label_counts[label_name] = label_counts.get(label_name, 0) + 1
                    
                    self.metrics["metrics"]["development"]["issues"] = {
                        "total": len(actual_issues),
                        "open": open_issues,
                        "closed": closed_issues,
                        "avg_time_to_close_hours": self._calculate_avg_close_time(actual_issues),
                        "labels": label_counts
                    }
                
                # Releases
                releases_url = f"{repo_url}/releases"
                releases_response = requests.get(releases_url, headers=headers)
                
                if releases_response.status_code == 200:
                    releases = releases_response.json()
                    
                    self.metrics["metrics"]["development"]["releases"] = {
                        "total": len(releases),
                        "latest": releases[0]["tag_name"] if releases else None,
                        "avg_time_between_releases_days": self._calculate_avg_release_interval(releases)
                    }
                    
            else:
                self.logger.error(f"GitHub API error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error collecting GitHub metrics: {e}")
    
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics."""
        self.logger.info("Collecting code quality metrics")
        
        try:
            # Lines of code
            loc_data = self._count_lines_of_code()
            
            # Test coverage (if pytest-cov is available)
            coverage_data = self._get_test_coverage()
            
            # Update metrics
            self.metrics.setdefault("metrics", {}).setdefault("code_quality", {}).update({
                "lines_of_code": loc_data,
                "test_coverage": coverage_data
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting code quality metrics: {e}")
    
    def collect_dependency_metrics(self) -> None:
        """Collect dependency metrics."""
        self.logger.info("Collecting dependency metrics")
        
        try:
            # Python dependencies from requirements.txt
            deps = self._analyze_python_dependencies()
            
            self.metrics.setdefault("metrics", {}).setdefault("code_quality", {})["dependencies"] = deps
            
        except Exception as e:
            self.logger.error(f"Error collecting dependency metrics: {e}")
    
    def collect_file_metrics(self) -> None:
        """Collect file and structure metrics."""
        self.logger.info("Collecting file metrics")
        
        try:
            # Count different file types
            file_counts = {
                "python": 0,
                "yaml": 0,
                "markdown": 0,
                "dockerfile": 0,
                "json": 0
            }
            
            for file_path in self.repo_path.rglob("*"):
                if file_path.is_file() and not self._is_ignored_path(file_path):
                    suffix = file_path.suffix.lower()
                    if suffix == ".py":
                        file_counts["python"] += 1
                    elif suffix in [".yml", ".yaml"]:
                        file_counts["yaml"] += 1
                    elif suffix == ".md":
                        file_counts["markdown"] += 1
                    elif file_path.name.lower() in ["dockerfile", "dockerfile.dev"]:
                        file_counts["dockerfile"] += 1
                    elif suffix == ".json":
                        file_counts["json"] += 1
            
            # Documentation metrics
            docs_dir = self.repo_path / "docs"
            doc_pages = 0
            doc_word_count = 0
            
            if docs_dir.exists():
                for md_file in docs_dir.rglob("*.md"):
                    doc_pages += 1
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc_word_count += len(content.split())
                    except Exception:
                        pass
            
            # Update metrics
            self.metrics.setdefault("metrics", {}).setdefault("community", {}).setdefault("documentation", {}).update({
                "pages": doc_pages,
                "word_count": doc_word_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "completeness_score": min(100, (doc_pages / 10) * 100)  # Arbitrary scoring
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting file metrics: {e}")
    
    def collect_test_metrics(self) -> None:
        """Collect test-related metrics."""
        self.logger.info("Collecting test metrics")
        
        try:
            # Count test files
            test_files = list(self.repo_path.rglob("test_*.py")) + list(self.repo_path.rglob("*_test.py"))
            test_count = len(test_files)
            
            # Try to run tests and get metrics
            if (self.repo_path / "pytest.ini").exists() or (self.repo_path / "pyproject.toml").exists():
                try:
                    result = subprocess.run(
                        ["python", "-m", "pytest", "--collect-only", "-q"],
                        capture_output=True, text=True, cwd=self.repo_path, timeout=30
                    )
                    if result.returncode == 0:
                        # Parse pytest collection output
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if "collected" in line and "item" in line:
                                # Extract number from "collected X items"
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "collected" and i + 1 < len(parts):
                                        try:
                                            test_count = int(parts[i + 1])
                                            break
                                        except ValueError:
                                            pass
                except Exception:
                    pass
            
            # Update metrics
            self.metrics.setdefault("metrics", {}).setdefault("ci_cd", {}).setdefault("tests", {}).update({
                "total_tests": test_count,
                "test_files": len(test_files)
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting test metrics: {e}")
    
    # Helper methods
    
    def _calculate_avg_merge_time(self, prs: List[Dict]) -> float:
        """Calculate average time to merge PRs in hours."""
        merge_times = []
        for pr in prs:
            if pr.get("merged_at") and pr.get("created_at"):
                try:
                    created = datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))
                    merged = datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00'))
                    hours = (merged - created).total_seconds() / 3600
                    merge_times.append(hours)
                except Exception:
                    continue
        
        return sum(merge_times) / len(merge_times) if merge_times else 0
    
    def _calculate_avg_close_time(self, issues: List[Dict]) -> float:
        """Calculate average time to close issues in hours."""
        close_times = []
        for issue in issues:
            if issue.get("closed_at") and issue.get("created_at"):
                try:
                    created = datetime.fromisoformat(issue["created_at"].replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(issue["closed_at"].replace('Z', '+00:00'))
                    hours = (closed - created).total_seconds() / 3600
                    close_times.append(hours)
                except Exception:
                    continue
        
        return sum(close_times) / len(close_times) if close_times else 0
    
    def _calculate_avg_release_interval(self, releases: List[Dict]) -> float:
        """Calculate average time between releases in days."""
        if len(releases) < 2:
            return 0
            
        intervals = []
        for i in range(len(releases) - 1):
            try:
                current = datetime.fromisoformat(releases[i]["created_at"].replace('Z', '+00:00'))
                previous = datetime.fromisoformat(releases[i + 1]["created_at"].replace('Z', '+00:00'))
                days = (current - previous).days
                intervals.append(days)
            except Exception:
                continue
        
        return sum(intervals) / len(intervals) if intervals else 0
    
    def _count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code by file type."""
        loc_data = {
            "total": 0,
            "python": 0,
            "yaml": 0,
            "markdown": 0,
            "dockerfile": 0
        }
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._is_ignored_path(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        loc_data["total"] += lines
                        
                        suffix = file_path.suffix.lower()
                        if suffix == ".py":
                            loc_data["python"] += lines
                        elif suffix in [".yml", ".yaml"]:
                            loc_data["yaml"] += lines
                        elif suffix == ".md":
                            loc_data["markdown"] += lines
                        elif file_path.name.lower() in ["dockerfile", "dockerfile.dev"]:
                            loc_data["dockerfile"] += lines
                            
                except Exception:
                    continue
        
        return loc_data
    
    def _get_test_coverage(self) -> Dict[str, Any]:
        """Get test coverage information."""
        coverage_data = {
            "percentage": 0,
            "lines_covered": 0,
            "lines_total": 0,
            "missing_coverage": []
        }
        
        # Try to run coverage if available
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=microdiff_matdesign", "--cov-report=json", "--cov-report=term"],
                capture_output=True, text=True, cwd=self.repo_path, timeout=120
            )
            
            # Look for coverage.json file
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_json = json.load(f)
                    
                coverage_data.update({
                    "percentage": coverage_json.get("totals", {}).get("percent_covered", 0),
                    "lines_covered": coverage_json.get("totals", {}).get("covered_lines", 0),
                    "lines_total": coverage_json.get("totals", {}).get("num_statements", 0)
                })
                
        except Exception as e:
            self.logger.debug(f"Could not get coverage data: {e}")
        
        return coverage_data
    
    def _analyze_python_dependencies(self) -> Dict[str, Any]:
        """Analyze Python dependencies."""
        deps_data = {
            "total": 0,
            "outdated": 0,
            "security_issues": 0,
            "licenses": {}
        }
        
        # Check requirements.txt
        req_file = self.repo_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    deps_data["total"] = len(deps)
            except Exception:
                pass
        
        # Check pyproject.toml
        pyproject_file = self.repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                    # Simple parsing for dependencies count
                    if "dependencies" in content:
                        deps_data["total"] = content.count('\n') // 10  # Rough estimate
            except Exception:
                pass
        
        return deps_data
    
    def _is_ignored_path(self, path: Path) -> bool:
        """Check if path should be ignored in metrics."""
        ignored_dirs = {".git", ".pytest_cache", "__pycache__", ".tox", "node_modules", ".venv", "venv"}
        ignored_files = {".gitignore", ".DS_Store"}
        
        # Check if any parent directory is ignored
        for parent in path.parents:
            if parent.name in ignored_dirs:
                return True
        
        # Check if file itself is ignored
        if path.name in ignored_files:
            return True
        
        return False
    
    def save_metrics(self, output_file: Optional[str] = None) -> None:
        """Save metrics to file."""
        output_file = output_file or self.config.output_file
        
        if not output_file:
            # Default output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"metrics_{timestamp}.json"
        
        output_path = Path(output_file)
        
        if self.config.output_format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        elif self.config.output_format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(self.metrics, f, default_flow_style=False)
        
        self.logger.info(f"Metrics saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        dev_metrics = self.metrics.get("metrics", {}).get("development", {})
        quality_metrics = self.metrics.get("metrics", {}).get("code_quality", {})
        community_metrics = self.metrics.get("metrics", {}).get("community", {})
        
        print("\n=== METRICS SUMMARY ===")
        
        if dev_metrics:
            print(f"\nDevelopment:")
            print(f"  Total commits: {dev_metrics.get('commits', {}).get('total', 'N/A')}")
            print(f"  Monthly commits: {dev_metrics.get('commits', {}).get('last_month', 'N/A')}")
            print(f"  Contributors: {len(dev_metrics.get('commits', {}).get('contributors', []))}")
            print(f"  Total branches: {dev_metrics.get('branches', {}).get('total', 'N/A')}")
        
        if quality_metrics:
            print(f"\nCode Quality:")
            loc = quality_metrics.get('lines_of_code', {})
            print(f"  Lines of code: {loc.get('total', 'N/A')}")
            print(f"  Python LOC: {loc.get('python', 'N/A')}")
            coverage = quality_metrics.get('test_coverage', {})
            print(f"  Test coverage: {coverage.get('percentage', 'N/A')}%")
        
        if community_metrics:
            github = community_metrics.get('github', {})
            docs = community_metrics.get('documentation', {})
            print(f"\nCommunity:")
            print(f"  GitHub stars: {github.get('stars', 'N/A')}")
            print(f"  Forks: {github.get('forks', 'N/A')}")
            print(f"  Documentation pages: {docs.get('pages', 'N/A')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-git", action="store_true", help="Skip Git metrics")
    parser.add_argument("--no-quality", action="store_true", help="Skip code quality metrics")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency metrics")
    parser.add_argument("--summary", action="store_true", help="Print summary to console")
    
    args = parser.parse_args()
    
    config = MetricsConfig(
        github_token=args.github_token or os.getenv("GITHUB_TOKEN"),
        output_format=args.format,
        output_file=args.output,
        include_git_stats=not args.no_git,
        include_code_quality=not args.no_quality,
        include_dependencies=not args.no_deps,
        verbose=args.verbose
    )
    
    collector = MetricsCollector(config)
    collector.collect_all_metrics()
    
    if args.output or config.output_file:
        collector.save_metrics()
    
    if args.summary:
        collector.print_summary()
    
    if not args.output and not config.output_file and not args.summary:
        # Default: print JSON to stdout
        print(json.dumps(collector.metrics, indent=2))


if __name__ == "__main__":
    main()