#!/usr/bin/env python3
"""
Repository Health Monitoring System

Comprehensive health monitoring for the MicroDiff-MatDesign repository including:
- Code quality trends
- Security posture tracking  
- Performance regression detection
- Dependency health monitoring
- SDLC compliance checking
- Automated issue creation for problems
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import requests
    import yaml
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install requests pyyaml")
    sys.exit(1)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    timestamp: str
    remediation: Optional[str] = None


@dataclass
class HealthReport:
    """Complete repository health report."""
    timestamp: str
    overall_status: HealthStatus
    overall_score: float
    checks: List[HealthCheck]
    trends: Dict[str, List[float]]
    recommendations: List[str]
    issues_created: List[str]


class RepositoryHealthMonitor:
    """Repository health monitoring system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".github/health-config.yaml")
        self.project_root = Path.cwd()
        self.config = self._load_config()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/microdiff-matdesign")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load health monitoring configuration."""
        default_config = {
            "thresholds": {
                "test_coverage": 80,
                "code_complexity": 10,
                "security_issues": 0,
                "outdated_dependencies": 5,
                "build_success_rate": 95,
                "documentation_coverage": 70
            },
            "trends": {
                "history_days": 30,
                "min_data_points": 5
            },
            "actions": {
                "create_issues": True,
                "assign_issues": True,
                "add_labels": True
            },
            "checks": {
                "code_quality": True,
                "security": True,
                "testing": True,
                "dependencies": True,
                "documentation": True,
                "performance": True,
                "compliance": True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self._merge_config(default_config, config)
        
        return default_config
    
    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _load_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load historical metrics data."""
        history_file = Path(".github/metrics-history.json")
        if not history_file.exists():
            return {}
        
        try:
            with open(history_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def _save_metrics_history(self, history: Dict[str, List[Dict[str, Any]]]):
        """Save metrics history."""
        history_file = Path(".github/metrics-history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _get_trend_direction(self, values: List[float]) -> str:
        """Determine trend direction from values."""
        if len(values) < 2:
            return "stable"
        
        recent = values[-3:] if len(values) >= 3 else values
        if len(recent) < 2:
            return "stable"
        
        slope = (recent[-1] - recent[0]) / len(recent)
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def check_code_quality(self) -> HealthCheck:
        """Check code quality metrics."""
        try:
            # Load latest metrics
            metrics_file = Path("metrics-report.json")
            if not metrics_file.exists():
                return HealthCheck(
                    name="Code Quality",
                    status=HealthStatus.UNKNOWN,
                    score=0,
                    message="No metrics data available",
                    details={},
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    remediation="Run metrics collection: python scripts/automation/advanced-metrics-collector.py"
                )
            
            with open(metrics_file) as f:
                metrics_data = json.load(f)
            
            # Extract relevant metrics
            complexity_score = 100
            maintainability_score = 100
            duplication_score = 100
            
            for metric in metrics_data.get("metrics", []):
                if metric["name"] == "code.complexity.average":
                    complexity = metric["value"]
                    complexity_score = max(0, 100 - (complexity - 5) * 10)
                elif metric["name"] == "code.maintainability.average":
                    maintainability = metric["value"]
                    maintainability_score = maintainability
                elif metric["name"] == "code.duplication.instances":
                    duplication = metric["value"]
                    duplication_score = max(0, 100 - duplication * 5)
            
            overall_score = (complexity_score + maintainability_score + duplication_score) / 3
            
            if overall_score >= 80:
                status = HealthStatus.HEALTHY
                message = f"Code quality is excellent (score: {overall_score:.1f})"
            elif overall_score >= 60:
                status = HealthStatus.WARNING
                message = f"Code quality needs attention (score: {overall_score:.1f})"
            else:
                status = HealthStatus.CRITICAL
                message = f"Code quality is poor (score: {overall_score:.1f})"
            
            return HealthCheck(
                name="Code Quality",
                status=status,
                score=overall_score,
                message=message,
                details={
                    "complexity_score": complexity_score,
                    "maintainability_score": maintainability_score,
                    "duplication_score": duplication_score
                },
                timestamp=datetime.utcnow().isoformat() + "Z",
                remediation="Refactor complex functions, improve maintainability, reduce duplication"
            )
            
        except Exception as e:
            return HealthCheck(
                name="Code Quality",
                status=HealthStatus.UNKNOWN,
                score=0,
                message=f"Failed to check code quality: {e}",
                details={},
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
    
    def check_security_posture(self) -> HealthCheck:
        """Check security posture."""
        try:
            # Check for security vulnerabilities
            vulnerabilities = {
                "high": 0,
                "medium": 0,
                "low": 0,
                "dependencies": 0
            }
            
            # Load from metrics if available
            metrics_file = Path("metrics-report.json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                
                for metric in metrics_data.get("metrics", []):
                    if metric["name"] == "security.vulnerabilities.high":
                        vulnerabilities["high"] = metric["value"]
                    elif metric["name"] == "security.vulnerabilities.medium":
                        vulnerabilities["medium"] = metric["value"]
                    elif metric["name"] == "security.vulnerabilities.low":
                        vulnerabilities["low"] = metric["value"]
                    elif metric["name"] == "security.dependencies.vulnerabilities":
                        vulnerabilities["dependencies"] = metric["value"]
            
            # Calculate security score
            total_high = vulnerabilities["high"]
            total_medium = vulnerabilities["medium"]
            total_low = vulnerabilities["low"]
            total_deps = vulnerabilities["dependencies"]
            
            # Weighted scoring
            security_penalty = (total_high * 20) + (total_medium * 10) + (total_low * 5) + (total_deps * 15)
            security_score = max(0, 100 - security_penalty)
            
            if total_high > 0 or total_deps > 0:
                status = HealthStatus.CRITICAL
                message = f"Critical security issues found: {total_high} high, {total_deps} dependency vulns"
            elif total_medium > 5:
                status = HealthStatus.WARNING
                message = f"Multiple medium security issues: {total_medium} found"
            elif security_score >= 90:
                status = HealthStatus.HEALTHY
                message = f"Security posture is strong (score: {security_score:.1f})"
            else:
                status = HealthStatus.WARNING
                message = f"Security posture needs improvement (score: {security_score:.1f})"
            
            return HealthCheck(
                name="Security Posture",
                status=status,
                score=security_score,
                message=message,
                details=vulnerabilities,
                timestamp=datetime.utcnow().isoformat() + "Z",
                remediation="Run security scans, update dependencies, fix vulnerabilities"
            )
            
        except Exception as e:
            return HealthCheck(
                name="Security Posture",
                status=HealthStatus.UNKNOWN,
                score=0,
                message=f"Failed to check security: {e}",
                details={},
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
    
    def check_testing_health(self) -> HealthCheck:
        """Check testing health."""
        try:
            coverage = 0
            test_results = {"passed": 0, "failed": 0, "total": 0}
            
            # Load from metrics
            metrics_file = Path("metrics-report.json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                
                for metric in metrics_data.get("metrics", []):
                    if metric["name"] == "tests.coverage.percentage":
                        coverage = metric["value"]
                    elif metric["name"] == "tests.results.passed":
                        test_results["passed"] = metric["value"]
                    elif metric["name"] == "tests.results.failed":
                        test_results["failed"] = metric["value"]
            
            test_results["total"] = test_results["passed"] + test_results["failed"]
            
            # Calculate test health score
            coverage_score = coverage
            success_rate = (test_results["passed"] / max(1, test_results["total"])) * 100
            
            test_score = (coverage_score * 0.6) + (success_rate * 0.4)
            
            threshold = self.config["thresholds"]["test_coverage"]
            
            if test_results["failed"] > 0:
                status = HealthStatus.CRITICAL
                message = f"Tests are failing: {test_results['failed']} failed tests"
            elif coverage < threshold:
                status = HealthStatus.WARNING
                message = f"Test coverage too low: {coverage}% (target: {threshold}%)"
            elif test_score >= 85:
                status = HealthStatus.HEALTHY
                message = f"Testing health is excellent (coverage: {coverage}%, score: {test_score:.1f})"
            else:
                status = HealthStatus.WARNING
                message = f"Testing health needs improvement (score: {test_score:.1f})"
            
            return HealthCheck(
                name="Testing Health",
                status=status,
                score=test_score,
                message=message,
                details={
                    "coverage": coverage,
                    "test_results": test_results,
                    "success_rate": success_rate
                },
                timestamp=datetime.utcnow().isoformat() + "Z",
                remediation="Add more tests, fix failing tests, improve coverage"
            )
            
        except Exception as e:
            return HealthCheck(
                name="Testing Health",
                status=HealthStatus.UNKNOWN,
                score=0,
                message=f"Failed to check testing health: {e}",
                details={},
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
    
    def check_dependency_health(self) -> HealthCheck:
        """Check dependency health."""
        try:
            outdated_count = 0
            total_deps = 0
            
            # Load from metrics
            metrics_file = Path("metrics-report.json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                
                for metric in metrics_data.get("metrics", []):
                    if metric["name"] == "dependencies.python.outdated":
                        outdated_count = metric["value"]
                    elif metric["name"] == "dependencies.python.count":
                        total_deps = metric["value"]
            
            if total_deps == 0:
                return HealthCheck(
                    name="Dependency Health",
                    status=HealthStatus.UNKNOWN,
                    score=0,
                    message="No dependency information available",
                    details={},
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
            
            outdated_percentage = (outdated_count / total_deps) * 100
            dependency_score = max(0, 100 - (outdated_percentage * 2))
            
            threshold = self.config["thresholds"]["outdated_dependencies"]
            
            if outdated_count > threshold * 2:
                status = HealthStatus.CRITICAL
                message = f"Too many outdated dependencies: {outdated_count}/{total_deps}"
            elif outdated_count > threshold:
                status = HealthStatus.WARNING
                message = f"Several outdated dependencies: {outdated_count}/{total_deps}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Dependencies are well maintained: {outdated_count}/{total_deps} outdated"
            
            return HealthCheck(
                name="Dependency Health",
                status=status,
                score=dependency_score,
                message=message,
                details={
                    "total_dependencies": total_deps,
                    "outdated_count": outdated_count,
                    "outdated_percentage": outdated_percentage
                },
                timestamp=datetime.utcnow().isoformat() + "Z",
                remediation="Update outdated dependencies, review compatibility"
            )
            
        except Exception as e:
            return HealthCheck(
                name="Dependency Health",
                status=HealthStatus.UNKNOWN,
                score=0,
                message=f"Failed to check dependencies: {e}",
                details={},
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
    
    def check_documentation_coverage(self) -> HealthCheck:
        """Check documentation coverage."""
        try:
            doc_coverage = 0
            doc_files = 0
            
            # Load from metrics
            metrics_file = Path("metrics-report.json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                
                for metric in metrics_data.get("metrics", []):
                    if metric["name"] == "docs.coverage.percentage":
                        doc_coverage = metric["value"]
                    elif metric["name"] == "docs.files.count":
                        doc_files = metric["value"]
            
            threshold = self.config["thresholds"]["documentation_coverage"]
            
            if doc_coverage < threshold * 0.5:
                status = HealthStatus.CRITICAL
                message = f"Documentation coverage is critically low: {doc_coverage}%"
            elif doc_coverage < threshold:
                status = HealthStatus.WARNING
                message = f"Documentation coverage below target: {doc_coverage}% (target: {threshold}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Documentation coverage is good: {doc_coverage}%"
            
            return HealthCheck(
                name="Documentation Coverage",
                status=status,
                score=doc_coverage,
                message=message,
                details={
                    "coverage_percentage": doc_coverage,
                    "doc_files": doc_files
                },
                timestamp=datetime.utcnow().isoformat() + "Z",
                remediation="Add docstrings, update documentation, improve API docs"
            )
            
        except Exception as e:
            return HealthCheck(
                name="Documentation Coverage",
                status=HealthStatus.UNKNOWN,
                score=0,
                message=f"Failed to check documentation: {e}",
                details={},
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
    
    def create_github_issue(self, check: HealthCheck) -> Optional[str]:
        """Create GitHub issue for critical health problems."""
        if not self.config["actions"]["create_issues"] or not self.github_token:
            return None
        
        if check.status not in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            return None
        
        # Check if similar issue already exists
        issue_title = f"🏥 Repository Health Alert: {check.name}"
        
        try:
            # Search for existing issues
            search_url = f"https://api.github.com/search/issues"
            search_params = {
                "q": f'repo:{self.github_repo} is:issue is:open "Repository Health Alert: {check.name}"'
            }
            
            headers = {"Authorization": f"token {self.github_token}"}
            response = requests.get(search_url, params=search_params, headers=headers)
            
            if response.json().get("total_count", 0) > 0:
                return None  # Issue already exists
            
            # Create new issue
            issue_body = f"""## Repository Health Alert

**Status:** {check.status.value.upper()}
**Score:** {check.score:.1f}/100
**Timestamp:** {check.timestamp}

### Problem Description
{check.message}

### Details
```json
{json.dumps(check.details, indent=2)}
```

### Recommended Actions
{check.remediation or 'See health monitoring documentation for guidance.'}

### Automation
This issue was automatically created by the repository health monitoring system.

---
*🤖 Generated by Repository Health Monitor*
"""
            
            issue_data = {
                "title": issue_title,
                "body": issue_body,
                "labels": ["health-monitor", "automated", check.status.value]
            }
            
            if self.config["actions"]["add_labels"]:
                if check.name.lower().replace(" ", "-") not in issue_data["labels"]:
                    issue_data["labels"].append(check.name.lower().replace(" ", "-"))
            
            create_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            response = requests.post(create_url, json=issue_data, headers=headers)
            
            if response.status_code == 201:
                return response.json()["html_url"]
            else:
                print(f"Failed to create issue: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            print(f"Error creating GitHub issue: {e}")
            return None
    
    def run_health_checks(self) -> HealthReport:
        """Run all health checks and generate report."""
        print("📈 Running repository health checks...")
        
        checks = []
        issues_created = []
        
        # Run individual checks
        if self.config["checks"]["code_quality"]:
            check = self.check_code_quality()
            checks.append(check)
            
        if self.config["checks"]["security"]:
            check = self.check_security_posture()
            checks.append(check)
            
        if self.config["checks"]["testing"]:
            check = self.check_testing_health()
            checks.append(check)
            
        if self.config["checks"]["dependencies"]:
            check = self.check_dependency_health()
            checks.append(check)
            
        if self.config["checks"]["documentation"]:
            check = self.check_documentation_coverage()
            checks.append(check)
        
        # Calculate overall health
        valid_scores = [check.score for check in checks if check.status != HealthStatus.UNKNOWN]
        
        if not valid_scores:
            overall_score = 0
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_score = sum(valid_scores) / len(valid_scores)
            
            critical_count = sum(1 for check in checks if check.status == HealthStatus.CRITICAL)
            warning_count = sum(1 for check in checks if check.status == HealthStatus.WARNING)
            
            if critical_count > 0:
                overall_status = HealthStatus.CRITICAL
            elif warning_count > 0:
                overall_status = HealthStatus.WARNING
            elif overall_score >= 80:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.WARNING
        
        # Create issues for problems
        for check in checks:
            if check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                issue_url = self.create_github_issue(check)
                if issue_url:
                    issues_created.append(issue_url)
        
        # Generate recommendations
        recommendations = []
        if overall_score < 60:
            recommendations.append("Repository health is concerning - immediate action required")
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            recommendations.append("Address critical issues as highest priority")
        if len([c for c in checks if c.status == HealthStatus.WARNING]) > 2:
            recommendations.append("Multiple areas need attention - create improvement plan")
        
        return HealthReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_status=overall_status,
            overall_score=overall_score,
            checks=checks,
            trends={},  # Will be populated with historical data
            recommendations=recommendations,
            issues_created=issues_created
        )
    
    def save_report(self, report: HealthReport) -> Path:
        """Save health report to file."""
        report_file = Path(".github/health-report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        return report_file
    
    def print_report(self, report: HealthReport):
        """Print health report to console."""
        print("\n" + "="*60)
        print("📈 REPOSITORY HEALTH REPORT")
        print("="*60)
        print(f"Overall Status: {report.overall_status.value.upper()}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Timestamp: {report.timestamp}")
        
        print("\n🗒  Health Checks:")
        for check in report.checks:
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.CRITICAL: "🚨",
                HealthStatus.UNKNOWN: "❓"
            }[check.status]
            
            print(f"  {status_emoji} {check.name}: {check.score:.1f}/100")
            print(f"     {check.message}")
        
        if report.recommendations:
            print("\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        if report.issues_created:
            print("\n🗋 Issues Created:")
            for issue_url in report.issues_created:
                print(f"  - {issue_url}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository health monitoring")
    parser.add_argument("--config", type=Path, help="Configuration file")
    parser.add_argument("--no-issues", action="store_true", help="Don't create GitHub issues")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    monitor = RepositoryHealthMonitor(config_path=args.config)
    
    if args.no_issues:
        monitor.config["actions"]["create_issues"] = False
    
    try:
        report = monitor.run_health_checks()
        report_file = monitor.save_report(report)
        
        if not args.quiet:
            monitor.print_report(report)
        
        print(f"\n📄 Health report saved to: {report_file}")
        
        # Exit with appropriate code
        if report.overall_status == HealthStatus.CRITICAL:
            sys.exit(2)
        elif report.overall_status == HealthStatus.WARNING:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Health check interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
