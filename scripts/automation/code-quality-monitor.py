#!/usr/bin/env python3
"""
Code quality monitoring script for MicroDiff-MatDesign.

This script monitors code quality metrics, runs static analysis tools,
and generates reports on code health trends over time.
"""

import json
import os
import sys
import subprocess
import logging
import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import requests


@dataclass
class QualityMetrics:
    """Code quality metrics data structure."""
    timestamp: str
    commit_hash: str
    lines_of_code: int
    test_coverage: float
    cyclomatic_complexity: float
    maintainability_index: float
    code_duplication: float
    security_issues: int
    lint_issues: int
    type_errors: int
    documentation_coverage: float


class CodeQualityMonitor:
    """Code quality monitoring and analysis system."""
    
    def __init__(self, project_root: Path, config: Dict[str, Any]):
        self.project_root = project_root
        self.config = config
        self.logger = self._setup_logging()
        self.metrics_history = []
        self._load_metrics_history()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG if self.config.get('verbose', False) else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_metrics_history(self) -> None:
        """Load historical metrics data."""
        history_file = self.project_root / ".code-quality-history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.metrics_history = json.load(f)
                self.logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            except Exception as e:
                self.logger.error(f"Failed to load metrics history: {e}")
                self.metrics_history = []
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to file."""
        history_file = self.project_root / ".code-quality-history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            self.logger.info("Metrics history saved")
        except Exception as e:
            self.logger.error(f"Failed to save metrics history: {e}")
    
    def collect_current_metrics(self) -> QualityMetrics:
        """Collect current code quality metrics."""
        self.logger.info("Collecting current code quality metrics")
        
        commit_hash = self._get_current_commit_hash()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Collect various metrics
        loc = self._count_lines_of_code()
        coverage = self._get_test_coverage()
        complexity = self._analyze_complexity()
        duplication = self._check_code_duplication()
        security = self._run_security_analysis()
        lint = self._run_lint_analysis()
        types = self._run_type_analysis()
        docs = self._check_documentation_coverage()
        
        metrics = QualityMetrics(
            timestamp=timestamp,
            commit_hash=commit_hash,
            lines_of_code=loc,
            test_coverage=coverage,
            cyclomatic_complexity=complexity[0],
            maintainability_index=complexity[1],
            code_duplication=duplication,
            security_issues=security,
            lint_issues=lint,
            type_errors=types,
            documentation_coverage=docs
        )
        
        self.logger.info("Metrics collection completed")
        return metrics
    
    def _get_current_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _count_lines_of_code(self) -> int:
        """Count total lines of code."""
        total_lines = 0
        
        for file_path in self.project_root.rglob("*.py"):
            if self._should_analyze_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Count non-empty, non-comment lines
                        code_lines = [
                            line.strip() for line in lines 
                            if line.strip() and not line.strip().startswith('#')
                        ]
                        total_lines += len(code_lines)
                except Exception:
                    continue
        
        return total_lines
    
    def _get_test_coverage(self) -> float:
        """Get test coverage percentage."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "python", "-m", "pytest", 
                "--cov=microdiff_matdesign", 
                "--cov-report=json",
                "--quiet"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)
            
            # Read coverage.json
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                return coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception as e:
            self.logger.warning(f"Could not get test coverage: {e}")
        
        return 0.0
    
    def _analyze_complexity(self) -> Tuple[float, float]:
        """Analyze code complexity and maintainability."""
        try:
            # Try to use radon for complexity analysis
            result = subprocess.run([
                "radon", "cc", str(self.project_root / "microdiff_matdesign"),
                "--json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                
                # Calculate average complexity
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get("type") == "function":
                            total_complexity += item.get("complexity", 0)
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                
                # Get maintainability index
                mi_result = subprocess.run([
                    "radon", "mi", str(self.project_root / "microdiff_matdesign"),
                    "--json"
                ], capture_output=True, text=True)
                
                maintainability = 0.0
                if mi_result.returncode == 0:
                    mi_data = json.loads(mi_result.stdout)
                    mi_scores = [data.get("mi", 0) for data in mi_data.values()]
                    maintainability = sum(mi_scores) / len(mi_scores) if mi_scores else 0
                
                return avg_complexity, maintainability
                
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not analyze complexity: {e}")
        
        return 0.0, 0.0
    
    def _check_code_duplication(self) -> float:
        """Check for code duplication."""
        try:
            # Simple duplication check using file comparison
            python_files = list(self.project_root.rglob("*.py"))
            python_files = [f for f in python_files if self._should_analyze_file(f)]
            
            total_lines = 0
            duplicate_lines = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Simple duplicate detection (identical lines)
                        line_counts = {}
                        for line in lines:
                            stripped = line.strip()
                            if stripped and not stripped.startswith('#'):
                                line_counts[stripped] = line_counts.get(stripped, 0) + 1
                        
                        for count in line_counts.values():
                            if count > 1:
                                duplicate_lines += count - 1
                                
                except Exception:
                    continue
            
            return (duplicate_lines / total_lines * 100) if total_lines > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Could not check code duplication: {e}")
        
        return 0.0
    
    def _run_security_analysis(self) -> int:
        """Run security analysis and count issues."""
        issues = 0
        
        try:
            # Run bandit security analysis
            result = subprocess.run([
                "bandit", "-r", str(self.project_root / "microdiff_matdesign"),
                "-f", "json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                issues += len(bandit_data.get("results", []))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not run bandit: {e}")
        
        try:
            # Run safety check
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                issues += len(safety_data)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not run safety check: {e}")
        
        return issues
    
    def _run_lint_analysis(self) -> int:
        """Run linting and count issues."""
        try:
            # Run flake8
            result = subprocess.run([
                "flake8", str(self.project_root / "microdiff_matdesign"),
                "--format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    lint_data = json.loads(result.stdout)
                    return len(lint_data)
                except json.JSONDecodeError:
                    # Count lines of output as issues
                    return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
        except FileNotFoundError:
            self.logger.warning("flake8 not available")
        
        return 0
    
    def _run_type_analysis(self) -> int:
        """Run type checking and count errors."""
        try:
            # Run mypy
            result = subprocess.run([
                "mypy", str(self.project_root / "microdiff_matdesign"),
                "--show-error-codes"
            ], capture_output=True, text=True)
            
            if result.stdout:
                # Count error lines
                error_lines = [
                    line for line in result.stdout.split('\n') 
                    if 'error:' in line.lower()
                ]
                return len(error_lines)
            
        except FileNotFoundError:
            self.logger.warning("mypy not available")
        
        return 0
    
    def _check_documentation_coverage(self) -> float:
        """Check documentation coverage."""
        try:
            python_files = list(self.project_root.rglob("*.py"))
            python_files = [f for f in python_files if self._should_analyze_file(f)]
            
            total_functions = 0
            documented_functions = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Simple docstring detection
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                            total_functions += 1
                            
                            # Check if next few lines contain docstring
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    documented_functions += 1
                                    break
                                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                                    break
                                    
                except Exception:
                    continue
            
            return (documented_functions / total_functions * 100) if total_functions > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Could not check documentation coverage: {e}")
        
        return 0.0
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be included in analysis."""
        excluded_dirs = {'.git', '__pycache__', '.pytest_cache', '.tox', 'venv', '.venv'}
        excluded_files = {'__init__.py'}
        
        # Check if any parent directory is excluded
        for parent in file_path.parents:
            if parent.name in excluded_dirs:
                return False
        
        # Check if file is excluded
        if file_path.name in excluded_files:
            return False
        
        # Must be under the main package
        try:
            file_path.relative_to(self.project_root / "microdiff_matdesign")
            return True
        except ValueError:
            # Also include test files
            try:
                file_path.relative_to(self.project_root / "tests")
                return True
            except ValueError:
                return False
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if len(self.metrics_history) < 2:
            return {"error": "Not enough historical data for trend analysis"}
        
        # Get recent metrics
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        
        trends = {}
        
        # Calculate trends for each metric
        for metric_name in ["test_coverage", "cyclomatic_complexity", "security_issues", "lint_issues"]:
            values = [m.get(metric_name, 0) for m in recent_metrics]
            if len(values) >= 2:
                trend = "improving" if values[-1] < values[0] else "declining" if values[-1] > values[0] else "stable"
                change = values[-1] - values[0]
                trends[metric_name] = {
                    "trend": trend,
                    "change": change,
                    "current": values[-1],
                    "previous": values[0]
                }
        
        return trends
    
    def generate_report(self, current_metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        trends = self.analyze_trends()
        
        # Quality score calculation
        quality_score = self._calculate_quality_score(current_metrics)
        
        # Recommendations
        recommendations = self._generate_recommendations(current_metrics)
        
        report = {
            "timestamp": current_metrics.timestamp,
            "commit_hash": current_metrics.commit_hash,
            "quality_score": quality_score,
            "current_metrics": asdict(current_metrics),
            "trends": trends,
            "recommendations": recommendations,
            "summary": {
                "lines_of_code": current_metrics.lines_of_code,
                "test_coverage": f"{current_metrics.test_coverage:.1f}%",
                "complexity": f"{current_metrics.cyclomatic_complexity:.1f}",
                "security_issues": current_metrics.security_issues,
                "overall_health": self._get_health_status(quality_score)
            }
        }
        
        return report
    
    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score (0-100)."""
        # Weighted scoring
        coverage_score = min(metrics.test_coverage, 100)
        complexity_score = max(0, 100 - (metrics.cyclomatic_complexity - 1) * 10)
        security_score = max(0, 100 - metrics.security_issues * 20)
        lint_score = max(0, 100 - metrics.lint_issues * 2)
        type_score = max(0, 100 - metrics.type_errors * 5)
        docs_score = min(metrics.documentation_coverage, 100)
        
        # Weighted average
        total_score = (
            coverage_score * 0.25 +
            complexity_score * 0.15 +
            security_score * 0.25 +
            lint_score * 0.15 +
            type_score * 0.10 +
            docs_score * 0.10
        )
        
        return round(total_score, 1)
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on quality score."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics.test_coverage < 80:
            recommendations.append(f"Increase test coverage from {metrics.test_coverage:.1f}% to at least 80%")
        
        if metrics.cyclomatic_complexity > 10:
            recommendations.append(f"Reduce cyclomatic complexity from {metrics.cyclomatic_complexity:.1f} to below 10")
        
        if metrics.security_issues > 0:
            recommendations.append(f"Fix {metrics.security_issues} security issues")
        
        if metrics.lint_issues > 20:
            recommendations.append(f"Address {metrics.lint_issues} linting issues")
        
        if metrics.type_errors > 0:
            recommendations.append(f"Fix {metrics.type_errors} type checking errors")
        
        if metrics.documentation_coverage < 70:
            recommendations.append(f"Improve documentation coverage from {metrics.documentation_coverage:.1f}% to at least 70%")
        
        if metrics.code_duplication > 5:
            recommendations.append(f"Reduce code duplication from {metrics.code_duplication:.1f}% to below 5%")
        
        if not recommendations:
            recommendations.append("Code quality is excellent! Keep up the good work.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """Save quality report to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"quality_report_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Quality report saved to {output_path}")
    
    def update_metrics_history(self, metrics: QualityMetrics) -> None:
        """Update metrics history with new data."""
        self.metrics_history.append(asdict(metrics))
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        self._save_metrics_history()
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print quality report summary."""
        print("\n=== CODE QUALITY REPORT ===")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Commit: {report['commit_hash'][:8]}")
        print(f"Quality Score: {report['quality_score']}/100 ({report['summary']['overall_health'].upper()})")
        
        print(f"\n=== METRICS ===")
        summary = report['summary']
        print(f"Lines of Code: {summary['lines_of_code']:,}")
        print(f"Test Coverage: {summary['test_coverage']}")
        print(f"Complexity: {summary['complexity']}")
        print(f"Security Issues: {summary['security_issues']}")
        
        if report['recommendations']:
            print(f"\n=== RECOMMENDATIONS ===")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        if report['trends'] and 'error' not in report['trends']:
            print(f"\n=== TRENDS ===")
            for metric, trend_data in report['trends'].items():
                trend_icon = "📈" if trend_data['trend'] == 'improving' else "📉" if trend_data['trend'] == 'declining' else "➡️"
                print(f"{metric}: {trend_icon} {trend_data['trend']} (Δ {trend_data['change']:+.1f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor code quality metrics")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-history", action="store_true", help="Don't update metrics history")
    
    args = parser.parse_args()
    
    config = {
        "verbose": args.verbose,
        "output_format": args.format,
        "update_history": not args.no_history
    }
    
    project_root = Path.cwd()
    monitor = CodeQualityMonitor(project_root, config)
    
    # Collect current metrics
    current_metrics = monitor.collect_current_metrics()
    
    # Generate report
    report = monitor.generate_report(current_metrics)
    
    # Update history
    if config["update_history"]:
        monitor.update_metrics_history(current_metrics)
    
    # Output results
    if args.output:
        monitor.save_report(report, args.output)
    
    if args.format == "summary":
        monitor.print_summary(report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()