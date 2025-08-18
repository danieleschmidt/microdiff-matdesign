#!/usr/bin/env python3
"""
Advanced Metrics Collection System for MicroDiff-MatDesign

Comprehensive metrics collection including:
- Code quality and complexity metrics
- Security vulnerability tracking
- Performance benchmarks
- Test coverage and success rates
- Documentation coverage
- Dependency health
- Container and deployment metrics
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    import psutil
    import requests
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install aiohttp psutil requests")
    sys.exit(1)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    unit: str
    timestamp: str
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class MetricsReport:
    """Complete metrics report."""
    timestamp: str
    collection_duration_ms: int
    metrics: List[MetricPoint]
    summary: Dict[str, Any]
    errors: List[str]


class AdvancedMetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".github/metrics-config.json")
        self.project_root = Path.cwd()
        self.start_time = time.time()
        self.metrics: List[MetricPoint] = []
        self.errors: List[str] = []
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics collection configuration."""
        default_config = {
            "git_metrics": True,
            "code_quality_metrics": True,
            "test_metrics": True,
            "security_metrics": True,
            "performance_metrics": True,
            "dependency_metrics": True,
            "container_metrics": True,
            "documentation_metrics": True,
            "github_api_token": os.getenv("GITHUB_TOKEN"),
            "output_format": "json",
            "output_file": "metrics-report.json"
        }
        
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Command failed: {' '.join(cmd)}: {e.stderr}")
            return None
        except FileNotFoundError:
            self.errors.append(f"Command not found: {cmd[0]}")
            return None
    
    def _add_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None, **metadata):
        """Add a metric point."""
        self.metrics.append(MetricPoint(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow().isoformat() + "Z",
            tags=tags or {},
            metadata=metadata
        ))
    
    async def collect_git_metrics(self):
        """Collect Git repository metrics."""
        print("📊 Collecting Git metrics...")
        
        try:
            # Commit metrics
            commits_30d = self._run_command(["git", "rev-list", "--count", "--since=30.days.ago", "HEAD"])
            if commits_30d:
                self._add_metric("git.commits.30_days", int(commits_30d), "count")
            
            commits_7d = self._run_command(["git", "rev-list", "--count", "--since=7.days.ago", "HEAD"])
            if commits_7d:
                self._add_metric("git.commits.7_days", int(commits_7d), "count")
            
            # Contributors
            contributors = self._run_command(["git", "shortlog", "-sn", "--since=30.days.ago"])
            if contributors:
                contributor_count = len(contributors.split("\n")) if contributors.strip() else 0
                self._add_metric("git.contributors.30_days", contributor_count, "count")
            
            # Repository size
            repo_size = self._run_command(["git", "count-objects", "-vH"])
            if repo_size:
                for line in repo_size.split("\n"):
                    if "size-pack" in line:
                        size_mb = float(line.split()[1].replace("M", "").replace("K", "0.001"))
                        self._add_metric("git.repository.size_mb", size_mb, "MB")
                        break
            
            # Branch metrics
            branches = self._run_command(["git", "branch", "-r"])
            if branches:
                branch_count = len([b for b in branches.split("\n") if b.strip() and not b.strip().startswith("origin/HEAD")])
                self._add_metric("git.branches.count", branch_count, "count")
            
            # Tags
            tags = self._run_command(["git", "tag", "--list"])
            if tags:
                tag_count = len([t for t in tags.split("\n") if t.strip()])
                self._add_metric("git.tags.count", tag_count, "count")
                
        except Exception as e:
            self.errors.append(f"Git metrics collection failed: {e}")
    
    async def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("📊 Collecting code quality metrics...")
        
        try:
            # Lines of code
            python_files = list(self.project_root.rglob("*.py"))
            if python_files:
                total_lines = 0
                total_files = 0
                for py_file in python_files:
                    if ".venv" not in str(py_file) and "__pycache__" not in str(py_file):
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                lines = len(f.readlines())
                                total_lines += lines
                                total_files += 1
                        except Exception:
                            pass
                
                self._add_metric("code.lines.total", total_lines, "lines")
                self._add_metric("code.files.python", total_files, "count")
                if total_files > 0:
                    self._add_metric("code.lines.average_per_file", total_lines / total_files, "lines")
            
            # Cyclomatic complexity (if radon is available)
            complexity_output = self._run_command(["radon", "cc", "--average", "."])
            if complexity_output and "Average complexity" in complexity_output:
                try:
                    avg_complexity = float(complexity_output.split("Average complexity: A (")[1].split(")")[0])
                    self._add_metric("code.complexity.average", avg_complexity, "score")
                except Exception:
                    pass
            
            # Maintainability index
            mi_output = self._run_command(["radon", "mi", "."])
            if mi_output:
                mi_scores = []
                for line in mi_output.split("\n"):
                    if ".py -" in line and "(" in line:
                        try:
                            score = float(line.split("(")[1].split(")")[0])
                            mi_scores.append(score)
                        except Exception:
                            pass
                
                if mi_scores:
                    self._add_metric("code.maintainability.average", sum(mi_scores) / len(mi_scores), "score")
                    self._add_metric("code.maintainability.min", min(mi_scores), "score")
            
            # Code duplication (if available)
            duplicate_output = self._run_command(["pylint", "--disable=all", "--enable=duplicate-code", "."])
            if duplicate_output:
                duplicate_count = duplicate_output.count("Similar lines")
                self._add_metric("code.duplication.instances", duplicate_count, "count")
                
        except Exception as e:
            self.errors.append(f"Code quality metrics collection failed: {e}")
    
    async def collect_test_metrics(self):
        """Collect testing metrics."""
        print("📊 Collecting test metrics...")
        
        try:
            # Test count
            test_files = list(self.project_root.rglob("test_*.py")) + list(self.project_root.rglob("*_test.py"))
            self._add_metric("tests.files.count", len(test_files), "count")
            
            # Run pytest with coverage (if available)
            coverage_output = self._run_command(["pytest", "--cov=.", "--cov-report=term-missing", "--tb=no", "-q"])
            if coverage_output:
                for line in coverage_output.split("\n"):
                    if "TOTAL" in line and "%" in line:
                        try:
                            coverage_pct = int(line.split("%")[0].split()[-1])
                            self._add_metric("tests.coverage.percentage", coverage_pct, "percent")
                        except Exception:
                            pass
                
                # Test results
                if "passed" in coverage_output:
                    passed_tests = coverage_output.count(" passed")
                    self._add_metric("tests.results.passed", passed_tests, "count")
                
                if "failed" in coverage_output:
                    failed_tests = coverage_output.count(" failed")
                    self._add_metric("tests.results.failed", failed_tests, "count")
            
            # Performance tests
            if (self.project_root / "tests" / "benchmarks").exists():
                benchmark_files = list((self.project_root / "tests" / "benchmarks").rglob("*.py"))
                self._add_metric("tests.benchmarks.count", len(benchmark_files), "count")
                
        except Exception as e:
            self.errors.append(f"Test metrics collection failed: {e}")
    
    async def collect_security_metrics(self):
        """Collect security metrics."""
        print("📊 Collecting security metrics...")
        
        try:
            # Bandit security scan
            bandit_output = self._run_command(["bandit", "-r", ".", "-f", "json"])
            if bandit_output:
                try:
                    bandit_data = json.loads(bandit_output)
                    high_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                    medium_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                    low_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"])
                    
                    self._add_metric("security.vulnerabilities.high", high_severity, "count")
                    self._add_metric("security.vulnerabilities.medium", medium_severity, "count")
                    self._add_metric("security.vulnerabilities.low", low_severity, "count")
                except json.JSONDecodeError:
                    pass
            
            # Safety check for dependencies
            safety_output = self._run_command(["safety", "check", "--json"])
            if safety_output:
                try:
                    safety_data = json.loads(safety_output)
                    vuln_count = len(safety_data)
                    self._add_metric("security.dependencies.vulnerabilities", vuln_count, "count")
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            self.errors.append(f"Security metrics collection failed: {e}")
    
    async def collect_dependency_metrics(self):
        """Collect dependency metrics."""
        print("📊 Collecting dependency metrics...")
        
        try:
            # Python dependencies
            if (self.project_root / "requirements.txt").exists():
                with open(self.project_root / "requirements.txt") as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    self._add_metric("dependencies.python.count", len(deps), "count")
            
            # Outdated packages
            outdated_output = self._run_command(["pip", "list", "--outdated", "--format=json"])
            if outdated_output:
                try:
                    outdated_data = json.loads(outdated_output)
                    self._add_metric("dependencies.python.outdated", len(outdated_data), "count")
                except json.JSONDecodeError:
                    pass
            
            # Docker dependencies
            if (self.project_root / "Dockerfile").exists():
                with open(self.project_root / "Dockerfile") as f:
                    dockerfile_content = f.read()
                    from_count = dockerfile_content.count("FROM ")
                    run_count = dockerfile_content.count("RUN ")
                    copy_count = dockerfile_content.count("COPY ")
                    
                    self._add_metric("docker.layers.from", from_count, "count")
                    self._add_metric("docker.layers.run", run_count, "count")
                    self._add_metric("docker.layers.copy", copy_count, "count")
                    
        except Exception as e:
            self.errors.append(f"Dependency metrics collection failed: {e}")
    
    async def collect_documentation_metrics(self):
        """Collect documentation metrics."""
        print("📊 Collecting documentation metrics...")
        
        try:
            # Documentation files
            doc_files = (
                list(self.project_root.rglob("*.md")) +
                list(self.project_root.rglob("*.rst")) +
                list(self.project_root.rglob("*.txt"))
            )
            
            # Filter out common non-doc files
            doc_files = [f for f in doc_files if "node_modules" not in str(f) and ".venv" not in str(f)]
            self._add_metric("docs.files.count", len(doc_files), "count")
            
            # Documentation coverage (if interrogate is available)
            interrogate_output = self._run_command(["interrogate", "-v", "."])
            if interrogate_output:
                for line in interrogate_output.split("\n"):
                    if "Overall coverage" in line and "%" in line:
                        try:
                            coverage_pct = float(line.split("%")[0].split()[-1])
                            self._add_metric("docs.coverage.percentage", coverage_pct, "percent")
                        except Exception:
                            pass
            
            # README quality
            readme_files = [f for f in doc_files if f.name.lower().startswith("readme")]
            if readme_files:
                readme_size = sum(f.stat().st_size for f in readme_files)
                self._add_metric("docs.readme.size_bytes", readme_size, "bytes")
                
        except Exception as e:
            self.errors.append(f"Documentation metrics collection failed: {e}")
    
    async def collect_performance_metrics(self):
        """Collect performance metrics."""
        print("📊 Collecting performance metrics...")
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self._add_metric("system.cpu.percent", cpu_percent, "percent")
            self._add_metric("system.memory.percent", memory.percent, "percent")
            self._add_metric("system.memory.available_gb", memory.available / 1024**3, "GB")
            self._add_metric("system.disk.percent", (disk.used / disk.total) * 100, "percent")
            self._add_metric("system.disk.free_gb", disk.free / 1024**3, "GB")
            
            # Build time estimation
            if (self.project_root / "Dockerfile").exists():
                start_time = time.time()
                # Simulate build time check (just validate Dockerfile)
                self._run_command(["docker", "build", "--dry-run", "."])
                build_check_time = time.time() - start_time
                self._add_metric("docker.build.validation_time_ms", build_check_time * 1000, "ms")
                
        except Exception as e:
            self.errors.append(f"Performance metrics collection failed: {e}")
    
    async def collect_all_metrics(self) -> MetricsReport:
        """Collect all enabled metrics."""
        collection_start = time.time()
        
        tasks = []
        if self.config.get("git_metrics", True):
            tasks.append(self.collect_git_metrics())
        if self.config.get("code_quality_metrics", True):
            tasks.append(self.collect_code_quality_metrics())
        if self.config.get("test_metrics", True):
            tasks.append(self.collect_test_metrics())
        if self.config.get("security_metrics", True):
            tasks.append(self.collect_security_metrics())
        if self.config.get("dependency_metrics", True):
            tasks.append(self.collect_dependency_metrics())
        if self.config.get("documentation_metrics", True):
            tasks.append(self.collect_documentation_metrics())
        if self.config.get("performance_metrics", True):
            tasks.append(self.collect_performance_metrics())
        
        # Run all collections concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        collection_duration = int((time.time() - collection_start) * 1000)
        
        # Generate summary
        summary = {
            "total_metrics": len(self.metrics),
            "categories": defaultdict(int),
            "collection_time_ms": collection_duration,
            "errors_count": len(self.errors)
        }
        
        for metric in self.metrics:
            category = metric.name.split('.')[0]
            summary["categories"][category] += 1
        
        return MetricsReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            collection_duration_ms=collection_duration,
            metrics=self.metrics,
            summary=dict(summary),
            errors=self.errors
        )
    
    def save_report(self, report: MetricsReport):
        """Save metrics report to file."""
        output_file = Path(self.config.get("output_file", "metrics-report.json"))
        
        # Convert to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "collection_duration_ms": report.collection_duration_ms,
            "metrics": [asdict(m) for m in report.metrics],
            "summary": report.summary,
            "errors": report.errors
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"📊 Metrics report saved to: {output_file}")
        return output_file
    
    def print_summary(self, report: MetricsReport):
        """Print metrics summary to console."""
        print("\n" + "="*60)
        print("📊 METRICS COLLECTION SUMMARY")
        print("="*60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Collection Duration: {report.collection_duration_ms}ms")
        print(f"Total Metrics: {report.summary['total_metrics']}")
        print(f"Errors: {report.summary['errors_count']}")
        
        print("\nMetrics by Category:")
        for category, count in report.summary["categories"].items():
            print(f"  {category}: {count}")
        
        if report.errors:
            print("\n⚠️  Errors:")
            for error in report.errors:
                print(f"  - {error}")
        
        # Highlight key metrics
        key_metrics = [
            ("git.commits.30_days", "Recent Commits (30d)"),
            ("tests.coverage.percentage", "Test Coverage"),
            ("security.vulnerabilities.high", "High Security Issues"),
            ("code.complexity.average", "Average Complexity"),
            ("dependencies.python.outdated", "Outdated Dependencies")
        ]
        
        print("\n🎯 Key Metrics:")
        for metric_name, display_name in key_metrics:
            for metric in report.metrics:
                if metric.name == metric_name:
                    print(f"  {display_name}: {metric.value} {metric.unit}")
                    break


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced metrics collection for MicroDiff-MatDesign")
    parser.add_argument("--config", type=Path, help="Metrics configuration file")
    parser.add_argument("--output", type=Path, help="Output file for metrics report")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    collector = AdvancedMetricsCollector(config_path=args.config)
    
    if args.output:
        collector.config["output_file"] = str(args.output)
    
    print("🚀 Starting advanced metrics collection...")
    
    try:
        report = await collector.collect_all_metrics()
        output_file = collector.save_report(report)
        
        if not args.quiet:
            collector.print_summary(report)
        
        print(f"\n✅ Metrics collection completed successfully!")
        print(f"📄 Report saved to: {output_file}")
        
        # Exit with non-zero code if there were errors
        if report.errors:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Metrics collection interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
