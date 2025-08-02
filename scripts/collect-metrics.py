#!/usr/bin/env python3
"""
Automated metrics collection script for MicroDiff-MatDesign project.

This script collects various metrics from different sources and updates
the project metrics tracking system.
"""

import json
import os
import sys
import subprocess
import requests
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

class MetricsCollector:
    """Collect and aggregate project metrics from various sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.metrics_data = self._load_metrics_config()
        self.collected_metrics = {}
        
    def _load_metrics_config(self) -> Dict[str, Any]:
        """Load metrics configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Metrics config file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in metrics config: {e}")
            sys.exit(1)
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("📊 Collecting Git metrics...")
        
        metrics = {}
        
        try:
            # Commit frequency (last 30 days)
            result = subprocess.run([
                'git', 'log', '--since=30.days.ago', '--oneline'
            ], capture_output=True, text=True, check=True)
            
            commit_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics['commit_frequency'] = commit_count
            
            # Contributors (last 30 days)
            result = subprocess.run([
                'git', 'log', '--since=30.days.ago', '--format=%ae'
            ], capture_output=True, text=True, check=True)
            
            contributors = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
            metrics['active_contributors'] = len(contributors)
            
            # Lines of code
            result = subprocess.run([
                'find', '.', '-name', '*.py', '-not', '-path', './.*', 
                '-not', '-path', './venv/*', '-not', '-path', './env/*',
                '-exec', 'wc', '-l', '{}', '+'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = 0
                for line in lines[:-1]:  # Skip total line
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        total_lines += int(parts[0])
                metrics['lines_of_code'] = total_lines
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error collecting Git metrics: {e}")
        
        return metrics
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect test coverage and execution metrics."""
        print("🧪 Collecting test metrics...")
        
        metrics = {}
        
        try:
            # Run tests with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', 
                '--cov=microdiff_matdesign',
                '--cov-report=json',
                '--quiet'
            ], capture_output=True, text=True)
            
            # Parse coverage report
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                metrics['test_coverage'] = round(coverage_data['totals']['percent_covered'], 2)
                metrics['lines_covered'] = coverage_data['totals']['covered_lines']
                metrics['total_lines'] = coverage_data['totals']['num_statements']
                
                # Clean up
                coverage_file.unlink()
            
            # Test execution time (approximate from output)
            if result.returncode == 0:
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'seconds' in line and ('passed' in line or 'failed' in line):
                        # Extract execution time
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'seconds' in part:
                                try:
                                    exec_time = float(parts[i-1])
                                    metrics['test_execution_time'] = exec_time
                                    break
                                except (ValueError, IndexError):
                                    pass
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error collecting test metrics: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error in test metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security vulnerability metrics."""
        print("🔒 Collecting security metrics...")
        
        metrics = {
            'vulnerabilities': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        try:
            # Run safety check
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        severity = vuln.get('vulnerability_id', '').lower()
                        if 'critical' in severity:
                            metrics['vulnerabilities']['critical'] += 1
                        elif 'high' in severity:
                            metrics['vulnerabilities']['high'] += 1
                        elif 'medium' in severity:
                            metrics['vulnerabilities']['medium'] += 1
                        else:
                            metrics['vulnerabilities']['low'] += 1
                except json.JSONDecodeError:
                    pass
            
            # Run bandit security check
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', 'microdiff_matdesign',
                '-f', 'json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    for issue in bandit_data.get('results', []):
                        severity = issue.get('issue_severity', '').lower()
                        if severity == 'high':
                            metrics['vulnerabilities']['high'] += 1
                        elif severity == 'medium':
                            metrics['vulnerabilities']['medium'] += 1
                        elif severity == 'low':
                            metrics['vulnerabilities']['low'] += 1
                except json.JSONDecodeError:
                    pass
        
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error collecting security metrics: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error in security metrics: {e}")
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency freshness metrics."""
        print("📦 Collecting dependency metrics...")
        
        metrics = {}
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                'python', '-m', 'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                metrics['outdated_dependencies'] = len(outdated_packages)
                
                # Calculate average age of outdated packages
                if outdated_packages:
                    # This is a simplified calculation
                    # In practice, you'd need to check package release dates
                    metrics['dependency_freshness_days'] = len(outdated_packages) * 7  # Rough estimate
                else:
                    metrics['dependency_freshness_days'] = 0
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error collecting dependency metrics: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error in dependency metrics: {e}")
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("⚡ Collecting code quality metrics...")
        
        metrics = {}
        
        try:
            # Run complexity analysis with radon
            result = subprocess.run([
                'python', '-c', '''
import subprocess
import sys
try:
    import radon.complexity as cc
    import radon.raw as raw
    from pathlib import Path
    
    total_complexity = 0
    file_count = 0
    total_loc = 0
    
    for py_file in Path("microdiff_matdesign").rglob("*.py"):
        if "__pycache__" not in str(py_file):
            try:
                with open(py_file, "r") as f:
                    code = f.read()
                
                # Complexity
                complexity = cc.cc_visit(code)
                for item in complexity:
                    if hasattr(item, "complexity"):
                        total_complexity += item.complexity
                
                # Lines of code
                raw_metrics = raw.analyze(code)
                total_loc += raw_metrics.loc
                file_count += 1
            except Exception:
                pass
    
    avg_complexity = total_complexity / max(file_count, 1)
    print(f"complexity:{avg_complexity:.2f}")
    print(f"loc:{total_loc}")
    print(f"files:{file_count}")
except ImportError:
    print("radon not available")
'''
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and 'radon not available' not in result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            if key == 'complexity':
                                metrics['average_complexity'] = float(value)
                            elif key == 'loc':
                                metrics['lines_of_code'] = int(value)
                            elif key == 'files':
                                metrics['python_files'] = int(value)
                        except ValueError:
                            pass
            
            # Check for TODO/FIXME comments
            result = subprocess.run([
                'grep', '-r', '--include=*.py', '-c', 'TODO\|FIXME\|XXX', 
                'microdiff_matdesign/'
            ], capture_output=True, text=True)
            
            todo_count = 0
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        try:
                            count = int(line.split(':')[1])
                            todo_count += count
                        except (ValueError, IndexError):
                            pass
            
            metrics['technical_debt_items'] = todo_count
            
        except Exception as e:
            print(f"⚠️ Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_build_metrics(self) -> Dict[str, Any]:
        """Collect build and CI/CD metrics."""
        print("🏗️ Collecting build metrics...")
        
        metrics = {}
        
        # These would typically come from CI/CD system APIs
        # For now, we'll simulate or collect what we can
        
        try:
            # Measure Docker build time
            start_time = datetime.datetime.now()
            result = subprocess.run([
                'docker', 'build', '-t', 'microdiff-test', '.'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                build_time = (datetime.datetime.now() - start_time).total_seconds()
                metrics['docker_build_time'] = round(build_time, 2)
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error collecting build metrics: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error in build metrics: {e}")
        
        return metrics
    
    def update_metrics_config(self, collected_metrics: Dict[str, Any]) -> None:
        """Update the metrics configuration with collected values."""
        print("📝 Updating metrics configuration...")
        
        # Update current values in the metrics structure
        for category in self.metrics_data.get('metrics', {}):
            for subcategory in self.metrics_data['metrics'][category]:
                for metric_name in self.metrics_data['metrics'][category][subcategory]:
                    metric_config = self.metrics_data['metrics'][category][subcategory][metric_name]
                    
                    # Find matching collected metric
                    if metric_name in collected_metrics:
                        metric_config['current'] = collected_metrics[metric_name]
                        metric_config['last_updated'] = datetime.datetime.utcnow().isoformat()
        
        # Update last updated timestamp
        self.metrics_data['repository']['last_updated'] = datetime.datetime.utcnow().isoformat()
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
    
    def generate_report(self, output_format: str = 'json') -> str:
        """Generate a metrics report."""
        print("📋 Generating metrics report...")
        
        report_data = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'collected_metrics': self.collected_metrics,
            'summary': {
                'total_metrics': len(self.collected_metrics),
                'collection_duration': 'N/A'
            }
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2)
        elif output_format == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            return str(report_data)
    
    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate a markdown formatted report."""
        lines = [
            "# 📊 Project Metrics Report",
            "",
            f"**Generated**: {data['timestamp']}",
            f"**Total Metrics**: {data['summary']['total_metrics']}",
            "",
            "## Collected Metrics",
            ""
        ]
        
        for metric, value in data['collected_metrics'].items():
            if isinstance(value, dict):
                lines.append(f"### {metric.replace('_', ' ').title()}")
                lines.append("")
                for sub_metric, sub_value in value.items():
                    lines.append(f"- **{sub_metric.replace('_', ' ').title()}**: {sub_value}")
                lines.append("")
            else:
                lines.append(f"- **{metric.replace('_', ' ').title()}**: {value}")
        
        return '\n'.join(lines)
    
    def run_collection(self) -> Dict[str, Any]:
        """Run the complete metrics collection process."""
        print("🚀 Starting metrics collection...")
        start_time = datetime.datetime.now()
        
        # Collect all metrics
        collectors = [
            ('git', self.collect_git_metrics),
            ('test', self.collect_test_metrics),
            ('security', self.collect_security_metrics),
            ('dependency', self.collect_dependency_metrics),
            ('code_quality', self.collect_code_quality_metrics),
            ('build', self.collect_build_metrics),
        ]
        
        for collector_name, collector_func in collectors:
            try:
                metrics = collector_func()
                self.collected_metrics.update(metrics)
                print(f"✅ {collector_name.title()} metrics collected")
            except Exception as e:
                print(f"❌ Failed to collect {collector_name} metrics: {e}")
        
        collection_duration = (datetime.datetime.now() - start_time).total_seconds()
        print(f"⏱️ Collection completed in {collection_duration:.2f} seconds")
        
        return self.collected_metrics


def main():
    """Main function for the metrics collection script."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                      help='Path to metrics configuration file')
    parser.add_argument('--output', choices=['json', 'markdown'], default='json',
                      help='Output format for the report')
    parser.add_argument('--update-config', action='store_true',
                      help='Update the metrics configuration with collected values')
    parser.add_argument('--report-file', 
                      help='Save report to file instead of printing to stdout')
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.config)
        collected_metrics = collector.run_collection()
        
        if args.update_config:
            collector.update_metrics_config(collected_metrics)
            print("✅ Metrics configuration updated")
        
        report = collector.generate_report(args.output)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {args.report_file}")
        else:
            print("\n" + "="*50)
            print("METRICS REPORT")
            print("="*50)
            print(report)
        
    except KeyboardInterrupt:
        print("\n❌ Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()