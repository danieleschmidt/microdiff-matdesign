#!/usr/bin/env python3
"""
Code quality monitoring script for MicroDiff-MatDesign.

This script monitors code quality metrics over time and generates reports
to track improvements and identify areas needing attention.
"""

import os
import sys
import json
import subprocess
import datetime
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

class CodeQualityMonitor:
    """Monitor and track code quality metrics over time."""
    
    def __init__(self, project_root: str = ".", db_path: str = ".quality_metrics.db"):
        self.project_root = Path(project_root).resolve()
        self.db_path = self.project_root / db_path
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize SQLite database for storing metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                file_path TEXT,
                details TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON quality_metrics(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_metric_name ON quality_metrics(metric_name)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run([
                'git', 'rev-parse', 'HEAD'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def collect_coverage_metrics(self) -> Dict[str, Any]:
        """Collect test coverage metrics."""
        print("📊 Collecting coverage metrics...")
        
        metrics = {}
        
        try:
            # Run tests with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/',
                '--cov=microdiff_matdesign',
                '--cov-report=json',
                '--cov-report=term',
                '--quiet'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse coverage JSON report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                # Overall coverage
                metrics['overall_coverage'] = coverage_data['totals']['percent_covered']
                metrics['lines_covered'] = coverage_data['totals']['covered_lines']
                metrics['total_lines'] = coverage_data['totals']['num_statements']
                metrics['missing_lines'] = coverage_data['totals']['missing_lines']
                
                # Per-file coverage
                file_coverage = {}
                for file_path, file_data in coverage_data['files'].items():
                    if 'microdiff_matdesign' in file_path:
                        file_coverage[file_path] = {
                            'coverage': file_data['summary']['percent_covered'],
                            'lines_covered': file_data['summary']['covered_lines'],
                            'total_lines': file_data['summary']['num_statements']
                        }
                
                metrics['file_coverage'] = file_coverage
                
                # Clean up
                coverage_file.unlink()
            
        except Exception as e:
            print(f"⚠️ Error collecting coverage metrics: {e}")
        
        return metrics
    
    def collect_complexity_metrics(self) -> Dict[str, Any]:
        """Collect code complexity metrics using radon."""
        print("🧮 Collecting complexity metrics...")
        
        metrics = {}
        
        try:
            # Install radon if not available
            try:
                import radon
            except ImportError:
                subprocess.run([
                    'pip', 'install', 'radon'
                ], check=True, capture_output=True)
            
            # Run radon for cyclomatic complexity
            result = subprocess.run([
                'python', '-m', 'radon', 'cc', 'microdiff_matdesign',
                '--json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout:
                complexity_data = json.loads(result.stdout)
                
                total_complexity = 0
                function_count = 0
                file_complexities = {}
                
                for file_path, functions in complexity_data.items():
                    file_complexity = 0
                    file_functions = 0
                    
                    for func in functions:
                        if isinstance(func, dict) and 'complexity' in func:
                            complexity = func['complexity']
                            total_complexity += complexity
                            file_complexity += complexity
                            function_count += 1
                            file_functions += 1
                    
                    if file_functions > 0:
                        file_complexities[file_path] = {
                            'average_complexity': file_complexity / file_functions,
                            'total_complexity': file_complexity,
                            'function_count': file_functions
                        }
                
                metrics['average_complexity'] = total_complexity / max(function_count, 1)
                metrics['total_complexity'] = total_complexity
                metrics['function_count'] = function_count
                metrics['file_complexities'] = file_complexities
            
            # Run radon for maintainability index
            result = subprocess.run([
                'python', '-m', 'radon', 'mi', 'microdiff_matdesign',
                '--json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout:
                mi_data = json.loads(result.stdout)
                
                mi_scores = []
                file_mi_scores = {}
                
                for file_path, mi_score in mi_data.items():
                    if isinstance(mi_score, (int, float)):
                        mi_scores.append(mi_score)
                        file_mi_scores[file_path] = mi_score
                
                if mi_scores:
                    metrics['average_maintainability'] = sum(mi_scores) / len(mi_scores)
                    metrics['file_maintainability'] = file_mi_scores
            
        except Exception as e:
            print(f"⚠️ Error collecting complexity metrics: {e}")
        
        return metrics
    
    def collect_duplication_metrics(self) -> Dict[str, Any]:
        """Collect code duplication metrics."""
        print("🔍 Collecting duplication metrics...")
        
        metrics = {}
        
        try:
            # Use simple line-based duplication detection
            # In practice, you might want to use tools like jscpd or sonar
            
            file_lines = {}
            total_lines = 0
            
            # Read all Python files
            for py_file in self.project_root.rglob('microdiff_matdesign/**/*.py'):
                if '__pycache__' not in str(py_file):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() 
                                   if line.strip() and not line.strip().startswith('#')]
                            file_lines[str(py_file)] = lines
                            total_lines += len(lines)
                    except Exception:
                        pass
            
            # Find duplicate lines
            line_counts = {}
            for file_path, lines in file_lines.items():
                for line in lines:
                    if len(line) > 20:  # Only check substantial lines
                        line_counts[line] = line_counts.get(line, 0) + 1
            
            duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
            duplication_ratio = (duplicate_lines / max(total_lines, 1)) * 100
            
            metrics['duplicate_lines'] = duplicate_lines
            metrics['total_lines'] = total_lines
            metrics['duplication_percentage'] = duplication_ratio
            
        except Exception as e:
            print(f"⚠️ Error collecting duplication metrics: {e}")
        
        return metrics
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation coverage metrics."""
        print("📚 Collecting documentation metrics...")
        
        metrics = {}
        
        try:
            # Use interrogate for docstring coverage
            result = subprocess.run([
                'python', '-c', '''
import subprocess
import sys
try:
    result = subprocess.run([
        "python", "-m", "interrogate", "microdiff_matdesign",
        "--quiet", "--verbose"
    ], capture_output=True, text=True)
    
    output = result.stdout
    for line in output.split("\\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "%" in part:
                    coverage = float(part.replace("%", ""))
                    print(f"docstring_coverage:{coverage}")
                    break
except ImportError:
    print("interrogate not available")
except Exception as e:
    print(f"error:{e}")
'''
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        if key == 'docstring_coverage':
                            metrics['docstring_coverage'] = float(value)
            
            # Count documentation files
            doc_files = list(self.project_root.rglob('docs/**/*.md'))
            doc_files.extend(self.project_root.rglob('*.md'))
            
            metrics['documentation_files'] = len(doc_files)
            
            # Check if key documentation exists
            key_docs = ['README.md', 'CONTRIBUTING.md', 'docs/']
            existing_docs = []
            for doc in key_docs:
                if (self.project_root / doc).exists():
                    existing_docs.append(doc)
            
            metrics['key_documentation_present'] = len(existing_docs)
            metrics['key_documentation_total'] = len(key_docs)
            
        except Exception as e:
            print(f"⚠️ Error collecting documentation metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        print("🔒 Collecting security metrics...")
        
        metrics = {}
        
        try:
            # Run bandit security scan
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', 'microdiff_matdesign',
                '-f', 'json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    issue_counts = {'high': 0, 'medium': 0, 'low': 0}
                    for issue in bandit_data.get('results', []):
                        severity = issue.get('issue_severity', '').lower()
                        if severity in issue_counts:
                            issue_counts[severity] += 1
                    
                    metrics['security_issues_high'] = issue_counts['high']
                    metrics['security_issues_medium'] = issue_counts['medium']
                    metrics['security_issues_low'] = issue_counts['low']
                    metrics['security_issues_total'] = sum(issue_counts.values())
                    
                except json.JSONDecodeError:
                    pass
            
            # Check for hardcoded secrets (simple patterns)
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{8,}["\']',
                r'token\s*=\s*["\'][^"\']{8,}["\']'
            ]
            
            potential_secrets = 0
            for py_file in self.project_root.rglob('microdiff_matdesign/**/*.py'):
                if '__pycache__' not in str(py_file):
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read()
                            for pattern in secret_patterns:
                                import re
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                potential_secrets += len(matches)
                    except Exception:
                        pass
            
            metrics['potential_secrets'] = potential_secrets
            
        except Exception as e:
            print(f"⚠️ Error collecting security metrics: {e}")
        
        return metrics
    
    def store_metrics(self, metrics: Dict[str, Any], commit_hash: Optional[str] = None) -> None:
        """Store collected metrics in the database."""
        print("💾 Storing metrics in database...")
        
        timestamp = datetime.datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        def store_metric(name: str, value: float, unit: str = '', file_path: str = '', details: str = ''):
            cursor.execute('''
                INSERT INTO quality_metrics 
                (timestamp, commit_hash, metric_name, metric_value, metric_unit, file_path, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, commit_hash, name, value, unit, file_path, details))
        
        # Store top-level metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                unit = 'percentage' if 'coverage' in metric_name or 'percentage' in metric_name else 'count'
                store_metric(metric_name, metric_value, unit)
            elif isinstance(metric_value, dict):
                # Store nested metrics (like per-file data)
                for sub_key, sub_value in metric_value.items():
                    if isinstance(sub_value, (int, float)):
                        store_metric(f"{metric_name}_{sub_key}", sub_value, '', sub_key)
                    elif isinstance(sub_value, dict):
                        # Handle deeply nested data
                        for deep_key, deep_value in sub_value.items():
                            if isinstance(deep_value, (int, float)):
                                store_metric(
                                    f"{metric_name}_{deep_key}", 
                                    deep_value, 
                                    '', 
                                    sub_key, 
                                    json.dumps(sub_value)
                                )
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, metric_name: str, days: int = 30) -> List[Tuple[str, float]]:
        """Get historical data for a specific metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT timestamp, metric_value 
            FROM quality_metrics 
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp
        ''', (metric_name, cutoff_date))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def generate_trend_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a trend report for key metrics."""
        print(f"📈 Generating trend report for last {days} days...")
        
        key_metrics = [
            'overall_coverage',
            'average_complexity',
            'average_maintainability',
            'duplication_percentage',
            'docstring_coverage',
            'security_issues_total'
        ]
        
        trend_report = {
            'period_days': days,
            'generated_at': datetime.datetime.utcnow().isoformat(),
            'metrics': {}
        }
        
        for metric in key_metrics:
            historical_data = self.get_historical_data(metric, days)
            
            if len(historical_data) >= 2:
                # Calculate trend
                first_value = historical_data[0][1]
                last_value = historical_data[-1][1]
                
                if first_value != 0:
                    trend_percentage = ((last_value - first_value) / first_value) * 100
                else:
                    trend_percentage = 0
                
                trend_report['metrics'][metric] = {
                    'first_value': first_value,
                    'last_value': last_value,
                    'trend_percentage': round(trend_percentage, 2),
                    'data_points': len(historical_data),
                    'trend_direction': 'improving' if trend_percentage > 0 else 'declining' if trend_percentage < 0 else 'stable'
                }
            else:
                trend_report['metrics'][metric] = {
                    'status': 'insufficient_data',
                    'data_points': len(historical_data)
                }
        
        return trend_report
    
    def run_quality_check(self) -> Dict[str, Any]:
        """Run complete quality check and return all metrics."""
        print("🔍 Running comprehensive code quality check...")
        
        commit_hash = self.get_current_commit()
        all_metrics = {}
        
        # Collect all metrics
        collectors = [
            ('coverage', self.collect_coverage_metrics),
            ('complexity', self.collect_complexity_metrics),
            ('duplication', self.collect_duplication_metrics),
            ('documentation', self.collect_documentation_metrics),
            ('security', self.collect_security_metrics),
        ]
        
        for collector_name, collector_func in collectors:
            try:
                metrics = collector_func()
                all_metrics.update(metrics)
                print(f"✅ {collector_name.title()} metrics collected")
            except Exception as e:
                print(f"❌ Failed to collect {collector_name} metrics: {e}")
        
        # Store metrics
        self.store_metrics(all_metrics, commit_hash)
        
        return all_metrics


def main():
    """Main function for the code quality monitor."""
    parser = argparse.ArgumentParser(description='Code quality monitoring')
    parser.add_argument('--project-root', default='.', 
                      help='Project root directory')
    parser.add_argument('--db-path', default='.quality_metrics.db',
                      help='Database file path')
    parser.add_argument('--trend-days', type=int, default=30,
                      help='Days to include in trend report')
    parser.add_argument('--output-format', choices=['json', 'markdown'], default='json',
                      help='Output format')
    parser.add_argument('--report-file',
                      help='Save report to file')
    
    args = parser.parse_args()
    
    try:
        monitor = CodeQualityMonitor(args.project_root, args.db_path)
        
        # Run quality check
        metrics = monitor.run_quality_check()
        
        # Generate trend report
        trend_report = monitor.generate_trend_report(args.trend_days)
        
        # Combine reports
        full_report = {
            'current_metrics': metrics,
            'trend_analysis': trend_report,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        
        # Output report
        if args.output_format == 'json':
            output = json.dumps(full_report, indent=2)
        else:
            # Generate markdown report
            lines = [
                "# 📊 Code Quality Report",
                "",
                f"**Generated**: {full_report['timestamp']}",
                "",
                "## Current Metrics",
                ""
            ]
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: {value}")
            
            lines.extend([
                "",
                "## Trend Analysis",
                ""
            ])
            
            for metric, trend_data in trend_report['metrics'].items():
                if 'trend_percentage' in trend_data:
                    direction = trend_data['trend_direction']
                    percentage = trend_data['trend_percentage']
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: {direction} ({percentage:+.1f}%)")
            
            output = '\n'.join(lines)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(output)
            print(f"📄 Quality report saved to {args.report_file}")
        else:
            print("\n" + "="*60)
            print("QUALITY REPORT")
            print("="*60)
            print(output)
        
    except KeyboardInterrupt:
        print("\n❌ Quality check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()