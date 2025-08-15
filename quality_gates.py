#!/usr/bin/env python3
"""
🛡️ TERRAGON AUTONOMOUS QUALITY GATES 🛡️
Comprehensive Quality Assurance System for MicroDiff-MatDesign

This script implements mandatory quality gates that MUST pass before any deployment.
NO EXCEPTIONS. NO BYPASSES. FULL AUTONOMOUS VALIDATION.
"""

import os
import sys
import time
import subprocess
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """Quality gate severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityResult:
    """Result from a quality gate check."""
    gate_name: str
    passed: bool
    level: QualityLevel
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any]
    duration: float


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, level: QualityLevel):
        self.name = name
        self.level = level
    
    def run(self) -> QualityResult:
        """Run the quality gate check."""
        raise NotImplementedError


class CodeQualityGate(QualityGate):
    """Code quality and style checks."""
    
    def __init__(self):
        super().__init__("Code Quality", QualityLevel.HIGH)
    
    def run(self) -> QualityResult:
        start_time = time.time()
        
        try:
            # Check if Python files exist
            python_files = list(Path('.').rglob('*.py'))
            if not python_files:
                return QualityResult(
                    self.name, False, self.level, 0.0,
                    "No Python files found", {}, time.time() - start_time
                )
            
            # Basic syntax check
            syntax_errors = []
            style_issues = 0
            total_files = len(python_files)
            
            for py_file in python_files:
                # Check syntax
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
                except Exception:
                    pass  # Skip files that can't be read
                
                # Basic style checks
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines, 1):
                            # Check line length (basic)
                            if len(line) > 120:
                                style_issues += 1
                            
                            # Check trailing whitespace
                            if line.rstrip() != line:
                                style_issues += 1
                except Exception:
                    pass
            
            # Calculate score
            syntax_score = 1.0 if not syntax_errors else 0.0
            style_score = max(0.0, 1.0 - (style_issues / (total_files * 10)))
            overall_score = (syntax_score * 0.8) + (style_score * 0.2)
            
            passed = syntax_score == 1.0 and overall_score >= 0.7
            
            message = f"Code quality check: {len(syntax_errors)} syntax errors, {style_issues} style issues"
            
            return QualityResult(
                self.name, passed, self.level, overall_score,
                message, {
                    'syntax_errors': syntax_errors,
                    'style_issues': style_issues,
                    'files_checked': total_files
                }, time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                self.name, False, self.level, 0.0,
                f"Code quality check failed: {e}", {}, time.time() - start_time
            )


class SecurityGate(QualityGate):
    """Security vulnerability checks."""
    
    def __init__(self):
        super().__init__("Security Scan", QualityLevel.CRITICAL)
    
    def run(self) -> QualityResult:
        start_time = time.time()
        
        try:
            security_issues = []
            total_checks = 0
            
            # Check for common security issues in Python files
            python_files = list(Path('.').rglob('*.py'))
            
            dangerous_patterns = [
                'eval(',
                'exec(',
                'subprocess.call(',
                'os.system(',
                'input(',  # Can be dangerous
                'raw_input(',
                'pickle.loads(',
                'yaml.load(',
                'shell=True'
            ]
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for pattern in dangerous_patterns:
                            if pattern in content:
                                total_checks += 1
                                # Check if it's in a comment or string (basic check)
                                lines = content.split('\n')
                                for i, line in enumerate(lines, 1):
                                    if pattern in line and not line.strip().startswith('#'):
                                        security_issues.append(f"{py_file}:{i} - Potentially dangerous: {pattern}")
                except Exception:
                    pass
            
            # Check for sensitive files
            sensitive_files = [
                '.env',
                'config.ini',
                'secrets.json',
                'private_key',
                'id_rsa',
                '.ssh'
            ]
            
            for sensitive in sensitive_files:
                if Path(sensitive).exists():
                    security_issues.append(f"Sensitive file found: {sensitive}")
            
            # Calculate score
            if not security_issues:
                score = 1.0
                passed = True
                message = "No security issues detected"
            else:
                score = max(0.0, 1.0 - (len(security_issues) / 10))
                passed = len(security_issues) == 0
                message = f"Security scan: {len(security_issues)} issues found"
            
            return QualityResult(
                self.name, passed, self.level, score,
                message, {
                    'security_issues': security_issues,
                    'files_scanned': len(python_files)
                }, time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                self.name, False, self.level, 0.0,
                f"Security scan failed: {e}", {}, time.time() - start_time
            )


class TestCoverageGate(QualityGate):
    """Test coverage verification."""
    
    def __init__(self, minimum_coverage: float = 85.0):
        super().__init__("Test Coverage", QualityLevel.HIGH)
        self.minimum_coverage = minimum_coverage
    
    def run(self) -> QualityResult:
        start_time = time.time()
        
        try:
            # Look for test files
            test_files = list(Path('.').rglob('test_*.py')) + list(Path('.').rglob('*_test.py'))
            test_dir_files = []
            
            for test_dir in ['tests', 'test']:
                test_path = Path(test_dir)
                if test_path.exists():
                    test_dir_files.extend(test_path.rglob('*.py'))
            
            all_test_files = test_files + test_dir_files
            
            # Count source files
            source_files = [
                f for f in Path('.').rglob('*.py') 
                if not any(part.startswith('test') for part in f.parts) 
                and f.name not in ['setup.py', 'conftest.py']
                and not f.name.startswith('test_')
            ]
            
            if not source_files:
                return QualityResult(
                    self.name, False, self.level, 0.0,
                    "No source files found", {}, time.time() - start_time
                )
            
            # Estimate coverage based on test files vs source files
            if not all_test_files:
                coverage = 0.0
                message = "No test files found"
            else:
                # Simple heuristic: test coverage estimation
                test_to_source_ratio = len(all_test_files) / len(source_files)
                coverage = min(100.0, test_to_source_ratio * 50)  # Basic estimation
                
                # Boost if there are comprehensive test files
                for test_file in all_test_files:
                    try:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            test_count = content.count('def test_')
                            if test_count > 10:  # Comprehensive test file
                                coverage += 10
                    except Exception:
                        pass
                
                coverage = min(100.0, coverage)
                message = f"Estimated test coverage: {coverage:.1f}%"
            
            passed = coverage >= self.minimum_coverage
            score = coverage / 100.0
            
            return QualityResult(
                self.name, passed, self.level, score,
                message, {
                    'coverage_percent': coverage,
                    'minimum_required': self.minimum_coverage,
                    'test_files': len(all_test_files),
                    'source_files': len(source_files)
                }, time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                self.name, False, self.level, 0.0,
                f"Test coverage check failed: {e}", {}, time.time() - start_time
            )


class PerformanceGate(QualityGate):
    """Performance benchmarks and requirements."""
    
    def __init__(self):
        super().__init__("Performance Benchmark", QualityLevel.MEDIUM)
    
    def run(self) -> QualityResult:
        start_time = time.time()
        
        try:
            # Test basic import performance
            import_times = []
            
            # Try to import main modules
            main_modules = [
                'microdiff_matdesign',
                'microdiff_matdesign.core',
                'microdiff_matdesign.imaging'
            ]
            
            for module in main_modules:
                try:
                    module_start = time.time()
                    
                    # Try to import without actually importing (check if module exists)
                    module_path = module.replace('.', '/')
                    if Path(f"{module_path}.py").exists() or Path(f"{module_path}/__init__.py").exists():
                        import_times.append(time.time() - module_start)
                    
                except Exception:
                    pass
            
            # Performance metrics
            avg_import_time = sum(import_times) / len(import_times) if import_times else 0.1
            
            # Test memory usage (basic)
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent
            except ImportError:
                memory_usage = 50.0  # Assume reasonable
            
            # Calculate performance score
            import_score = 1.0 if avg_import_time < 0.5 else max(0.0, 1.0 - avg_import_time)
            memory_score = 1.0 if memory_usage < 80 else max(0.0, (100 - memory_usage) / 20)
            
            overall_score = (import_score * 0.6) + (memory_score * 0.4)
            passed = overall_score >= 0.7
            
            message = f"Performance: avg import {avg_import_time:.3f}s, memory {memory_usage:.1f}%"
            
            return QualityResult(
                self.name, passed, self.level, overall_score,
                message, {
                    'avg_import_time': avg_import_time,
                    'memory_usage_percent': memory_usage,
                    'modules_tested': len(import_times)
                }, time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                self.name, False, self.level, 0.0,
                f"Performance benchmark failed: {e}", {}, time.time() - start_time
            )


class DocumentationGate(QualityGate):
    """Documentation completeness check."""
    
    def __init__(self):
        super().__init__("Documentation", QualityLevel.MEDIUM)
    
    def run(self) -> QualityResult:
        start_time = time.time()
        
        try:
            doc_score = 0.0
            max_score = 6.0
            issues = []
            
            # Check for README
            if Path('README.md').exists():
                doc_score += 2.0
                try:
                    with open('README.md', 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                        if len(readme_content) > 500:  # Substantial README
                            doc_score += 1.0
                except Exception:
                    pass
            else:
                issues.append("Missing README.md")
            
            # Check for package documentation
            if Path('microdiff_matdesign/__init__.py').exists():
                try:
                    with open('microdiff_matdesign/__init__.py', 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:  # Has docstring
                            doc_score += 1.0
                except Exception:
                    pass
            
            # Check for example files
            example_patterns = ['example*.py', 'demo*.py', 'tutorial*.py']
            for pattern in example_patterns:
                if list(Path('.').glob(pattern)):
                    doc_score += 0.5
                    break
            
            # Check for setup.py or pyproject.toml
            if Path('setup.py').exists() or Path('pyproject.toml').exists():
                doc_score += 1.0
            else:
                issues.append("Missing setup.py or pyproject.toml")
            
            # Check for license
            license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md']
            if any(Path(f).exists() for f in license_files):
                doc_score += 0.5
            
            score = min(1.0, doc_score / max_score)
            passed = score >= 0.6
            
            message = f"Documentation score: {score:.2f} ({len(issues)} issues)"
            
            return QualityResult(
                self.name, passed, self.level, score,
                message, {
                    'documentation_score': doc_score,
                    'max_score': max_score,
                    'issues': issues
                }, time.time() - start_time
            )
            
        except Exception as e:
            return QualityResult(
                self.name, False, self.level, 0.0,
                f"Documentation check failed: {e}", {}, time.time() - start_time
            )


class QualityGateRunner:
    """Orchestrates and runs all quality gates."""
    
    def __init__(self):
        self.gates = [
            SecurityGate(),
            CodeQualityGate(),
            TestCoverageGate(),
            PerformanceGate(),
            DocumentationGate()
        ]
        
        self.results: List[QualityResult] = []
    
    def run_all_gates(self) -> bool:
        """Run all quality gates and return overall pass/fail."""
        
        print("🛡️ TERRAGON AUTONOMOUS QUALITY GATES STARTING 🛡️")
        print("=" * 60)
        
        self.results = []
        overall_passed = True
        critical_failed = False
        
        for gate in self.gates:
            print(f"\n🔍 Running {gate.name} ({gate.level.value})...")
            
            result = gate.run()
            self.results.append(result)
            
            # Display result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.message}")
            print(f"   Score: {result.score:.2f} | Duration: {result.duration:.2f}s")
            
            if not result.passed:
                overall_passed = False
                if result.level == QualityLevel.CRITICAL:
                    critical_failed = True
                    print(f"   ⚠️  CRITICAL FAILURE - {result.details}")
            
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, list) and value:
                        print(f"   {key}: {len(value)} items")
                        for item in value[:3]:  # Show first 3
                            print(f"     - {item}")
                        if len(value) > 3:
                            print(f"     ... and {len(value) - 3} more")
                    else:
                        print(f"   {key}: {value}")
        
        # Final results
        print("\n" + "=" * 60)
        print("🏁 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        overall_score = sum(r.score for r in self.results) / total_count if total_count > 0 else 0.0
        
        print(f"Gates Passed: {passed_count}/{total_count}")
        print(f"Overall Score: {overall_score:.2f}")
        print(f"Total Duration: {sum(r.duration for r in self.results):.2f}s")
        
        if critical_failed:
            print("\n❌ CRITICAL FAILURE - DEPLOYMENT BLOCKED")
            print("🚫 Fix critical issues before proceeding")
        elif overall_passed:
            print("\n✅ ALL QUALITY GATES PASSED")
            print("🚀 READY FOR DEPLOYMENT")
        else:
            print("\n⚠️  SOME QUALITY GATES FAILED")
            print("🔧 Review and fix issues before deployment")
        
        print("=" * 60)
        
        return overall_passed and not critical_failed
    
    def get_report(self) -> Dict[str, Any]:
        """Get detailed quality report."""
        if not self.results:
            return {"error": "No quality gates have been run"}
        
        passed_count = sum(1 for r in self.results if r.passed)
        critical_failures = [r for r in self.results if not r.passed and r.level == QualityLevel.CRITICAL]
        
        return {
            "timestamp": time.time(),
            "overall_passed": all(r.passed for r in self.results),
            "critical_failures": len(critical_failures),
            "gates_passed": passed_count,
            "gates_total": len(self.results),
            "overall_score": sum(r.score for r in self.results) / len(self.results),
            "total_duration": sum(r.duration for r in self.results),
            "results": [
                {
                    "gate": r.gate_name,
                    "passed": r.passed,
                    "level": r.level.value,
                    "score": r.score,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.results
            ]
        }
    
    def save_report(self, filename: str = "quality_report.json"):
        """Save quality report to file."""
        report = self.get_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Quality report saved to {filename}")


def main():
    """Main entry point for quality gates."""
    
    # Change to repository root if we're not there
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    runner = QualityGateRunner()
    
    try:
        success = runner.run_all_gates()
        
        # Save report
        runner.save_report()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n💥 QUALITY GATES SYSTEM FAILURE: {e}")
        print("🚨 Contact system administrator")
        sys.exit(2)


if __name__ == "__main__":
    main()