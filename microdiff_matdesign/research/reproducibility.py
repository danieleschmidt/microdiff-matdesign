"""Reproducibility Framework for Materials Science Research.

This module implements comprehensive reproducibility tools for ensuring
research results can be reliably replicated across different environments,
hardware configurations, and research groups.

Key Features:
- Deterministic random number generation
- Hardware-independent numerical stability
- Complete experimental logging and tracking
- Cross-platform compatibility validation
- Research artifact preservation
"""

import os
import json
import hashlib
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from packaging import version


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility framework."""
    
    # Random seeds
    numpy_seed: int = 42
    torch_seed: int = 42
    python_seed: int = 42
    
    # Determinism settings
    deterministic_algorithms: bool = True
    benchmark_mode: bool = False
    
    # Hardware settings
    use_deterministic_cudnn: bool = True
    allow_tf32: bool = False
    
    # Environment tracking
    track_environment: bool = True
    track_git_state: bool = True
    track_dependencies: bool = True
    
    # Validation settings
    numerical_tolerance: float = 1e-6
    cross_run_validation: bool = True
    

@dataclass
class ReproducibilityReport:
    """Comprehensive reproducibility report."""
    
    timestamp: str
    experiment_id: str
    config: ReproducibilityConfig
    
    # Environment information
    system_info: Dict[str, str]
    hardware_info: Dict[str, Any]
    dependencies: Dict[str, str]
    git_info: Dict[str, str]
    
    # Reproducibility validation
    cross_run_results: Dict[str, Dict[str, float]]
    numerical_stability: Dict[str, bool]
    determinism_check: Dict[str, bool]
    
    # Research artifacts
    data_checksums: Dict[str, str]
    model_checksums: Dict[str, str]
    results_checksums: Dict[str, str]


class ReproducibilityManager:
    """Comprehensive reproducibility management system."""
    
    def __init__(
        self,
        config: Optional[ReproducibilityConfig] = None,
        output_dir: str = "./reproducibility"
    ):
        """Initialize reproducibility manager."""
        self.config = config or ReproducibilityConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = self._generate_experiment_id()
        self.artifacts_dir = self.output_dir / f"experiment_{self.experiment_id}"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize reproducibility
        self._setup_reproducible_environment()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def _setup_reproducible_environment(self):
        """Setup reproducible computing environment."""
        print(f"🔒 Setting up reproducible environment (ID: {self.experiment_id})")
        
        # Set random seeds
        self._set_random_seeds()
        
        # Configure deterministic algorithms
        if self.config.deterministic_algorithms:
            os.environ['PYTHONHASHSEED'] = str(self.config.python_seed)
            
            if torch.cuda.is_available():
                # CUDA determinism
                torch.backends.cudnn.deterministic = self.config.use_deterministic_cudnn
                torch.backends.cudnn.benchmark = self.config.benchmark_mode
                
                # Disable TF32 for reproducibility
                if not self.config.allow_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                
                # Use deterministic algorithms where available
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception as e:
                    warnings.warn(f"Could not enable all deterministic algorithms: {e}")
        
        print("✅ Reproducible environment configured")
    
    def _set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        # NumPy
        np.random.seed(self.config.numpy_seed)
        
        # PyTorch
        torch.manual_seed(self.config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.torch_seed)
            torch.cuda.manual_seed_all(self.config.torch_seed)
        
        # Python
        import random
        random.seed(self.config.python_seed)
        
        print(f"🌱 Random seeds set: numpy={self.config.numpy_seed}, torch={self.config.torch_seed}")
    
    def collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        print("📊 Collecting environment information...")
        
        env_info = {
            'system_info': self._get_system_info(),
            'hardware_info': self._get_hardware_info(),
            'dependencies': self._get_dependencies(),
            'git_info': self._get_git_info() if self.config.track_git_state else {}
        }
        
        # Save environment info
        with open(self.artifacts_dir / "environment.json", 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
        
        return env_info
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware = {
            'cpu_count': os.cpu_count(),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            hardware.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
            })
        
        return hardware
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get dependency versions."""
        deps = {}
        
        # Core dependencies
        core_packages = [
            'numpy', 'torch', 'torchvision', 'scipy', 'scikit-learn',
            'matplotlib', 'seaborn', 'pillow', 'h5py', 'tqdm'
        ]
        
        for package in core_packages:
            try:
                module = __import__(package)
                version_attr = getattr(module, '__version__', 'unknown')
                deps[package] = str(version_attr)
            except ImportError:
                deps[package] = 'not_installed'
        
        # Get pip freeze output
        try:
            result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
            if result.returncode == 0:
                freeze_output = result.stdout.strip()
                with open(self.artifacts_dir / "requirements_freeze.txt", 'w') as f:
                    f.write(freeze_output)
        except Exception as e:
            warnings.warn(f"Could not run pip freeze: {e}")
        
        return deps
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        git_info = {}
        
        try:
            # Get git hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get git branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['is_dirty'] = len(result.stdout.strip()) > 0
                git_info['status'] = result.stdout.strip()
            
            # Get git remote
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
                
        except Exception as e:
            warnings.warn(f"Could not get git information: {e}")
        
        return git_info
    
    def validate_reproducibility(
        self,
        experiment_func: callable,
        n_runs: int = 3,
        **experiment_kwargs
    ) -> Dict[str, Any]:
        """Validate reproducibility across multiple runs."""
        print(f"🔍 Validating reproducibility across {n_runs} runs...")
        
        results = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}")
            
            # Reset environment for each run
            self._set_random_seeds()
            
            # Run experiment
            try:
                result = experiment_func(**experiment_kwargs)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Run {run + 1} failed: {e}")
                results.append(None)
        
        # Analyze results
        validation_report = self._analyze_reproducibility(results)
        
        # Save validation report
        with open(self.artifacts_dir / "reproducibility_validation.json", 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        return validation_report
    
    def _analyze_reproducibility(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze reproducibility of results."""
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) < 2:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 2 successful runs for validation'
            }
        
        analysis = {
            'n_runs': len(results),
            'successful_runs': len(valid_results),
            'identical_results': True,
            'numerical_differences': {}
        }
        
        # Check for identical results
        reference = valid_results[0]
        
        for i, result in enumerate(valid_results[1:], 1):
            if isinstance(result, dict) and isinstance(reference, dict):
                # Dictionary comparison
                for key in reference.keys():
                    if key in result:
                        if isinstance(reference[key], (int, float)):
                            diff = abs(reference[key] - result[key])
                            if diff > self.config.numerical_tolerance:
                                analysis['identical_results'] = False
                                analysis['numerical_differences'][f'run_{i}_{key}'] = diff
                        elif isinstance(reference[key], np.ndarray):
                            diff = np.max(np.abs(reference[key] - result[key]))
                            if diff > self.config.numerical_tolerance:
                                analysis['identical_results'] = False
                                analysis['numerical_differences'][f'run_{i}_{key}'] = float(diff)
            
            elif isinstance(result, (int, float)) and isinstance(reference, (int, float)):
                # Scalar comparison
                diff = abs(reference - result)
                if diff > self.config.numerical_tolerance:
                    analysis['identical_results'] = False
                    analysis['numerical_differences'][f'run_{i}'] = diff
            
            elif isinstance(result, np.ndarray) and isinstance(reference, np.ndarray):
                # Array comparison
                if result.shape != reference.shape:
                    analysis['identical_results'] = False
                    analysis['numerical_differences'][f'run_{i}_shape'] = f"{reference.shape} vs {result.shape}"
                else:
                    diff = np.max(np.abs(reference - result))
                    if diff > self.config.numerical_tolerance:
                        analysis['identical_results'] = False
                        analysis['numerical_differences'][f'run_{i}_values'] = float(diff)
        
        # Overall status
        if analysis['identical_results']:
            analysis['status'] = 'reproducible'
            analysis['message'] = 'All runs produced identical results within tolerance'
        else:
            analysis['status'] = 'non_reproducible'
            analysis['message'] = 'Results differ between runs beyond tolerance'
        
        return analysis
    
    def compute_data_checksum(self, data: Union[np.ndarray, torch.Tensor, dict, str]) -> str:
        """Compute checksum for data reproducibility."""
        hasher = hashlib.sha256()
        
        if isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
        elif isinstance(data, torch.Tensor):
            hasher.update(data.detach().cpu().numpy().tobytes())
        elif isinstance(data, dict):
            hasher.update(json.dumps(data, sort_keys=True).encode())
        elif isinstance(data, str):
            if os.path.exists(data):
                # File checksum
                with open(data, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            else:
                # String checksum
                hasher.update(data.encode())
        else:
            hasher.update(str(data).encode())
        
        return hasher.hexdigest()
    
    def create_reproducibility_package(
        self,
        results: Any,
        data_files: Optional[List[str]] = None,
        model_files: Optional[List[str]] = None,
        additional_files: Optional[List[str]] = None
    ) -> str:
        """Create complete reproducibility package."""
        print("📦 Creating reproducibility package...")
        
        package_dir = self.artifacts_dir / "reproducibility_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment information
        env_info = self.collect_environment_info()
        
        # Data checksums
        data_checksums = {}
        if data_files:
            for data_file in data_files:
                if os.path.exists(data_file):
                    checksum = self.compute_data_checksum(data_file)
                    data_checksums[data_file] = checksum
                    # Copy data file
                    shutil.copy2(data_file, package_dir / os.path.basename(data_file))
        
        # Model checksums
        model_checksums = {}
        if model_files:
            for model_file in model_files:
                if os.path.exists(model_file):
                    checksum = self.compute_data_checksum(model_file)
                    model_checksums[model_file] = checksum
                    # Copy model file
                    shutil.copy2(model_file, package_dir / os.path.basename(model_file))
        
        # Results checksum
        results_checksum = self.compute_data_checksum(results)
        
        # Additional files
        if additional_files:
            for file_path in additional_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, package_dir / os.path.basename(file_path))
        
        # Create reproducibility report
        report = ReproducibilityReport(
            timestamp=datetime.now().isoformat(),
            experiment_id=self.experiment_id,
            config=self.config,
            system_info=env_info['system_info'],
            hardware_info=env_info['hardware_info'],
            dependencies=env_info['dependencies'],
            git_info=env_info['git_info'],
            cross_run_results={},  # Filled by validate_reproducibility
            numerical_stability={},  # Filled by specific tests
            determinism_check={},  # Filled by specific tests
            data_checksums=data_checksums,
            model_checksums=model_checksums,
            results_checksums={'results': results_checksum}
        )
        
        # Save report
        with open(package_dir / "reproducibility_report.json", 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Create README
        readme_content = self._generate_readme(report)
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Reproducibility package created: {package_dir}")
        return str(package_dir)
    
    def _generate_readme(self, report: ReproducibilityReport) -> str:
        """Generate README for reproducibility package."""
        return f"""# Reproducibility Package

## Experiment Information
- **Experiment ID**: {report.experiment_id}
- **Timestamp**: {report.timestamp}
- **Environment**: {report.system_info.get('platform', 'unknown')}

## System Requirements
- **Python**: {report.system_info.get('python_version', 'unknown')}
- **Platform**: {report.system_info.get('platform', 'unknown')}
- **CUDA Available**: {report.hardware_info.get('cuda_available', False)}

## Key Dependencies
{chr(10).join([f"- {pkg}: {ver}" for pkg, ver in report.dependencies.items()])}

## Files Included
{chr(10).join([f"- {name}: {checksum[:8]}..." for name, checksum in report.data_checksums.items()])}

## Reproducibility Configuration
- **Random Seeds**: numpy={report.config.numpy_seed}, torch={report.config.torch_seed}
- **Deterministic**: {report.config.deterministic_algorithms}
- **Numerical Tolerance**: {report.config.numerical_tolerance}

## Git Information
- **Commit**: {report.git_info.get('commit_hash', 'unknown')[:8]}...
- **Branch**: {report.git_info.get('branch', 'unknown')}
- **Clean Repository**: {not report.git_info.get('is_dirty', True)}

## Usage
1. Install dependencies from `requirements_freeze.txt`
2. Verify data checksums match included files
3. Run experiment with provided configuration
4. Compare results with included checksums

## Validation
Run reproducibility validation with:
```python
from microdiff_matdesign.research.reproducibility import ReproducibilityManager

manager = ReproducibilityManager()
validation = manager.validate_reproducibility(experiment_func, n_runs=3)
```
"""

    def cross_platform_validation(
        self,
        experiment_func: callable,
        platforms: Optional[List[str]] = None,
        **experiment_kwargs
    ) -> Dict[str, Any]:
        """Validate cross-platform reproducibility (placeholder for distributed testing)."""
        # This would typically involve running experiments on different platforms
        # For now, we document the current platform and provide framework
        
        current_platform = platform.platform()
        
        validation_report = {
            'current_platform': current_platform,
            'validation_status': 'single_platform',
            'message': 'Cross-platform validation requires distributed testing setup',
            'recommendations': [
                'Test on Linux, Windows, and macOS',
                'Test with different Python versions',
                'Test with different CUDA versions',
                'Validate numerical stability across platforms'
            ]
        }
        
        # Save platform-specific results
        platform_dir = self.artifacts_dir / f"platform_{current_platform.replace(' ', '_')}"
        platform_dir.mkdir(parents=True, exist_ok=True)
        
        # Run experiment on current platform
        result = experiment_func(**experiment_kwargs)
        
        # Save platform-specific result
        with open(platform_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return validation_report