"""Comprehensive Benchmarking Framework for Materials Science Diffusion Models.

This module implements a rigorous benchmarking and comparison framework for 
evaluating novel diffusion model approaches against established baselines.
Designed for research reproducibility and statistical significance validation.

Research Framework:
- Controlled experimental design with proper baselines
- Statistical significance testing  
- Reproducible experimental protocols
- Comprehensive performance metrics
- Academic publication-ready results
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve

from ..models.diffusion import DiffusionModel
from ..models.physics_informed import PhysicsInformedDiffusion
from ..models.bayesian_uncertainty import BayesianDiffusion
from ..models.hierarchical_multiscale import HierarchicalDiffusion


@dataclass
class ExperimentConfig:
    """Configuration for benchmark experiments."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    random_seed: int = 42
    
    # Dataset configuration
    dataset_name: str = "ti64_lpbf"
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 500
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # Evaluation configuration
    n_runs: int = 5
    confidence_level: float = 0.95
    statistical_test: str = "wilcoxon"
    
    # Model-specific configurations
    model_configs: Dict[str, Dict[str, Any]] = None


@dataclass 
class BenchmarkResults:
    """Results from benchmark experiments."""
    
    # Experiment metadata
    config: ExperimentConfig
    timestamp: str
    duration: float
    
    # Performance metrics
    metrics: Dict[str, Dict[str, float]]  # model_name -> metric -> value
    statistical_tests: Dict[str, Dict[str, float]]  # comparison -> test -> p_value
    
    # Detailed results
    predictions: Dict[str, np.ndarray]  # model_name -> predictions
    ground_truth: np.ndarray
    uncertainties: Dict[str, np.ndarray]  # model_name -> uncertainties
    
    # Timing information
    training_times: Dict[str, float]
    inference_times: Dict[str, float]
    
    # Physics consistency
    physics_metrics: Dict[str, Dict[str, float]]


class BenchmarkSuite:
    """Comprehensive benchmarking suite for diffusion models."""
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        device: str = "auto"
    ):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory for saving results
            device: Compute device ('auto', 'cpu', 'cuda')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model registry
        self.model_registry = {
            'baseline_diffusion': DiffusionModel,
            'physics_informed': PhysicsInformedDiffusion,
            'bayesian_uncertainty': BayesianDiffusion,
            'hierarchical_multiscale': HierarchicalDiffusion
        }
        
        # Metrics registry
        self.metrics_registry = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score,
            'mape': self._mean_absolute_percentage_error,
            'physics_consistency': self._physics_consistency_score
        }
        
    def run_comprehensive_benchmark(
        self,
        config: ExperimentConfig,
        dataset_generator: Callable,
        models_to_test: Optional[List[str]] = None
    ) -> BenchmarkResults:
        """Run comprehensive benchmark experiment.
        
        Args:
            config: Experiment configuration
            dataset_generator: Function to generate datasets
            models_to_test: List of model names to test (None = all)
            
        Returns:
            Comprehensive benchmark results
        """
        start_time = time.time()
        
        if models_to_test is None:
            models_to_test = list(self.model_registry.keys())
        
        print(f"🧪 Running comprehensive benchmark: {config.experiment_name}")
        print(f"📊 Models to test: {models_to_test}")
        print(f"🎯 Random seed: {config.random_seed}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Generate datasets
        print("📦 Generating datasets...")
        datasets = dataset_generator(config)
        
        # Initialize results storage
        results = {
            'metrics': {},
            'predictions': {},
            'uncertainties': {},
            'training_times': {},
            'inference_times': {},
            'physics_metrics': {}
        }
        
        # Run experiments for each model
        for model_name in models_to_test:
            print(f"\n🚀 Testing {model_name}...")
            
            model_results = self._run_model_experiment(
                model_name, config, datasets
            )
            
            # Store results
            for key in results:
                results[key][model_name] = model_results[key]
        
        # Compute statistical comparisons
        print("\n📈 Computing statistical comparisons...")
        statistical_tests = self._compute_statistical_tests(
            results['predictions'], datasets['test']['targets'], config
        )
        
        # Create comprehensive results
        benchmark_results = BenchmarkResults(
            config=config,
            timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
            duration=time.time() - start_time,
            metrics=results['metrics'],
            statistical_tests=statistical_tests,
            predictions=results['predictions'],
            ground_truth=datasets['test']['targets'],
            uncertainties=results['uncertainties'],
            training_times=results['training_times'],
            inference_times=results['inference_times'],
            physics_metrics=results['physics_metrics']
        )
        
        # Save results
        self._save_results(benchmark_results)
        
        print(f"\n✅ Benchmark completed in {benchmark_results.duration:.2f}s")
        print(f"📁 Results saved to {self.output_dir}")
        
        return benchmark_results
    
    def _run_model_experiment(
        self,
        model_name: str,
        config: ExperimentConfig,
        datasets: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Run experiment for a single model."""
        
        model_results = {
            'metrics': {},
            'predictions': None,
            'uncertainties': None,
            'training_time': 0.0,
            'inference_time': 0.0,
            'physics_metrics': {}
        }
        
        # Run multiple trials for statistical significance
        all_predictions = []
        all_metrics = []
        training_times = []
        inference_times = []
        
        for run in range(config.n_runs):
            print(f"  Run {run + 1}/{config.n_runs}")
            
            # Set seed for this run
            run_seed = config.random_seed + run * 1000
            self._set_random_seeds(run_seed)
            
            # Initialize model
            model = self._initialize_model(model_name, config)
            
            # Training
            train_start = time.time()
            trained_model = self._train_model(
                model, datasets['train'], datasets['val'], config
            )
            training_time = time.time() - train_start
            training_times.append(training_time)
            
            # Inference
            inference_start = time.time()
            predictions, uncertainties = self._evaluate_model(
                trained_model, datasets['test'], model_name
            )
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            all_predictions.append(predictions)
            
            # Compute metrics for this run
            run_metrics = self._compute_metrics(
                predictions, datasets['test']['targets']
            )
            all_metrics.append(run_metrics)
        
        # Aggregate results across runs
        model_results['predictions'] = np.mean(all_predictions, axis=0)
        model_results['training_time'] = np.mean(training_times)
        model_results['inference_time'] = np.mean(inference_times)
        
        # Aggregate metrics with confidence intervals
        for metric_name in all_metrics[0].keys():
            metric_values = [m[metric_name] for m in all_metrics]
            model_results['metrics'][metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'ci_lower': np.percentile(metric_values, 2.5),
                'ci_upper': np.percentile(metric_values, 97.5)
            }
        
        # Physics consistency evaluation
        if hasattr(trained_model, 'evaluate_physics_consistency'):
            physics_metrics = trained_model.evaluate_physics_consistency(
                torch.tensor(model_results['predictions'])
            )
            model_results['physics_metrics'] = physics_metrics
        
        return model_results
    
    def _initialize_model(self, model_name: str, config: ExperimentConfig) -> nn.Module:
        """Initialize model with configuration."""
        
        model_class = self.model_registry[model_name]
        
        # Get model-specific config
        model_config = {}
        if config.model_configs and model_name in config.model_configs:
            model_config = config.model_configs[model_name]
        
        # Default configurations for each model
        default_configs = {
            'baseline_diffusion': {'input_dim': 256, 'hidden_dim': 512, 'num_steps': 1000},
            'physics_informed': {'input_dim': 256, 'hidden_dim': 512, 'physics_weight': 0.1},
            'bayesian_uncertainty': {'input_dim': 256, 'hidden_dim': 512, 'n_mc_samples': 10},
            'hierarchical_multiscale': {'input_dim': 256, 'scales': [32, 64, 128, 256]}
        }
        
        final_config = {**default_configs.get(model_name, {}), **model_config}
        
        model = model_class(**final_config).to(self.device)
        return model
    
    def _train_model(
        self,
        model: nn.Module,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        config: ExperimentConfig
    ) -> nn.Module:
        """Train model with given configuration."""
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        train_inputs = torch.tensor(train_data['inputs'], dtype=torch.float32).to(self.device)
        train_targets = torch.tensor(train_data['targets'], dtype=torch.float32).to(self.device)
        
        # Training loop
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            
            # Forward pass (simplified - would need model-specific logic)
            if hasattr(model, 'forward_with_physics'):
                # Physics-informed model
                outputs, _ = model.forward_with_physics(train_inputs, torch.zeros_like(train_targets[:, 0]))
                loss = nn.MSELoss()(outputs, train_targets)
            elif hasattr(model, 'predict_with_uncertainty'):
                # Bayesian model
                outputs, _ = model.predict_with_uncertainty(train_inputs)
                loss = nn.MSELoss()(outputs, train_targets)
            else:
                # Standard model
                outputs = model(train_inputs)
                loss = nn.MSELoss()(outputs, train_targets)
            
            loss.backward()
            optimizer.step()
            
            # Early stopping could be implemented here
            
        return model
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_data: Dict[str, np.ndarray],
        model_name: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate trained model."""
        
        model.eval()
        
        test_inputs = torch.tensor(test_data['inputs'], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            if hasattr(model, 'predict_with_uncertainty'):
                # Bayesian model with uncertainty
                predictions, uncertainty_dict = model.predict_with_uncertainty(test_inputs)
                uncertainties = uncertainty_dict['total_std'].cpu().numpy()
            else:
                # Standard prediction
                predictions = model(test_inputs)
                uncertainties = None
        
        predictions = predictions.cpu().numpy()
        
        return predictions, uncertainties
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        
        metrics = {}
        
        for metric_name, metric_func in self.metrics_registry.items():
            try:
                if metric_name == 'physics_consistency':
                    # Special handling for physics metrics
                    metrics[metric_name] = metric_func(predictions)
                else:
                    metrics[metric_name] = metric_func(targets, predictions)
            except Exception as e:
                warnings.warn(f"Failed to compute {metric_name}: {e}")
                metrics[metric_name] = np.nan
        
        return metrics
    
    def _compute_statistical_tests(
        self,
        predictions_dict: Dict[str, np.ndarray],
        targets: np.ndarray,
        config: ExperimentConfig
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical significance tests."""
        
        statistical_tests = {}
        model_names = list(predictions_dict.keys())
        
        # Compute pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                # Compute errors for both models
                errors1 = np.abs(predictions_dict[model1] - targets)
                errors2 = np.abs(predictions_dict[model2] - targets)
                
                comparison_key = f"{model1}_vs_{model2}"
                statistical_tests[comparison_key] = {}
                
                # Wilcoxon signed-rank test
                try:
                    statistic, p_value = stats.wilcoxon(
                        errors1.flatten(), errors2.flatten()
                    )
                    statistical_tests[comparison_key]['wilcoxon_p'] = p_value
                    statistical_tests[comparison_key]['wilcoxon_statistic'] = statistic
                except Exception as e:
                    warnings.warn(f"Wilcoxon test failed for {comparison_key}: {e}")
                
                # Paired t-test
                try:
                    statistic, p_value = stats.ttest_rel(
                        errors1.flatten(), errors2.flatten()
                    )
                    statistical_tests[comparison_key]['ttest_p'] = p_value
                    statistical_tests[comparison_key]['ttest_statistic'] = statistic
                except Exception as e:
                    warnings.warn(f"T-test failed for {comparison_key}: {e}")
                
                # Effect size (Cohen's d)
                try:
                    pooled_std = np.sqrt(
                        (np.var(errors1) + np.var(errors2)) / 2
                    )
                    cohens_d = (np.mean(errors1) - np.mean(errors2)) / pooled_std
                    statistical_tests[comparison_key]['cohens_d'] = cohens_d
                except Exception as e:
                    warnings.warn(f"Cohen's d failed for {comparison_key}: {e}")
        
        return statistical_tests
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MAPE metric."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _physics_consistency_score(self, predictions: np.ndarray) -> float:
        """Compute physics consistency score for process parameters."""
        
        # Extract process parameters (simplified)
        laser_power = predictions[:, 0] if predictions.shape[1] > 0 else np.array([200])
        scan_speed = predictions[:, 1] if predictions.shape[1] > 1 else np.array([800])
        layer_thickness = predictions[:, 2] if predictions.shape[1] > 2 else np.array([30])
        hatch_spacing = predictions[:, 3] if predictions.shape[1] > 3 else np.array([120])
        
        # Calculate energy density
        energy_density = laser_power / (scan_speed * hatch_spacing * layer_thickness * 1e-6)
        
        # Physics consistency: fraction within reasonable bounds (40-120 J/mm³)
        in_bounds = np.logical_and(energy_density >= 40, energy_density <= 120)
        
        return np.mean(in_bounds.astype(float))
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _save_results(self, results: BenchmarkResults):
        """Save benchmark results."""
        
        timestamp = results.timestamp
        
        # Create experiment directory
        exp_dir = self.output_dir / f"{results.config.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(asdict(results.config), f, indent=2)
        
        # Save metrics
        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(results.metrics, f, indent=2)
        
        # Save statistical tests
        with open(exp_dir / "statistical_tests.json", 'w') as f:
            json.dump(results.statistical_tests, f, indent=2)
        
        # Save predictions and targets
        np.save(exp_dir / "ground_truth.npy", results.ground_truth)
        for model_name, predictions in results.predictions.items():
            np.save(exp_dir / f"predictions_{model_name}.npy", predictions)
        
        # Save uncertainties if available
        for model_name, uncertainties in results.uncertainties.items():
            if uncertainties is not None:
                np.save(exp_dir / f"uncertainties_{model_name}.npy", uncertainties)
        
        # Generate and save plots
        self._generate_benchmark_plots(results, exp_dir)
        
        print(f"📊 Results saved to {exp_dir}")
    
    def _generate_benchmark_plots(self, results: BenchmarkResults, output_dir: Path):
        """Generate comprehensive benchmark plots."""
        
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Benchmark Results: {results.config.experiment_name}', fontsize=16)
        
        # 1. MAE Comparison
        models = list(results.metrics.keys())
        mae_means = [results.metrics[m]['mae']['mean'] for m in models]
        mae_stds = [results.metrics[m]['mae']['std'] for m in models]
        
        axes[0, 0].bar(models, mae_means, yerr=mae_stds, capsize=5)
        axes[0, 0].set_title('Mean Absolute Error')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. R² Comparison  
        r2_means = [results.metrics[m]['r2']['mean'] for m in models]
        r2_stds = [results.metrics[m]['r2']['std'] for m in models]
        
        axes[0, 1].bar(models, r2_means, yerr=r2_stds, capsize=5)
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Time Comparison
        train_times = [results.training_times[m] for m in models]
        
        axes[1, 0].bar(models, train_times)
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Inference Time Comparison
        infer_times = [results.inference_times[m] for m in models]
        
        axes[1, 1].bar(models, infer_times)
        axes[1, 1].set_title('Inference Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction scatter plots
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, model in enumerate(models):
            pred = results.predictions[model]
            truth = results.ground_truth
            
            axes[i].scatter(truth.flatten(), pred.flatten(), alpha=0.6)
            axes[i].plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'r--')
            axes[i].set_xlabel('Ground Truth')
            axes[i].set_ylabel('Predictions')
            axes[i].set_title(f'{model}')
            
            # Add R² to plot
            r2 = results.metrics[model]['r2']['mean']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to {output_dir}")


def create_synthetic_dataset(config: ExperimentConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Create synthetic dataset for benchmarking."""
    
    np.random.seed(config.random_seed)
    
    def generate_data(size: int) -> Dict[str, np.ndarray]:
        # Synthetic microstructure features (simplified)
        inputs = np.random.randn(size, 256)
        
        # Synthetic process parameters with physics-based relationships
        laser_power = 150 + 100 * np.random.rand(size) + 0.1 * inputs[:, 0]
        scan_speed = 600 + 400 * np.random.rand(size) + 0.1 * inputs[:, 1] 
        layer_thickness = 20 + 20 * np.random.rand(size) + 0.05 * inputs[:, 2]
        hatch_spacing = 80 + 80 * np.random.rand(size) + 0.05 * inputs[:, 3]
        
        targets = np.column_stack([laser_power, scan_speed, layer_thickness, hatch_spacing])
        
        return {'inputs': inputs, 'targets': targets}
    
    return {
        'train': generate_data(config.train_size),
        'val': generate_data(config.val_size), 
        'test': generate_data(config.test_size)
    }