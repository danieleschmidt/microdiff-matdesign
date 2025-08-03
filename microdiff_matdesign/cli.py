"""Command line interface for MicroDiff-MatDesign."""

import os
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings

import click
import numpy as np
import torch
import yaml

from .core import MicrostructureDiffusion, ProcessParameters, train_diffusion_model
from .imaging import MicroCTProcessor
from .services.parameter_generation import ParameterGenerationService
from .services.optimization import OptimizationService
from .services.analysis import MicrostructureAnalysisService
from .services.prediction import PropertyPredictionService
from .data.datasets import create_microstructure_dataset, create_parameter_dataset
from .data.models import ExperimentData
from .utils.helpers import setup_device, set_random_seed, save_config, load_config
from .utils.validation import validate_process_parameters, validate_microstructure_data


@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto', help='Compute device')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, config, device, verbose):
    """MicroDiff-MatDesign: AI-driven material optimization for additive manufacturing.
    
    This CLI provides tools for inverse design, parameter optimization, 
    microstructure analysis, and property prediction using diffusion models.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup device
    if device == 'auto':
        ctx.obj['device'] = setup_device(prefer_gpu=True)
    elif device == 'cuda':
        if torch.cuda.is_available():
            ctx.obj['device'] = torch.device('cuda')
        else:
            click.echo("CUDA not available, falling back to CPU", err=True)
            ctx.obj['device'] = torch.device('cpu')
    else:
        ctx.obj['device'] = torch.device('cpu')
    
    # Load configuration
    ctx.obj['config'] = {}
    if config:
        try:
            ctx.obj['config'] = load_config(config)
            if verbose:
                click.echo(f"Loaded configuration from {config}")
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
    
    ctx.obj['verbose'] = verbose
    
    # Set random seed for reproducibility
    set_random_seed(ctx.obj['config'].get('random_seed', 42))


@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--alloy', default='Ti-6Al-4V', help='Target alloy composition')
@click.option('--process', type=click.Choice(['laser_powder_bed_fusion', 'electron_beam_melting', 'directed_energy_deposition']), 
              default='laser_powder_bed_fusion', help='Manufacturing process')
@click.option('--samples', '-n', default=10, help='Number of parameter samples to generate')
@click.option('--guidance-scale', default=7.5, help='Guidance scale for diffusion sampling')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON format)')
@click.option('--uncertainty', is_flag=True, help='Enable uncertainty quantification')
@click.option('--visualize', is_flag=True, help='Generate visualization outputs')
@click.pass_context
def inverse_design(ctx, image_path, alloy, process, samples, guidance_scale, output, uncertainty, visualize):
    """Generate process parameters from microstructure image.
    
    Takes a microstructure image and generates optimal process parameters
    using the trained diffusion model.
    """
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    try:
        if verbose:
            click.echo(f"Loading microstructure from {image_path}...")
        
        # Initialize processor and load image
        processor = MicroCTProcessor(voxel_size=0.5, cache_enabled=True)
        
        if Path(image_path).is_dir():
            microstructure = processor.load_volume(image_path)
        else:
            microstructure = processor.load_image(image_path)
        
        if verbose:
            click.echo(f"Loaded microstructure with shape: {microstructure.shape}")
        
        # Initialize diffusion model
        model = MicrostructureDiffusion(
            alloy=alloy, 
            process=process,
            device=device
        )
        
        # Load pretrained weights if available
        model_path = ctx.obj['config'].get('model_path', f'models/{alloy}_{process}_diffusion.pt')
        if Path(model_path).exists():
            model.load_model(model_path)
            if verbose:
                click.echo(f"Loaded pretrained model from {model_path}")
        else:
            click.echo(f"Warning: No pretrained model found at {model_path}. Using random initialization.", err=True)
        
        # Perform inverse design
        if verbose:
            click.echo(f"Generating {samples} parameter samples...")
        
        if uncertainty:
            parameters, uncertainty_metrics = model.inverse_design(
                microstructure, 
                num_samples=samples,
                guidance_scale=guidance_scale,
                uncertainty_quantification=True
            )
        else:
            parameters = model.inverse_design(
                microstructure, 
                num_samples=samples,
                guidance_scale=guidance_scale
            )
            uncertainty_metrics = None
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("INVERSE DESIGN RESULTS")
        click.echo("="*50)
        click.echo(f"Target Alloy: {alloy}")
        click.echo(f"Process: {process}")
        click.echo(f"Samples Generated: {samples}")
        click.echo()
        
        # Show optimal parameters
        click.echo("Optimal Process Parameters:")
        click.echo(f"  Laser Power: {parameters.laser_power:.1f} W")
        click.echo(f"  Scan Speed: {parameters.scan_speed:.1f} mm/s")
        click.echo(f"  Layer Thickness: {parameters.layer_thickness:.1f} μm")
        click.echo(f"  Hatch Spacing: {parameters.hatch_spacing:.1f} μm")
        click.echo(f"  Powder Bed Temperature: {parameters.powder_bed_temp:.1f} °C")
        
        # Show uncertainty if available
        if uncertainty_metrics:
            click.echo("\nUncertainty Metrics:")
            for param, uncertainty_val in uncertainty_metrics.items():
                click.echo(f"  {param}: ±{uncertainty_val:.3f}")
        
        # Validate parameters
        validation_result = validate_process_parameters(parameters.to_dict(), process)
        if not validation_result['valid']:
            click.echo("\nValidation Warnings:")
            for warning in validation_result['warnings']:
                click.echo(f"  ⚠️  {warning}")
        
        # Save results if output path specified
        if output:
            results = {
                'input_image': str(image_path),
                'alloy': alloy,
                'process': process,
                'parameters': parameters.to_dict(),
                'uncertainty_metrics': uncertainty_metrics,
                'validation': validation_result,
                'microstructure_shape': microstructure.shape,
                'guidance_scale': guidance_scale,
                'samples_generated': samples
            }
            
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            
            if verbose:
                click.echo(f"\nResults saved to {output}")
        
        # Generate visualizations if requested
        if visualize:
            viz_dir = Path(output).parent / "visualizations" if output else Path("visualizations")
            viz_dir.mkdir(exist_ok=True)
            
            # Generate microstructure visualizations
            viz_data = processor.visualize_features(microstructure)
            
            import matplotlib.pyplot as plt
            for viz_name, viz_array in viz_data.items():
                plt.figure(figsize=(8, 6))
                plt.imshow(viz_array, cmap='gray')
                plt.title(f"{viz_name.title()} Visualization")
                plt.colorbar()
                plt.savefig(viz_dir / f"{viz_name}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            if verbose:
                click.echo(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        click.echo(f"Error during inverse design: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--alloy', default='Ti-6Al-4V', help='Target alloy for training')
@click.option('--process', type=click.Choice(['laser_powder_bed_fusion', 'electron_beam_melting', 'directed_energy_deposition']), 
              default='laser_powder_bed_fusion', help='Manufacturing process')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=8, help='Training batch size')
@click.option('--learning-rate', default=1e-4, help='Learning rate')
@click.option('--save-path', default='models', help='Directory to save trained model')
@click.option('--resume', type=click.Path(exists=True), help='Path to checkpoint to resume training')
@click.option('--validation-split', default=0.2, help='Fraction of data for validation')
@click.pass_context
def train(ctx, data_dir, alloy, process, epochs, batch_size, learning_rate, save_path, resume, validation_split):
    """Train diffusion model on microstructure data.
    
    Trains a diffusion model for inverse design using microstructure images
    and corresponding process parameters.
    """
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    try:
        if verbose:
            click.echo(f"Setting up training for {alloy} with {process}")
            click.echo(f"Data directory: {data_dir}")
            click.echo(f"Device: {device}")
        
        # Create datasets
        train_dataset, val_dataset, _ = create_microstructure_dataset(
            data_dir,
            train_ratio=1-validation_split,
            val_ratio=validation_split,
            test_ratio=0.0
        )
        
        if verbose:
            click.echo(f"Training samples: {len(train_dataset)}")
            click.echo(f"Validation samples: {len(val_dataset)}")
        
        # Initialize model
        model = MicrostructureDiffusion(
            alloy=alloy,
            process=process,
            device=device
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume:
            checkpoint_info = model.load_model(resume)
            start_epoch = checkpoint_info.get('epoch', 0)
            if verbose:
                click.echo(f"Resumed training from epoch {start_epoch}")
        
        # Create data loaders
        train_loader = train_dataset.get_data_loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = val_dataset.get_data_loader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Training configuration
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'device': str(device),
            'alloy': alloy,
            'process': process,
            'data_dir': str(data_dir),
            'save_path': save_path
        }
        
        # Save training configuration
        os.makedirs(save_path, exist_ok=True)
        config_path = Path(save_path) / f"{alloy}_{process}_config.yaml"
        save_config(config, str(config_path))
        
        if verbose:
            click.echo(f"Training configuration saved to {config_path}")
        
        # Start training
        click.echo(f"\nStarting training for {epochs} epochs...")
        
        train_diffusion_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            save_path=save_path,
            start_epoch=start_epoch,
            verbose=verbose
        )
        
        click.echo(f"\nTraining completed! Model saved to {save_path}")
        
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--features', multiple=True, default=['all'], help='Specific features to extract')
@click.option('--visualize', is_flag=True, help='Generate feature visualizations')
@click.pass_context
def analyze(ctx, image_path, output, features, visualize):
    """Analyze microstructure features and properties.
    
    Performs comprehensive analysis of microstructure including grain size,
    phase fractions, porosity, and texture analysis.
    """
    verbose = ctx.obj['verbose']
    
    try:
        if verbose:
            click.echo(f"Analyzing microstructure from {image_path}...")
        
        # Initialize processor and analysis service
        processor = MicroCTProcessor(voxel_size=0.5, cache_enabled=True)
        analysis_service = MicrostructureAnalysisService()
        
        # Load microstructure
        if Path(image_path).is_dir():
            microstructure = processor.load_volume(image_path)
        else:
            microstructure = processor.load_image(image_path)
        
        if verbose:
            click.echo(f"Loaded microstructure with shape: {microstructure.shape}")
        
        # Perform analysis
        if 'all' in features:
            feature_types = None  # Use default feature set
        else:
            feature_types = list(features)
        
        # Extract basic features
        basic_features = processor.extract_features(microstructure, feature_types=feature_types)
        
        # Perform comprehensive analysis
        analysis_results = analysis_service.analyze_microstructure(microstructure)
        
        # Porosity analysis
        porosity_results = processor.analyze_porosity(microstructure)
        
        # Combine all results
        complete_results = {
            'basic_features': basic_features,
            'comprehensive_analysis': analysis_results,
            'porosity_analysis': porosity_results,
            'microstructure_shape': microstructure.shape,
            'input_file': str(image_path)
        }
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("MICROSTRUCTURE ANALYSIS RESULTS")
        click.echo("="*50)
        
        # Porosity metrics
        click.echo("\nPorosity Analysis:")
        click.echo(f"  Total Porosity: {porosity_results['total_porosity']:.3f}")
        click.echo(f"  Pore Count: {porosity_results['pore_count']}")
        click.echo(f"  Mean Pore Volume: {porosity_results['mean_pore_volume']:.2e} μm³")
        
        # Quality assessment
        if 'quality_score' in analysis_results:
            click.echo(f"\nQuality Score: {analysis_results['quality_score']:.3f}/1.0")
        
        # Feature summary
        if basic_features:
            click.echo("\nKey Features:")
            for feature, value in list(basic_features.items())[:10]:  # Show first 10 features
                if isinstance(value, float):
                    click.echo(f"  {feature}: {value:.3f}")
                else:
                    click.echo(f"  {feature}: {value}")
        
        # Save results
        if output:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            if verbose:
                click.echo(f"\nAnalysis results saved to {output}")
        
        # Generate visualizations
        if visualize:
            viz_dir = Path(output).parent / "analysis_viz" if output else Path("analysis_viz")
            viz_dir.mkdir(exist_ok=True)
            
            viz_data = processor.visualize_features(microstructure)
            
            import matplotlib.pyplot as plt
            for viz_name, viz_array in viz_data.items():
                plt.figure(figsize=(10, 8))
                plt.imshow(viz_array, cmap='viridis')
                plt.title(f"{viz_name.title()} Analysis")
                plt.colorbar()
                plt.savefig(viz_dir / f"analysis_{viz_name}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            if verbose:
                click.echo(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--laser-power', type=float, help='Laser power (W)')
@click.option('--scan-speed', type=float, help='Scan speed (mm/s)')
@click.option('--layer-thickness', type=float, help='Layer thickness (μm)')
@click.option('--hatch-spacing', type=float, help='Hatch spacing (μm)')
@click.option('--powder-bed-temp', type=float, help='Powder bed temperature (°C)')
@click.option('--alloy', default='Ti-6Al-4V', help='Target alloy composition')
@click.option('--process', type=click.Choice(['laser_powder_bed_fusion', 'electron_beam_melting', 'directed_energy_deposition']), 
              default='laser_powder_bed_fusion', help='Manufacturing process')
@click.option('--output', '-o', type=click.Path(), help='Output file for predictions')
@click.pass_context
def predict(ctx, laser_power, scan_speed, layer_thickness, hatch_spacing, powder_bed_temp, alloy, process, output):
    """Predict material properties from process parameters.
    
    Uses trained models to predict mechanical properties, microstructure
    characteristics, and process outcomes.
    """
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    try:
        # Create process parameters
        if not all([laser_power, scan_speed, layer_thickness, hatch_spacing]):
            click.echo("Error: All process parameters must be specified", err=True)
            sys.exit(1)
        
        parameters = ProcessParameters(
            laser_power=laser_power,
            scan_speed=scan_speed,
            layer_thickness=layer_thickness,
            hatch_spacing=hatch_spacing,
            powder_bed_temp=powder_bed_temp or 80.0
        )
        
        if verbose:
            click.echo("Process Parameters:")
            click.echo(f"  Laser Power: {parameters.laser_power:.1f} W")
            click.echo(f"  Scan Speed: {parameters.scan_speed:.1f} mm/s")
            click.echo(f"  Layer Thickness: {parameters.layer_thickness:.1f} μm")
            click.echo(f"  Hatch Spacing: {parameters.hatch_spacing:.1f} μm")
            click.echo(f"  Powder Bed Temperature: {parameters.powder_bed_temp:.1f} °C")
        
        # Initialize prediction service
        prediction_service = PropertyPredictionService(device=device)
        
        # Predict properties
        predictions = prediction_service.predict_properties(parameters, alloy, process)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("PROPERTY PREDICTIONS")
        click.echo("="*50)
        
        # Mechanical properties
        if 'mechanical_properties' in predictions:
            mech_props = predictions['mechanical_properties']
            click.echo("\nMechanical Properties:")
            click.echo(f"  Tensile Strength: {mech_props.get('tensile_strength', 0):.1f} MPa")
            click.echo(f"  Yield Strength: {mech_props.get('yield_strength', 0):.1f} MPa")
            click.echo(f"  Elongation: {mech_props.get('elongation', 0):.1f} %")
            click.echo(f"  Hardness: {mech_props.get('hardness', 0):.1f} HV")
        
        # Microstructure predictions
        if 'microstructure_properties' in predictions:
            micro_props = predictions['microstructure_properties']
            click.echo("\nMicrostructure Properties:")
            click.echo(f"  Density: {micro_props.get('density', 0):.3f} g/cm³")
            click.echo(f"  Porosity: {micro_props.get('porosity', 0):.3f}")
            click.echo(f"  Surface Roughness: {micro_props.get('surface_roughness', 0):.2f} μm")
        
        # Process outcome predictions
        if 'process_outcomes' in predictions:
            process_outcomes = predictions['process_outcomes']
            click.echo("\nProcess Outcomes:")
            click.echo(f"  Build Success Rate: {process_outcomes.get('success_probability', 0):.1%}")
            click.echo(f"  Defect Probability: {process_outcomes.get('defect_probability', 0):.1%}")
        
        # Recommendations
        if 'recommendations' in predictions:
            recommendations = predictions['recommendations']
            click.echo("\nRecommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
        
        # Save predictions
        if output:
            results = {
                'input_parameters': parameters.to_dict(),
                'alloy': alloy,
                'process': process,
                'predictions': predictions
            }
            
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if verbose:
                click.echo(f"\nPredictions saved to {output}")
        
    except Exception as e:
        click.echo(f"Error during prediction: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--alloy', default='Ti-6Al-4V', help='Target alloy composition')
@click.option('--process', type=click.Choice(['laser_powder_bed_fusion', 'electron_beam_melting', 'directed_energy_deposition']), 
              default='laser_powder_bed_fusion', help='Manufacturing process')
@click.option('--target-property', multiple=True, help='Target properties to optimize for')
@click.option('--target-value', multiple=True, type=float, help='Target values for properties')
@click.option('--method', type=click.Choice(['genetic', 'particle_swarm', 'bayesian']), 
              default='genetic', help='Optimization algorithm')
@click.option('--generations', default=50, help='Number of optimization generations')
@click.option('--population-size', default=50, help='Population size for optimization')
@click.option('--output', '-o', type=click.Path(), help='Output file for optimal parameters')
@click.pass_context
def optimize(ctx, alloy, process, target_property, target_value, method, generations, population_size, output):
    """Optimize process parameters for target properties.
    
    Uses multi-objective optimization to find process parameters that
    achieve desired material properties.
    """
    device = ctx.obj['device']
    verbose = ctx.obj['verbose']
    
    try:
        # Validate inputs
        if len(target_property) != len(target_value):
            click.echo("Error: Number of target properties must match number of target values", err=True)
            sys.exit(1)
        
        if not target_property:
            click.echo("Error: At least one target property must be specified", err=True)
            sys.exit(1)
        
        # Create optimization objectives
        objectives = dict(zip(target_property, target_value))
        
        if verbose:
            click.echo(f"Optimization targets:")
            for prop, value in objectives.items():
                click.echo(f"  {prop}: {value}")
            click.echo(f"Method: {method}")
            click.echo(f"Generations: {generations}")
            click.echo(f"Population Size: {population_size}")
        
        # Initialize optimization service
        optimization_service = OptimizationService(device=device)
        
        # Define parameter constraints for the process
        constraints = optimization_service.get_default_constraints(process)
        
        # Run optimization
        click.echo(f"\nStarting {method} optimization...")
        
        results = optimization_service.optimize_parameters(
            objectives=objectives,
            alloy=alloy,
            process=process,
            method=method,
            max_generations=generations,
            population_size=population_size,
            constraints=constraints
        )
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("OPTIMIZATION RESULTS")
        click.echo("="*50)
        
        optimal_params = results['optimal_parameters']
        click.echo("\nOptimal Process Parameters:")
        click.echo(f"  Laser Power: {optimal_params.laser_power:.1f} W")
        click.echo(f"  Scan Speed: {optimal_params.scan_speed:.1f} mm/s")
        click.echo(f"  Layer Thickness: {optimal_params.layer_thickness:.1f} μm")
        click.echo(f"  Hatch Spacing: {optimal_params.hatch_spacing:.1f} μm")
        click.echo(f"  Powder Bed Temperature: {optimal_params.powder_bed_temp:.1f} °C")
        
        # Show predicted properties
        if 'predicted_properties' in results:
            predicted = results['predicted_properties']
            click.echo("\nPredicted Properties:")
            for prop, value in predicted.items():
                target_val = objectives.get(prop, "N/A")
                click.echo(f"  {prop}: {value:.3f} (target: {target_val})")
        
        # Show optimization metrics
        if 'optimization_metrics' in results:
            metrics = results['optimization_metrics']
            click.echo(f"\nOptimization Metrics:")
            click.echo(f"  Final Fitness: {metrics.get('final_fitness', 0):.3f}")
            click.echo(f"  Convergence Generation: {metrics.get('convergence_generation', generations)}")
            click.echo(f"  Total Evaluations: {metrics.get('total_evaluations', 0)}")
        
        # Save results
        if output:
            optimization_results = {
                'objectives': objectives,
                'alloy': alloy,
                'process': process,
                'method': method,
                'optimal_parameters': optimal_params.to_dict(),
                'optimization_results': results
            }
            
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)
            
            if verbose:
                click.echo(f"\nOptimization results saved to {output}")
        
    except Exception as e:
        click.echo(f"Error during optimization: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path())
@click.option('--dry-run', is_flag=True, help='Show what would be created without creating it')
@click.pass_context
def init_project(ctx, config_file, dry_run):
    """Initialize a new MicroDiff-MatDesign project.
    
    Creates project structure and configuration files for a new material
    design project.
    """
    verbose = ctx.obj['verbose']
    
    try:
        # Default project structure
        project_structure = {
            'data': {
                'raw': {},
                'processed': {},
                'experiments': {}
            },
            'models': {},
            'results': {
                'inverse_design': {},
                'optimization': {},
                'analysis': {},
                'visualizations': {}
            },
            'configs': {}
        }
        
        # Default configuration
        default_config = {
            'project_name': 'microdiff_project',
            'version': '1.0.0',
            'alloys': ['Ti-6Al-4V', 'AlSi10Mg', 'Inconel 718'],
            'processes': ['laser_powder_bed_fusion', 'electron_beam_melting'],
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'epochs': 100,
                'validation_split': 0.2
            },
            'optimization': {
                'method': 'genetic',
                'generations': 50,
                'population_size': 50
            },
            'device': 'auto',
            'random_seed': 42
        }
        
        if dry_run:
            click.echo("Project structure that would be created:")
            click.echo(yaml.dump(project_structure, default_flow_style=False))
            click.echo("\nDefault configuration:")
            click.echo(yaml.dump(default_config, default_flow_style=False))
        else:
            # Create project directory
            project_path = Path(config_file).parent
            project_path.mkdir(exist_ok=True)
            
            # Create directory structure
            from .utils.helpers import create_directory_structure
            create_directory_structure(str(project_path), project_structure)
            
            # Save configuration
            save_config(default_config, config_file)
            
            # Create README
            readme_content = f"""# MicroDiff-MatDesign Project
            
## Overview
This project uses AI-driven diffusion models for inverse material design
and process parameter optimization in additive manufacturing.

## Usage
- `microdiff inverse-design`: Generate process parameters from microstructures
- `microdiff train`: Train diffusion models on your data
- `microdiff analyze`: Analyze microstructure features
- `microdiff predict`: Predict properties from parameters
- `microdiff optimize`: Optimize parameters for target properties

## Configuration
Edit `{config_file}` to customize project settings.

## Data Structure
- `data/raw/`: Raw microstructure images and experimental data
- `data/processed/`: Preprocessed datasets
- `data/experiments/`: Experiment metadata and results
- `models/`: Trained model checkpoints
- `results/`: Analysis and optimization results
"""
            
            readme_path = project_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            if verbose:
                click.echo(f"Project initialized at {project_path}")
                click.echo(f"Configuration saved to {config_file}")
                click.echo(f"Edit the configuration file and add your data to get started!")
        
    except Exception as e:
        click.echo(f"Error initializing project: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.pass_context
def info(ctx):
    """Display system and environment information."""
    verbose = ctx.obj['verbose']
    device = ctx.obj['device']
    
    click.echo("MicroDiff-MatDesign System Information")
    click.echo("="*40)
    
    # System info
    click.echo(f"Device: {device}")
    click.echo(f"PyTorch Version: {torch.__version__}")
    click.echo(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        click.echo(f"CUDA Version: {torch.version.cuda}")
        click.echo(f"GPU Count: {torch.cuda.device_count()}")
        click.echo(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    try:
        from .utils.helpers import get_memory_usage
        memory_info = get_memory_usage()
        
        click.echo(f"\nMemory Usage:")
        click.echo(f"  System Memory: {memory_info['system_memory_used_gb']:.1f}GB / {memory_info['system_memory_total_gb']:.1f}GB")
        
        if 'gpu_memory_allocated_gb' in memory_info:
            click.echo(f"  GPU Memory: {memory_info['gpu_memory_allocated_gb']:.1f}GB allocated, {memory_info['gpu_memory_reserved_gb']:.1f}GB reserved")
            
    except Exception as e:
        if verbose:
            click.echo(f"Could not get memory info: {e}")
    
    # Configuration info
    config = ctx.obj.get('config', {})
    if config:
        click.echo(f"\nLoaded Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                click.echo(f"  {key}: {len(value)} items")
            else:
                click.echo(f"  {key}: {value}")


if __name__ == '__main__':
    main()