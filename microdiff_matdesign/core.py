"""Core diffusion model functionality."""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from .models.diffusion import DiffusionModel
from .models.encoders import MicrostructureEncoder
from .models.decoders import ParameterDecoder
from .utils.validation import validate_microstructure, validate_parameters
from .utils.preprocessing import normalize_microstructure, denormalize_parameters


class ProcessParameters:
    """Container for manufacturing process parameters."""
    
    def __init__(self, **kwargs):
        self.laser_power = kwargs.get('laser_power', 200.0)
        self.scan_speed = kwargs.get('scan_speed', 800.0) 
        self.layer_thickness = kwargs.get('layer_thickness', 30.0)
        self.hatch_spacing = kwargs.get('hatch_spacing', 120.0)
        self.powder_bed_temp = kwargs.get('powder_bed_temp', 80.0)
        self.atmosphere = kwargs.get('atmosphere', 'argon')
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'laser_power': self.laser_power,
            'scan_speed': self.scan_speed,
            'layer_thickness': self.layer_thickness,
            'hatch_spacing': self.hatch_spacing,
            'powder_bed_temp': self.powder_bed_temp,
            'atmosphere': self.atmosphere
        }
        
    def to_tensor(self) -> torch.Tensor:
        """Convert parameters to normalized tensor for model input."""
        values = [self.laser_power, self.scan_speed, self.layer_thickness, 
                 self.hatch_spacing, self.powder_bed_temp]
        normalized = [(v - mean) / std for v, mean, std in zip(values, 
                     [200, 800, 30, 120, 80], [50, 200, 10, 30, 20])]
        return torch.tensor(normalized, dtype=torch.float32)


class MicrostructureDiffusion:
    """Main diffusion model for microstructure inverse design."""
    
    def __init__(
        self,
        alloy: str = "Ti-6Al-4V",
        process: str = "laser_powder_bed_fusion",
        pretrained: bool = True,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize the diffusion model."""
        self.alloy = alloy
        self.process = process
        self.pretrained = pretrained
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Model architecture parameters
        self.config = self._load_config()
        
        # Initialize model components
        self.encoder = MicrostructureEncoder(
            input_dim=self.config['encoder']['input_dim'],
            hidden_dim=self.config['encoder']['hidden_dim'],
            latent_dim=self.config['encoder']['latent_dim']
        ).to(self.device)
        
        self.diffusion_model = DiffusionModel(
            input_dim=self.config['diffusion']['input_dim'],
            hidden_dim=self.config['diffusion']['hidden_dim'],
            num_steps=self.config['diffusion']['num_steps']
        ).to(self.device)
        
        self.decoder = ParameterDecoder(
            latent_dim=self.config['decoder']['latent_dim'],
            hidden_dim=self.config['decoder']['hidden_dim'],
            output_dim=self.config['decoder']['output_dim']
        ).to(self.device)
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(model_path)
            
        # Set to evaluation mode by default
        self.eval()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = Path(__file__).parent / 'configs' / f'{self.alloy.lower().replace("-", "_")}_{self.process}.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration - simplified for Generation 1
            return {
                'encoder': {'input_dim': 64*64*64, 'hidden_dim': 256, 'latent_dim': 128},
                'diffusion': {'input_dim': 128, 'hidden_dim': 256, 'num_steps': 10},  # Very simple for testing
                'decoder': {'latent_dim': 128, 'hidden_dim': 256, 'output_dim': 5}
            }
    
    def _load_pretrained_weights(self, model_path: Optional[str] = None):
        """Load pretrained model weights."""
        if model_path is None:
            model_path = Path(__file__).parent / 'pretrained' / f'{self.alloy}_{self.process}.pth'
            
        if Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.diffusion_model.load_state_dict(checkpoint['diffusion'])
                self.decoder.load_state_dict(checkpoint['decoder'])
                print(f"Loaded pretrained weights from {model_path}")
            except Exception as e:
                warnings.warn(f"Failed to load pretrained weights: {e}")
        else:
            warnings.warn(f"Pretrained model not found at {model_path}. Using random initialization.")
    
    def train(self):
        """Set model to training mode."""
        self.encoder.train()
        self.diffusion_model.train()
        self.decoder.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.encoder.eval()
        self.diffusion_model.eval()
        self.decoder.eval()
        
    def inverse_design(
        self,
        target_microstructure: np.ndarray,
        num_samples: int = 10,
        guidance_scale: float = 7.5,
        uncertainty_quantification: bool = False
    ) -> Union[ProcessParameters, Tuple[ProcessParameters, Dict[str, float]]]:
        """Generate process parameters from target microstructure."""
        
        # Validate input
        validate_microstructure(target_microstructure)
        
        # Preprocess microstructure
        microstructure_tensor = normalize_microstructure(target_microstructure)
        microstructure_tensor = torch.from_numpy(microstructure_tensor).float().to(self.device)
        
        if len(microstructure_tensor.shape) == 3:
            microstructure_tensor = microstructure_tensor.unsqueeze(0)  # Add batch dimension
            
        with torch.no_grad():
            # Encode microstructure to latent space
            latent_encoding = self.encoder(microstructure_tensor.flatten(1))
            
            # Generate multiple samples for uncertainty quantification
            parameter_samples = []
            
            for _ in range(num_samples):
                # Sample noise and condition on microstructure encoding
                noise = torch.randn(latent_encoding.shape, device=self.device)
                
                # Iterative denoising with guidance
                denoised_latent = self.diffusion_model.sample(
                    noise, 
                    condition=latent_encoding,
                    guidance_scale=guidance_scale
                )
                
                # Decode to process parameters
                parameter_tensor = self.decoder(denoised_latent)
                parameter_samples.append(parameter_tensor.cpu().numpy())
                
        # Convert to ProcessParameters objects
        parameter_samples = np.array(parameter_samples)
        
        # Denormalize parameters
        denormalized_params = denormalize_parameters(parameter_samples)
        
        # Calculate statistics
        mean_params = np.mean(denormalized_params, axis=0)
        
        # Create ProcessParameters object with mean values
        # Handle both single sample and batch cases
        if mean_params.ndim > 1:
            mean_params = mean_params[0]  # Take first sample if batch
            
        # Clip parameters to valid ranges for untrained model
        clipped_params = np.clip(mean_params, 
                               [100.0, 400.0, 20.0, 80.0, 50.0],  # mins
                               [400.0, 1500.0, 80.0, 250.0, 150.0])  # maxs
        
        result_params = ProcessParameters(
            laser_power=float(clipped_params[0]) if len(clipped_params) > 0 else 200.0,
            scan_speed=float(clipped_params[1]) if len(clipped_params) > 1 else 800.0,
            layer_thickness=float(clipped_params[2]) if len(clipped_params) > 2 else 30.0,
            hatch_spacing=float(clipped_params[3]) if len(clipped_params) > 3 else 120.0,
            powder_bed_temp=float(clipped_params[4]) if len(clipped_params) > 4 else 80.0
        )
        
        validate_parameters(result_params.to_dict(), self.process)
        
        if uncertainty_quantification:
            # Calculate uncertainty metrics
            std_params = np.std(denormalized_params, axis=0)
            uncertainty = {
                'laser_power_std': float(std_params[0]),
                'scan_speed_std': float(std_params[1]),
                'layer_thickness_std': float(std_params[2]),
                'hatch_spacing_std': float(std_params[3]),
                'confidence_interval': 0.95
            }
            return result_params, uncertainty
            
        return result_params
    
    def predict_microstructure(
        self,
        parameters: ProcessParameters,
        num_samples: int = 1
    ) -> np.ndarray:
        """Forward prediction: generate microstructure from parameters."""
        
        parameter_tensor = parameters.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use decoder in reverse mode (this is a simplified approach)
            latent = self.decoder.inverse(parameter_tensor)
            
            # Generate microstructure samples
            microstructures = []
            for _ in range(num_samples):
                noise = torch.randn_like(latent)
                conditioned_latent = latent + 0.1 * noise  # Add controlled noise
                
                # Decode to microstructure
                microstructure = self.encoder.decode(conditioned_latent)
                microstructures.append(microstructure.cpu().numpy())
                
        if num_samples == 1:
            return microstructures[0].reshape(128, 128, 128)
        else:
            return np.array(microstructures).reshape(num_samples, 128, 128, 128)
    
    def optimize_parameters(
        self,
        target_properties: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        num_iterations: int = 100
    ) -> ProcessParameters:
        """Optimize parameters for target material properties."""
        
        # Initialize with random parameters
        best_params = ProcessParameters()
        best_score = float('inf')
        
        for _ in range(num_iterations):
            # Generate random perturbation
            perturbation = {
                'laser_power': np.random.normal(0, 10),
                'scan_speed': np.random.normal(0, 50),
                'layer_thickness': np.random.normal(0, 2),
                'hatch_spacing': np.random.normal(0, 10)
            }
            
            # Apply perturbation
            trial_params = ProcessParameters(
                laser_power=best_params.laser_power + perturbation['laser_power'],
                scan_speed=best_params.scan_speed + perturbation['scan_speed'],
                layer_thickness=best_params.layer_thickness + perturbation['layer_thickness'],
                hatch_spacing=best_params.hatch_spacing + perturbation['hatch_spacing']
            )
            
            # Apply constraints
            if constraints:
                trial_params = self._apply_constraints(trial_params, constraints)
            
            # Evaluate parameters (simplified scoring)
            score = self._evaluate_parameters(trial_params, target_properties)
            
            if score < best_score:
                best_score = score
                best_params = trial_params
                
        return best_params
    
    def _apply_constraints(
        self, 
        params: ProcessParameters, 
        constraints: Dict[str, Tuple[float, float]]
    ) -> ProcessParameters:
        """Apply parameter constraints."""
        
        result = ProcessParameters()
        
        for param_name, (min_val, max_val) in constraints.items():
            if hasattr(params, param_name):
                current_val = getattr(params, param_name)
                constrained_val = np.clip(current_val, min_val, max_val)
                setattr(result, param_name, constrained_val)
                
        return result
    
    def _evaluate_parameters(
        self, 
        params: ProcessParameters, 
        target_properties: Dict[str, float]
    ) -> float:
        """Evaluate parameter quality based on target properties."""
        
        # Simplified property prediction (in practice, this would use 
        # physics-based models or additional ML models)
        predicted_properties = {
            'density': 0.99 - (params.scan_speed - 800) / 2000,
            'roughness': 5 + (params.laser_power - 200) / 40,
            'strength': 1000 + (params.laser_power - 200) * 2
        }
        
        # Calculate weighted MSE
        score = 0.0
        for prop, target_val in target_properties.items():
            if prop in predicted_properties:
                error = (predicted_properties[prop] - target_val) ** 2
                score += error
                
        return score
    
    def save_model(self, save_path: str):
        """Save model weights and configuration."""
        
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'diffusion': self.diffusion_model.state_dict(),
            'decoder': self.decoder.state_dict(),
            'config': self.config,
            'alloy': self.alloy,
            'process': self.process
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model weights and configuration."""
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        self.config = checkpoint['config']
        self.alloy = checkpoint['alloy']
        self.process = checkpoint['process']
        
        print(f"Model loaded from {load_path}")


def train_diffusion_model(
    dataset,
    architecture: str = "unet3d",
    diffusion_steps: int = 1000,
    batch_size: int = 8,
    epochs: int = 500,
    learning_rate: float = 1e-4,
    device: Optional[str] = None
) -> MicrostructureDiffusion:
    """Train a diffusion model on the provided dataset."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = MicrostructureDiffusion(pretrained=False, device=device)
    model.train()
    
    # Setup optimization
    params = list(model.encoder.parameters()) + \
             list(model.diffusion_model.parameters()) + \
             list(model.decoder.parameters())
    
    optimizer = AdamW(params, lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            microstructures, parameters = batch
            microstructures = microstructures.to(device)
            parameters = parameters.to(device)
            
            optimizer.zero_grad()
            
            # Encode microstructures
            latent_micro = model.encoder(microstructures.flatten(1))
            
            # Forward diffusion process
            t = torch.randint(0, diffusion_steps, (microstructures.shape[0],), device=device)
            noise = torch.randn_like(latent_micro)
            
            # Add noise to latent representations
            noisy_latent = model.diffusion_model.add_noise(latent_micro, noise, t)
            
            # Predict noise
            predicted_noise = model.diffusion_model(noisy_latent, t, condition=parameters)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Validation
        if hasattr(dataset, 'val_loader'):
            val_loss = _validate_model(model, dataset.val_loader, device)
            print(f"Validation Loss = {val_loss:.4f}")
    
    model.eval()
    return model


def _validate_model(model, val_loader, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            microstructures, parameters = batch
            microstructures = microstructures.to(device)
            parameters = parameters.to(device)
            
            # Encode microstructures
            latent_micro = model.encoder(microstructures.flatten(1))
            
            # Sample random timestep
            t = torch.randint(0, model.config['diffusion']['num_steps'], 
                            (microstructures.shape[0],), device=device)
            noise = torch.randn_like(latent_micro)
            
            # Add noise
            noisy_latent = model.diffusion_model.add_noise(latent_micro, noise, t)
            
            # Predict noise
            predicted_noise = model.diffusion_model(noisy_latent, t, condition=parameters)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0