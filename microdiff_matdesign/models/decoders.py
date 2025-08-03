"""Decoder models for converting latent representations to process parameters."""

from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ParameterDecoder(nn.Module):
    """Decoder for converting latent representations to process parameters."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512, output_dim: int = 6,
                 num_layers: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Parameter-specific decoders
        self.laser_power_decoder = self._make_parameter_decoder(latent_dim, hidden_dim, 1)
        self.scan_speed_decoder = self._make_parameter_decoder(latent_dim, hidden_dim, 1)
        self.layer_thickness_decoder = self._make_parameter_decoder(latent_dim, hidden_dim, 1)
        self.hatch_spacing_decoder = self._make_parameter_decoder(latent_dim, hidden_dim, 1)
        self.powder_temp_decoder = self._make_parameter_decoder(latent_dim, hidden_dim, 1)
        
        # Atmosphere classifier (categorical)
        self.atmosphere_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # argon, nitrogen, vacuum, air
        )
        
        # Global decoder for joint parameter prediction
        layers = []
        current_dim = latent_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim - 1))  # Exclude categorical atmosphere
        
        self.global_decoder = nn.Sequential(*layers)
        
        # Parameter constraints and normalization
        self.register_buffer('param_means', torch.tensor([200.0, 800.0, 30.0, 120.0, 80.0]))
        self.register_buffer('param_stds', torch.tensor([50.0, 200.0, 10.0, 30.0, 20.0]))
        self.register_buffer('param_mins', torch.tensor([50.0, 200.0, 10.0, 50.0, 20.0]))
        self.register_buffer('param_maxs', torch.tensor([500.0, 2000.0, 100.0, 300.0, 200.0]))
    
    def _make_parameter_decoder(self, input_dim: int, hidden_dim: int, 
                               output_dim: int) -> nn.Module:
        """Create a parameter-specific decoder."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, z: torch.Tensor, use_individual_decoders: bool = False) -> torch.Tensor:
        """Decode latent representation to process parameters."""
        
        if use_individual_decoders:
            # Use parameter-specific decoders
            laser_power = self.laser_power_decoder(z)
            scan_speed = self.scan_speed_decoder(z)
            layer_thickness = self.layer_thickness_decoder(z)
            hatch_spacing = self.hatch_spacing_decoder(z)
            powder_temp = self.powder_temp_decoder(z)
            
            # Combine continuous parameters
            continuous_params = torch.cat([
                laser_power, scan_speed, layer_thickness, 
                hatch_spacing, powder_temp
            ], dim=1)
        else:
            # Use global decoder
            continuous_params = self.global_decoder(z)
        
        # Apply normalization and constraints
        normalized_params = self._denormalize_parameters(continuous_params)
        constrained_params = self._apply_constraints(normalized_params)
        
        return constrained_params
    
    def forward_with_atmosphere(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode to both continuous and categorical parameters."""
        continuous_params = self.forward(z)
        atmosphere_logits = self.atmosphere_decoder(z)
        return continuous_params, atmosphere_logits
    
    def _denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """Convert normalized parameters to physical values."""
        return normalized_params * self.param_stds + self.param_means
    
    def _apply_constraints(self, params: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to parameters."""
        return torch.clamp(params, min=self.param_mins, max=self.param_maxs)
    
    def predict_uncertainty(self, z: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict parameters with uncertainty estimation using dropout."""
        self.train()  # Enable dropout for uncertainty
        
        samples = []
        for _ in range(num_samples):
            sample = self.forward(z)
            samples.append(sample.unsqueeze(0))
        
        samples = torch.cat(samples, dim=0)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        
        self.eval()  # Return to eval mode
        return mean, std
    
    def inverse(self, parameters: torch.Tensor) -> torch.Tensor:
        """Approximate inverse mapping from parameters to latent space."""
        # This is a simplified inverse - in practice, you might train a separate encoder
        # or use more sophisticated techniques
        
        # Normalize parameters
        normalized = (parameters - self.param_means) / self.param_stds
        
        # Simple linear inverse (this could be replaced with a learned inverse)
        # For now, we'll use the pseudoinverse of the decoder weights
        with torch.no_grad():
            # Get the last linear layer weights
            decoder_weights = []
            for module in self.global_decoder.modules():
                if isinstance(module, nn.Linear):
                    decoder_weights.append(module.weight.data)
            
            if decoder_weights:
                # Use pseudoinverse of the last layer
                last_weights = decoder_weights[-1]
                pseudo_inv = torch.pinverse(last_weights)
                latent_approx = torch.matmul(normalized, pseudo_inv.T)
                return latent_approx
            else:
                # Fallback: random latent vector
                return torch.randn(parameters.size(0), self.latent_dim, device=parameters.device)


class MultiObjectiveDecoder(nn.Module):
    """Decoder for multi-objective optimization scenarios."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512, 
                 objectives: List[str] = ['quality', 'speed', 'cost']):
        super().__init__()
        
        self.objectives = objectives
        self.num_objectives = len(objectives)
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Objective-specific heads
        self.objective_heads = nn.ModuleDict()
        for obj in objectives:
            self.objective_heads[obj] = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 4, 5)  # 5 process parameters per objective
            )
        
        # Pareto optimization layer
        self.pareto_weights = nn.Parameter(torch.ones(self.num_objectives) / self.num_objectives)
    
    def forward(self, z: torch.Tensor, objective_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode with multi-objective consideration."""
        
        # Extract shared features
        shared_features = self.shared_layers(z)
        
        # Get objective-specific parameters
        objective_params = {}
        for obj in self.objectives:
            objective_params[obj] = self.objective_heads[obj](shared_features)
        
        # Combine objectives using weights
        if objective_weights is None:
            objective_weights = F.softmax(self.pareto_weights, dim=0)
        
        # Weighted combination of objective-specific parameters
        combined_params = torch.zeros_like(objective_params[self.objectives[0]])
        for i, obj in enumerate(self.objectives):
            combined_params += objective_weights[i] * objective_params[obj]
        
        return combined_params
    
    def get_pareto_front(self, z: torch.Tensor, num_points: int = 100) -> torch.Tensor:
        """Generate Pareto front of parameter solutions."""
        
        # Generate diverse weight combinations
        weights = torch.rand(num_points, self.num_objectives, device=z.device)
        weights = F.softmax(weights, dim=1)
        
        # Generate parameters for each weight combination
        pareto_solutions = []
        for i in range(num_points):
            params = self.forward(z, weights[i])
            pareto_solutions.append(params)
        
        return torch.stack(pareto_solutions, dim=1)


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for multi-scale parameter prediction."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Global parameters (affect entire build)
        self.global_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # powder_bed_temp, atmosphere
        )
        
        # Layer-level parameters
        self.layer_decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),  # Include global params
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # layer_thickness, scan_strategy
        )
        
        # Track-level parameters
        self.track_decoder = nn.Sequential(
            nn.Linear(latent_dim + 4, hidden_dim),  # Include global + layer params
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # laser_power, scan_speed, hatch_spacing
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hierarchical parameter prediction."""
        
        # Global parameters
        global_params = self.global_decoder(z)
        
        # Layer parameters (conditioned on global)
        layer_input = torch.cat([z, global_params], dim=1)
        layer_params = self.layer_decoder(layer_input)
        
        # Track parameters (conditioned on global + layer)
        track_input = torch.cat([z, global_params, layer_params], dim=1)
        track_params = self.track_decoder(track_input)
        
        return {
            'global': global_params,
            'layer': layer_params,
            'track': track_params
        }


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that adjusts based on material properties."""
    
    def __init__(self, latent_dim: int, material_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        
        self.material_dim = material_dim
        
        # Material embedding network
        self.material_encoder = nn.Sequential(
            nn.Linear(material_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8)
        )
        
        # Adaptive decoder layers
        self.decoder_layers = nn.ModuleList([
            AdaptiveLayer(latent_dim, hidden_dim, hidden_dim // 8),
            AdaptiveLayer(hidden_dim, hidden_dim // 2, hidden_dim // 8),
            AdaptiveLayer(hidden_dim // 2, hidden_dim // 4, hidden_dim // 8),
            nn.Linear(hidden_dim // 4, 5)  # Output layer
        ])
    
    def forward(self, z: torch.Tensor, material_properties: torch.Tensor) -> torch.Tensor:
        """Decode with material-specific adaptation."""
        
        # Encode material properties
        material_emb = self.material_encoder(material_properties)
        
        # Apply adaptive layers
        h = z
        for layer in self.decoder_layers[:-1]:
            h = layer(h, material_emb)
        
        # Final output
        return self.decoder_layers[-1](h)


class AdaptiveLayer(nn.Module):
    """Layer that adapts based on material properties."""
    
    def __init__(self, input_dim: int, output_dim: int, material_emb_dim: int):
        super().__init__()
        
        # Base transformation
        self.base_linear = nn.Linear(input_dim, output_dim)
        
        # Material-dependent modulation
        self.scale_net = nn.Linear(material_emb_dim, output_dim)
        self.shift_net = nn.Linear(material_emb_dim, output_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, material_emb: torch.Tensor) -> torch.Tensor:
        """Forward with material-dependent modulation."""
        
        # Base transformation
        h = self.base_linear(x)
        
        # Material-dependent scaling and shifting
        scale = torch.sigmoid(self.scale_net(material_emb))
        shift = self.shift_net(material_emb)
        
        # Apply modulation
        h = h * scale + shift
        
        return self.dropout(self.activation(h))


class UncertaintyDecoder(nn.Module):
    """Decoder with built-in uncertainty quantification."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512, output_dim: int = 5):
        super().__init__()
        
        # Shared encoder
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Variance prediction head (log variance for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict parameters with uncertainty."""
        
        # Shared features
        shared_features = self.shared_layers(z)
        
        # Predict mean and log variance
        mean = self.mean_head(shared_features)
        logvar = self.logvar_head(shared_features)
        
        return mean, logvar
    
    def sample(self, z: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample parameters from predicted distribution."""
        
        mean, logvar = self.forward(z)
        std = torch.exp(0.5 * logvar)
        
        # Sample multiple times if requested
        if num_samples == 1:
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            samples = []
            for _ in range(num_samples):
                eps = torch.randn_like(std)
                samples.append((mean + eps * std).unsqueeze(0))
            return torch.cat(samples, dim=0)
    
    def compute_uncertainty_loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty-aware loss."""
        
        mean, logvar = self.forward(z)
        
        # Heteroscedastic loss (uncertainty-weighted MSE)
        precision = torch.exp(-logvar)
        loss = torch.sum(precision * (target - mean) ** 2 + logvar, dim=1)
        
        return torch.mean(loss)