"""Encoder models for microstructure analysis and feature extraction."""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MicrostructureEncoder(nn.Module):
    """Encoder for converting microstructure images to latent representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 256,
                 num_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Simplified architecture for Generation 1
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        ]
        
        self.encoder = nn.Sequential(*layers)
        
        # Simplified decoder for Generation 1
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Normalize output
        ]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode microstructure to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to microstructure."""
        return self.decoder(z)
    
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode cycle for reconstruction loss."""
        z = self.forward(x)
        return self.decode(z)


class Conv3DEncoder(nn.Module):
    """3D Convolutional encoder for volumetric microstructure data."""
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 256,
                 base_channels: int = 32):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # 3D convolutional layers with progressive downsampling
        self.conv_layers = nn.Sequential(
            # First block: 128x128x128 -> 64x64x64
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Second block: 64x64x64 -> 32x32x32
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Third block: 32x32x32 -> 16x16x16
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Fourth block: 16x16x16 -> 8x8x8
            nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(),
            nn.Conv3d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        # Global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D encoder."""
        # Apply convolutional layers
        h = self.conv_layers(x)
        
        # Global pooling
        h = self.global_pool(h)
        h = h.view(h.size(0), -1)
        
        # Final projection
        return self.fc(h)


class FeatureExtractor(nn.Module):
    """Extract quantitative features from microstructures."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        
        # Learnable feature extractors
        self.grain_analyzer = nn.Sequential(
            nn.Conv3d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv3d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Grain size features
        )
        
        self.texture_analyzer = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # Texture features
        )
        
        self.porosity_analyzer = nn.Sequential(
            nn.Conv3d(1, 4, 7, padding=3),
            nn.ReLU(),
            nn.Conv3d(4, 8, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(16),
            nn.Flatten(),
            nn.Linear(8 * 16 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # Porosity features
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 8, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and fuse microstructure features."""
        
        # Extract different types of features
        grain_features = self.grain_analyzer(x)
        texture_features = self.texture_analyzer(x)
        porosity_features = self.porosity_analyzer(x)
        
        # Concatenate and fuse features
        combined_features = torch.cat([grain_features, texture_features, porosity_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        return fused_features
    
    def extract_grain_size_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Extract grain size distribution features."""
        return self.grain_analyzer(x)
    
    def extract_texture_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """Extract texture coefficient features."""
        return self.texture_analyzer(x)
    
    def extract_porosity_metrics(self, x: torch.Tensor) -> torch.Tensor:
        """Extract porosity-related features."""
        return self.porosity_analyzer(x)


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder for capturing features at different resolutions."""
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        
        # Encoders for different scales
        self.scale1_encoder = self._make_scale_encoder(in_channels, 32, 2)  # Full resolution
        self.scale2_encoder = self._make_scale_encoder(in_channels, 64, 4)  # Half resolution
        self.scale3_encoder = self._make_scale_encoder(in_channels, 128, 8)  # Quarter resolution
        
        # Feature fusion
        total_features = 32 + 64 + 128
        self.fusion = nn.Sequential(
            nn.Linear(total_features, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def _make_scale_encoder(self, in_channels: int, out_features: int, 
                           downsample_factor: int) -> nn.Module:
        """Create encoder for specific scale."""
        
        layers = []
        current_channels = in_channels
        
        # Progressive downsampling to reach target factor
        num_pools = int(np.log2(downsample_factor))
        for i in range(num_pools + 1):
            next_channels = min(out_features // (2 ** (num_pools - i)), out_features)
            
            layers.extend([
                nn.Conv3d(current_channels, next_channels, 3, padding=1),
                nn.BatchNorm3d(next_channels),
                nn.ReLU()
            ])
            
            if i < num_pools:
                layers.append(nn.MaxPool3d(2))
            
            current_channels = next_channels
        
        # Global pooling and projection
        layers.extend([
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_features, out_features)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale feature extraction."""
        
        # Extract features at different scales
        feat1 = self.scale1_encoder(x)
        
        # Downsample for other scales
        x_half = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)
        feat2 = self.scale2_encoder(x_half)
        
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=False)
        feat3 = self.scale3_encoder(x_quarter)
        
        # Fuse multi-scale features
        combined_features = torch.cat([feat1, feat2, feat3], dim=1)
        return self.fusion(combined_features)


class VariationalEncoder(nn.Module):
    """Variational autoencoder for microstructure representation."""
    
    def __init__(self, input_dim: int, latent_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Mean and log variance projections
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to mean and log variance."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mean, and log variance."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the latent space."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class ContrastiveEncoder(nn.Module):
    """Contrastive learning encoder for microstructure similarity."""
    
    def __init__(self, input_dim: int, latent_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        
        self.temperature = temperature
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for contrastive learning."""
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
    
    def compute_contrastive_loss(self, features1: torch.Tensor, 
                                features2: torch.Tensor, 
                                labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between feature pairs."""
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Create labels for positive and negative pairs
        batch_size = features1.size(0)
        mask = torch.eye(batch_size, device=features1.device).bool()
        
        # Compute cross-entropy loss
        logits = similarity_matrix[mask].view(batch_size, -1)
        targets = torch.arange(batch_size, device=features1.device)
        
        loss = F.cross_entropy(logits, targets)
        return loss