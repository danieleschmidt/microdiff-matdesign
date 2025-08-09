"""Diffusion model architectures for microstructure inverse design."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with timestep and condition embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 cond_emb_dim: Optional[int] = None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        if cond_emb_dim:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_emb_dim, out_channels)
            )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)
        
        # Add time embedding
        h = h + self.time_mlp(time_emb)
        
        # Add conditional embedding if provided
        if cond_emb is not None and self.cond_emb_dim:
            h = h + self.cond_mlp(cond_emb)
        
        h = self.block2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(1)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        h = (attn @ v).reshape(B, C)
        h = self.proj(h)
        
        return x + h


class DiffusionModel(nn.Module):
    """Main diffusion model for noise prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_steps: int = 1000,
                 time_emb_dim: int = 128, cond_emb_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_emb_dim),
            AttentionBlock(hidden_dim, num_heads=8),
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_emb_dim),
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_emb_dim),
            AttentionBlock(hidden_dim, num_heads=8),
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_emb_dim),
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize diffusion schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1. - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule for diffusion process."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for noise prediction."""
        
        # Embed timestep
        time_emb = self.time_embed(timestep)
        
        # Project input
        h = self.input_proj(x)
        
        # Apply blocks
        for i, block in enumerate(self.blocks):
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, condition)
            else:  # AttentionBlock
                h = block(h)
        
        # Output projection
        return self.output_proj(h)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, 
                  timestep: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to diffusion schedule."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample(self, noise: torch.Tensor, condition: Optional[torch.Tensor] = None,
               guidance_scale: float = 1.0) -> torch.Tensor:
        """Sample from the diffusion model using DDPM sampling."""
        
        # Use provided noise
        x = noise
        device = x.device
        
        # Reverse diffusion process
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            if condition is not None and guidance_scale != 1.0:
                # Classifier-free guidance
                noise_pred_cond = self(x, t_tensor, condition)
                noise_pred_uncond = self(x, t_tensor, None)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self(x, t_tensor, condition)
            
            # Compute previous sample
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            # Compute coefficients
            pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # Compute previous sample mean
            pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
            prev_sample = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction
            
            # Add noise for non-final step
            if t > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(self.betas[t]) * noise
                prev_sample = prev_sample + variance
            
            x = prev_sample
        
        return x


class DiffusionUNet3D(nn.Module):
    """3D U-Net architecture for volumetric diffusion."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 8, 
                 base_channels: int = 64, channel_multipliers: list = [1, 2, 4, 8],
                 attention_resolutions: list = [16, 8], num_res_blocks: int = 2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input convolution
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(ch, out_ch, time_emb_dim))
                ch = out_ch
            self.encoder_blocks.append(blocks)
            
            # Attention
            if i in attention_resolutions:
                self.encoder_attns.append(SpatialAttention3D(ch))
            else:
                self.encoder_attns.append(nn.Identity())
            
            # Downsample
            if i < len(channel_multipliers) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample3D(ch)]))
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock3D(ch, ch, time_emb_dim),
            SpatialAttention3D(ch),
            ResBlock3D(ch, ch, time_emb_dim)
        )
        
        # Decoder (similar structure, reversed)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            # Upsample
            if i > 0:
                self.decoder_blocks.append(nn.ModuleList([Upsample3D(ch)]))
            
            # Residual blocks with skip connections
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                blocks.append(ResBlock3D(ch + out_ch if j == 0 else ch, out_ch, time_emb_dim))
                ch = out_ch
            self.decoder_blocks.append(blocks)
            
            # Attention
            if (len(channel_multipliers) - 1 - i) in attention_resolutions:
                self.decoder_attns.append(SpatialAttention3D(ch))
            else:
                self.decoder_attns.append(nn.Identity())
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D U-Net."""
        
        # Time embedding
        time_emb = self.time_embed(timestep)
        
        # Input projection
        h = self.input_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for i, (blocks, attn) in enumerate(zip(self.encoder_blocks, self.encoder_attns)):
            for block in blocks:
                if isinstance(block, ResBlock3D):
                    h = block(h, time_emb)
                else:  # Downsample
                    h = block(h)
            h = attn(h)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h)
        
        # Decoder
        for i, (blocks, attn) in enumerate(zip(self.decoder_blocks, self.decoder_attns)):
            # Upsample
            if i > 0 and isinstance(blocks[0], Upsample3D):
                h = blocks[0](h)
                blocks = blocks[1:]
            
            # Skip connection
            if skip_connections:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
            
            # Residual blocks
            for block in blocks:
                h = block(h, time_emb)
            
            h = attn(h)
        
        # Output
        return self.output_conv(h)


class ResBlock3D(nn.Module):
    """3D Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Add time embedding
        time_proj = self.time_mlp(time_emb)
        h = h + time_proj[..., None, None, None]
        
        h = self.block2(h)
        return h + self.shortcut(x)


class SpatialAttention3D(nn.Module):
    """3D spatial attention mechanism."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, D * H * W)
        q, k, v = qkv.unbind(1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        h = (attn @ v).reshape(B, C, D, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample3D(nn.Module):
    """3D downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class DiffusionTransformer(nn.Module):
    """Transformer-based diffusion model for irregular data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Project input
        h = self.input_proj(x)
        
        # Add time embedding
        time_emb = self.time_embed(timestep)
        h = h + time_emb.unsqueeze(1)
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)
        
        # Output projection
        return self.output_proj(h)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x