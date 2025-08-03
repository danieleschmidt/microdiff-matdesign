"""Conditional diffusion models for guided parameter generation."""

from typing import Optional, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .diffusion import DiffusionModel


class ConditionalDiffusion(nn.Module):
    """Conditional diffusion model with multi-modal conditioning."""
    
    def __init__(self, base_diffusion: DiffusionModel, condition_types: List[str],
                 condition_dims: Dict[str, int]):
        super().__init__()
        
        self.base_diffusion = base_diffusion
        self.condition_types = condition_types
        self.condition_dims = condition_dims
        
        # Condition encoders for different modalities
        self.condition_encoders = nn.ModuleDict()
        
        for cond_type in condition_types:
            input_dim = condition_dims[cond_type]
            
            if cond_type == 'microstructure':
                # For image-like conditions
                self.condition_encoders[cond_type] = MicrostructureConditionEncoder(input_dim)
            elif cond_type == 'material_properties':
                # For material property vectors
                self.condition_encoders[cond_type] = PropertyConditionEncoder(input_dim)
            elif cond_type == 'constraints':
                # For manufacturing constraints
                self.condition_encoders[cond_type] = ConstraintConditionEncoder(input_dim)
            elif cond_type == 'text':
                # For text descriptions
                self.condition_encoders[cond_type] = TextConditionEncoder(input_dim)
            else:
                # Generic encoder
                self.condition_encoders[cond_type] = GenericConditionEncoder(input_dim)
        
        # Cross-attention mechanism for condition fusion
        latent_dim = base_diffusion.hidden_dim
        self.condition_fusion = ConditionFusion(latent_dim, len(condition_types))
        
        # Classifier-free guidance
        self.null_embeddings = nn.ParameterDict()
        for cond_type in condition_types:
            self.null_embeddings[cond_type] = nn.Parameter(torch.randn(1, latent_dim))
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                conditions: Optional[Dict[str, torch.Tensor]] = None,
                dropout_prob: float = 0.1) -> torch.Tensor:
        """Forward pass with conditional information."""
        
        # Encode conditions
        condition_embeddings = {}
        
        if conditions is not None:
            for cond_type, cond_data in conditions.items():
                if cond_type in self.condition_encoders:
                    # Apply dropout for classifier-free guidance
                    if self.training and torch.rand(1) < dropout_prob:
                        # Use null embedding
                        batch_size = cond_data.size(0)
                        condition_embeddings[cond_type] = self.null_embeddings[cond_type].repeat(batch_size, 1)
                    else:
                        condition_embeddings[cond_type] = self.condition_encoders[cond_type](cond_data)
        
        # Fuse conditions
        if condition_embeddings:
            fused_condition = self.condition_fusion(condition_embeddings)
        else:
            fused_condition = None
        
        # Apply base diffusion model
        return self.base_diffusion(x, timestep, fused_condition)
    
    def conditional_generation(self, conditions: Dict[str, torch.Tensor],
                             num_proposals: int = 100, select_best: bool = True,
                             guidance_scale: float = 7.5) -> torch.Tensor:
        """Generate parameters satisfying specified conditions."""
        
        batch_size = list(conditions.values())[0].size(0)
        device = list(conditions.values())[0].device
        
        # Generate multiple proposals
        proposals = []
        
        for _ in range(num_proposals):
            # Sample noise
            noise_shape = (batch_size, self.base_diffusion.input_dim)
            noise = torch.randn(noise_shape, device=device)
            
            # Conditional sampling
            sample = self.sample_conditional(noise, conditions, guidance_scale)
            proposals.append(sample)
        
        proposals = torch.stack(proposals, dim=1)  # [batch, num_proposals, param_dim]
        
        if select_best:
            # Select best proposal based on condition satisfaction
            scores = self._evaluate_proposals(proposals, conditions)
            best_indices = torch.argmax(scores, dim=1)
            
            best_proposals = proposals[torch.arange(batch_size), best_indices]
            return best_proposals
        else:
            return proposals
    
    def sample_conditional(self, noise: torch.Tensor, conditions: Dict[str, torch.Tensor],
                          guidance_scale: float = 7.5) -> torch.Tensor:
        """Sample with conditional guidance."""
        
        x = noise
        device = noise.device
        
        # Reverse diffusion process
        for t in reversed(range(self.base_diffusion.num_steps)):
            t_tensor = torch.full((noise.size(0),), t, device=device, dtype=torch.long)
            
            if guidance_scale != 1.0:
                # Classifier-free guidance
                noise_pred_cond = self(x, t_tensor, conditions)
                noise_pred_uncond = self(x, t_tensor, None)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self(x, t_tensor, conditions)
            
            # DDPM sampling step
            x = self._ddpm_step(x, noise_pred, t)
        
        return x
    
    def _ddpm_step(self, x: torch.Tensor, noise_pred: torch.Tensor, t: int) -> torch.Tensor:
        """Single DDPM sampling step."""
        
        alpha_t = self.base_diffusion.alphas[t]
        alpha_cumprod_t = self.base_diffusion.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.base_diffusion.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # Predict original sample
        pred_original = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Compute previous sample
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
        prev_sample = torch.sqrt(alpha_cumprod_t_prev) * pred_original + pred_sample_direction
        
        # Add noise for non-final step
        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.base_diffusion.betas[t]) * noise
            prev_sample = prev_sample + variance
        
        return prev_sample
    
    def _evaluate_proposals(self, proposals: torch.Tensor, 
                           conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate how well proposals satisfy conditions."""
        
        batch_size, num_proposals, param_dim = proposals.shape
        scores = torch.zeros(batch_size, num_proposals, device=proposals.device)
        
        # Evaluate each condition type
        for cond_type, cond_data in conditions.items():
            if cond_type == 'material_properties':
                # Score based on expected property satisfaction
                property_scores = self._score_material_properties(proposals, cond_data)
                scores += property_scores
            elif cond_type == 'constraints':
                # Score based on constraint satisfaction
                constraint_scores = self._score_constraints(proposals, cond_data)
                scores += constraint_scores
        
        return scores
    
    def _score_material_properties(self, proposals: torch.Tensor, 
                                  target_properties: torch.Tensor) -> torch.Tensor:
        """Score proposals based on material property targets."""
        
        # Simplified property prediction (would use real models in practice)
        batch_size, num_proposals, param_dim = proposals.shape
        
        # Extract parameters
        laser_power = proposals[:, :, 0]
        scan_speed = proposals[:, :, 1]
        
        # Predict properties (simplified)
        predicted_density = 0.99 - (scan_speed - 800) / 2000
        predicted_strength = 1000 + (laser_power - 200) * 2
        
        # Target properties (assuming density and strength)
        target_density = target_properties[:, 0].unsqueeze(1)
        target_strength = target_properties[:, 1].unsqueeze(1)
        
        # Compute scores (negative MSE)
        density_error = (predicted_density - target_density) ** 2
        strength_error = (predicted_strength - target_strength) ** 2
        
        scores = -(density_error + strength_error / 10000)  # Normalize strength error
        
        return scores
    
    def _score_constraints(self, proposals: torch.Tensor, 
                          constraints: torch.Tensor) -> torch.Tensor:
        """Score proposals based on constraint satisfaction."""
        
        # Constraints format: [min_power, max_power, min_speed, max_speed, ...]
        batch_size, num_proposals, param_dim = proposals.shape
        
        scores = torch.zeros(batch_size, num_proposals, device=proposals.device)
        
        # Check parameter bounds
        for i in range(min(param_dim, constraints.size(1) // 2)):
            param_values = proposals[:, :, i]
            min_constraint = constraints[:, i * 2].unsqueeze(1)
            max_constraint = constraints[:, i * 2 + 1].unsqueeze(1)
            
            # Penalty for violating constraints
            violation_penalty = torch.clamp(min_constraint - param_values, min=0) + \
                               torch.clamp(param_values - max_constraint, min=0)
            
            scores -= violation_penalty * 10  # Heavy penalty for violations
        
        return scores


class MicrostructureConditionEncoder(nn.Module):
    """Encoder for microstructure-based conditioning."""
    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        
        # Assume input is flattened 3D volume
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))


class PropertyConditionEncoder(nn.Module):
    """Encoder for material property conditioning."""
    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConstraintConditionEncoder(nn.Module):
    """Encoder for manufacturing constraint conditioning."""
    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Constraints are typically bounded
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TextConditionEncoder(nn.Module):
    """Encoder for text-based conditioning."""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3
        )
        self.projection = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is token indices [batch, seq_len]
        embedded = self.embedding(x)
        
        # Create attention mask for padding
        mask = (x == 0)  # Assuming 0 is padding token
        
        # Apply transformer
        features = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Global average pooling
        pooled = features.mean(dim=1)
        
        return self.projection(pooled)


class GenericConditionEncoder(nn.Module):
    """Generic encoder for any condition type."""
    
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, min(input_dim * 2, 512)),
            nn.ReLU(),
            nn.Linear(min(input_dim * 2, 512), output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))


class ConditionFusion(nn.Module):
    """Fuse multiple condition embeddings using cross-attention."""
    
    def __init__(self, embed_dim: int, num_conditions: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_conditions = num_conditions
        
        # Cross-attention for condition fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # Learnable query for fusion
        self.fusion_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Final projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, condition_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple condition embeddings."""
        
        if not condition_embeddings:
            batch_size = 1  # Default
            return torch.zeros(batch_size, self.embed_dim, device=next(self.parameters()).device)
        
        # Stack condition embeddings
        conditions = []
        for cond_emb in condition_embeddings.values():
            conditions.append(cond_emb.unsqueeze(1))  # Add sequence dimension
        
        condition_stack = torch.cat(conditions, dim=1)  # [batch, num_conditions, embed_dim]
        batch_size = condition_stack.size(0)
        
        # Expand fusion query for batch
        query = self.fusion_query.expand(batch_size, -1, -1)
        
        # Cross-attention fusion
        fused_embedding, _ = self.cross_attention(query, condition_stack, condition_stack)
        
        # Project and squeeze
        output = self.output_proj(fused_embedding.squeeze(1))
        
        return output


class GuidedDiffusion(nn.Module):
    """Diffusion model with explicit guidance functions."""
    
    def __init__(self, base_diffusion: DiffusionModel, guidance_functions: Dict[str, callable]):
        super().__init__()
        
        self.base_diffusion = base_diffusion
        self.guidance_functions = guidance_functions
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                guidance_targets: Optional[Dict[str, torch.Tensor]] = None,
                guidance_scale: float = 1.0) -> torch.Tensor:
        """Forward pass with explicit guidance."""
        
        # Base noise prediction
        noise_pred = self.base_diffusion(x, timestep)
        
        if guidance_targets is not None:
            # Apply guidance gradients
            x.requires_grad_(True)
            
            total_guidance = 0
            for guide_name, target in guidance_targets.items():
                if guide_name in self.guidance_functions:
                    guidance_fn = self.guidance_functions[guide_name]
                    guidance_loss = guidance_fn(x, target)
                    
                    # Compute gradient
                    guidance_grad = torch.autograd.grad(
                        guidance_loss.sum(), x, retain_graph=True
                    )[0]
                    
                    total_guidance += guidance_grad
            
            # Apply guidance to noise prediction
            noise_pred = noise_pred - guidance_scale * total_guidance
            
            x.requires_grad_(False)
        
        return noise_pred