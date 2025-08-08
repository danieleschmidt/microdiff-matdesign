"""Uncertainty-Aware Bayesian Diffusion Models.

This module implements principled Bayesian uncertainty quantification for 
diffusion models in materials science applications. Provides calibrated 
confidence intervals essential for safety-critical manufacturing decisions.

Research Contribution:
- Bayesian uncertainty decomposition (aleatoric vs epistemic)
- Monte Carlo Dropout for epistemic uncertainty
- Calibrated confidence intervals  
- Out-of-distribution detection capabilities

Expected Performance:
- Properly calibrated uncertainty estimates
- Enhanced reliability for certification requirements  
- Better detection of novel parameter combinations
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from .diffusion import DiffusionModel


class VariationalLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        prior_std: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight mean and log variance
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))
        
        # Bias mean and log variance
        self.bias_mean = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight_mean)
        nn.init.constant_(self.weight_logvar, -5.0)  # Start with low variance
        nn.init.zeros_(self.bias_mean)
        nn.init.constant_(self.bias_logvar, -5.0)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optional sampling.
        
        Args:
            x: Input tensor
            sample: Whether to sample from posterior (True) or use mean (False)
            
        Returns:
            Output tensor
        """
        if sample and self.training:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(self.weight_mean)
            bias_eps = torch.randn_like(self.bias_mean)
            
            weight = self.weight_mean + weight_std * weight_eps
            bias = self.bias_mean + bias_std * bias_eps
        else:
            # Use mean values
            weight = self.weight_mean
            bias = self.bias_mean
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with prior."""
        # KL(q(w)||p(w)) for Gaussian distributions
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        weight_kl = 0.5 * torch.sum(
            self.weight_mean**2 / self.prior_std**2 +
            weight_var / self.prior_std**2 -
            self.weight_logvar +
            math.log(self.prior_std**2) - 1
        )
        
        bias_kl = 0.5 * torch.sum(
            self.bias_mean**2 / self.prior_std**2 +
            bias_var / self.prior_std**2 -
            self.bias_logvar +
            math.log(self.prior_std**2) - 1
        )
        
        return weight_kl + bias_kl


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer for epistemic uncertainty."""
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout always enabled."""
        return F.dropout(x, p=self.p, training=True)


class BayesianDiffusionDecoder(nn.Module):
    """Bayesian decoder with uncertainty quantification."""
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 6,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
        use_variational: bool = True
    ):
        """Initialize Bayesian decoder.
        
        Args:
            latent_dim: Input latent dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output parameter dimension
            n_samples: Number of MC samples for uncertainty
            dropout_rate: Dropout rate for MC Dropout
            use_variational: Whether to use variational layers
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        self.use_variational = use_variational
        
        if use_variational:
            # Variational Bayesian layers
            self.fc1 = VariationalLinear(latent_dim, hidden_dim)
            self.fc2 = VariationalLinear(hidden_dim, hidden_dim)
            self.mean_head = VariationalLinear(hidden_dim, output_dim)
            self.logvar_head = VariationalLinear(hidden_dim, output_dim)
        else:
            # Standard layers with MC Dropout
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.mean_head = nn.Linear(hidden_dim, output_dim)
            self.logvar_head = nn.Linear(hidden_dim, output_dim)
            
        self.dropout1 = MCDropout(dropout_rate)
        self.dropout2 = MCDropout(dropout_rate)
        
        # Learnable epistemic uncertainty parameter
        self.log_epistemic_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(
        self, 
        z: torch.Tensor, 
        return_uncertainty: bool = False,
        n_samples: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with optional uncertainty quantification.
        
        Args:
            z: Latent input tensor
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of MC samples (overrides default)
            
        Returns:
            If return_uncertainty=False: mean predictions
            If return_uncertainty=True: (mean, aleatoric_var, epistemic_var)
        """
        if not return_uncertainty:
            # Single forward pass
            h = F.silu(self.fc1(z))
            h = self.dropout1(h)
            h = F.silu(self.fc2(h))
            h = self.dropout2(h)
            
            mean = self.mean_head(h)
            return mean
        
        # Multiple forward passes for uncertainty quantification
        if n_samples is None:
            n_samples = self.n_samples
            
        means = []
        logvars = []
        
        for _ in range(n_samples):
            h = F.silu(self.fc1(z))
            h = self.dropout1(h)
            h = F.silu(self.fc2(h))
            h = self.dropout2(h)
            
            mean = self.mean_head(h)
            logvar = self.logvar_head(h)
            
            means.append(mean)
            logvars.append(logvar)
            
        means = torch.stack(means, dim=0)  # [n_samples, batch, output_dim]
        logvars = torch.stack(logvars, dim=0)
        
        # Compute statistics
        predictive_mean = torch.mean(means, dim=0)
        
        # Aleatoric uncertainty (average of predicted variances)
        aleatoric_var = torch.mean(torch.exp(logvars), dim=0)
        
        # Epistemic uncertainty (variance of predictions)
        epistemic_var = torch.var(means, dim=0) + torch.exp(self.log_epistemic_std)**2
        
        return predictive_mean, aleatoric_var, epistemic_var
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for variational layers."""
        if not self.use_variational:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        kl_total = (
            self.fc1.kl_divergence() +
            self.fc2.kl_divergence() +
            self.mean_head.kl_divergence() +
            self.logvar_head.kl_divergence()
        )
        
        return kl_total


class BayesianDiffusion(DiffusionModel):
    """Uncertainty-Aware Bayesian Diffusion Model.
    
    Implements principled Bayesian uncertainty quantification with:
    - Aleatoric uncertainty (data uncertainty)
    - Epistemic uncertainty (model uncertainty)  
    - Calibrated confidence intervals
    - Out-of-distribution detection
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_steps: int = 1000,
        output_dim: int = 6,
        n_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        use_variational: bool = True,
        beta_kl: float = 0.001
    ):
        """Initialize Bayesian Diffusion Model.
        
        Args:
            input_dim: Input dimensionality
            hidden_dim: Hidden layer dimensionality
            num_steps: Number of diffusion steps
            output_dim: Output parameter dimensionality
            n_mc_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty
            use_variational: Whether to use variational Bayesian layers
            beta_kl: Weight for KL divergence loss
        """
        super().__init__(input_dim, hidden_dim, num_steps)
        
        self.output_dim = output_dim
        self.n_mc_samples = n_mc_samples
        self.beta_kl = beta_kl
        
        # Replace decoder with Bayesian version
        self.bayesian_decoder = BayesianDiffusionDecoder(
            latent_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_samples=n_mc_samples,
            dropout_rate=dropout_rate,
            use_variational=use_variational
        )
        
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict with uncertainty quantification.
        
        Args:
            x: Input tensor
            condition: Conditioning information
            n_samples: Number of MC samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, uncertainty_dict)
        """
        with torch.no_grad():
            # Generate samples through full diffusion process
            latent_samples = self.sample(
                x.shape, condition=condition, 
                num_samples=n_samples or self.n_mc_samples
            )
            
            # Decode with uncertainty
            mean, aleatoric_var, epistemic_var = self.bayesian_decoder(
                latent_samples, return_uncertainty=True, n_samples=n_samples
            )
            
            # Total uncertainty
            total_var = aleatoric_var + epistemic_var
            total_std = torch.sqrt(total_var)
            
            # Confidence intervals
            z_score = torch.distributions.Normal(0, 1).icdf(
                torch.tensor((1 + confidence_level) / 2)
            )
            
            lower_bound = mean - z_score * total_std
            upper_bound = mean + z_score * total_std
            
            uncertainty_dict = {
                'aleatoric_variance': aleatoric_var,
                'epistemic_variance': epistemic_var,
                'total_variance': total_var,
                'total_std': total_std,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': confidence_level
            }
            
        return mean, uncertainty_dict
    
    def compute_uncertainty_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aleatoric_var: torch.Tensor,
        epistemic_var: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty-aware losses.
        
        Args:
            predictions: Model predictions
            targets: Target values
            aleatoric_var: Aleatoric uncertainty
            epistemic_var: Epistemic uncertainty (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Negative log-likelihood with aleatoric uncertainty
        nll = 0.5 * (
            torch.log(2 * math.pi * aleatoric_var) +
            (predictions - targets)**2 / aleatoric_var
        )
        losses['nll'] = torch.mean(nll)
        
        # KL divergence for variational layers
        if hasattr(self.bayesian_decoder, 'kl_divergence'):
            losses['kl_div'] = self.bayesian_decoder.kl_divergence()
        
        # Total variational loss
        losses['total'] = losses['nll'] + self.beta_kl * losses.get('kl_div', 0)
        
        return losses
    
    def calibration_metrics(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics for uncertainty estimates.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties (std)
            targets: Ground truth targets
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        with torch.no_grad():
            # Calculate prediction errors
            errors = torch.abs(predictions - targets)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Confidence = probability that error < uncertainty
                confidences = torch.norm((errors < uncertainties).float(), dim=-1)
                
                # Find predictions in this confidence bin
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = (errors[in_bin] < uncertainties[in_bin]).float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Prediction Interval Coverage Probability (PICP)
            # For 1-sigma intervals (68% expected coverage)
            in_interval = (errors < uncertainties).float()
            picp = torch.mean(in_interval)
            
            # Mean Prediction Interval Width (MPIW)
            mpiw = torch.mean(2 * uncertainties)  # ±1 sigma
            
            # Sharpness (negative log-likelihood)
            nll = torch.mean(
                0.5 * torch.log(2 * math.pi * uncertainties**2) +
                errors**2 / (2 * uncertainties**2)
            )
            
        return {
            'expected_calibration_error': float(ece),
            'prediction_interval_coverage': float(picp),
            'mean_interval_width': float(mpiw),
            'negative_log_likelihood': float(nll)
        }
    
    def detect_ood(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        threshold_percentile: float = 95.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect out-of-distribution samples using uncertainty.
        
        Args:
            x: Input samples
            condition: Conditioning information
            threshold_percentile: Percentile for OOD threshold
            
        Returns:
            Tuple of (ood_scores, ood_flags)
        """
        with torch.no_grad():
            _, uncertainty_dict = self.predict_with_uncertainty(x, condition)
            
            # Use total uncertainty as OOD score
            ood_scores = uncertainty_dict['total_std'].mean(dim=-1)
            
            # Threshold based on training distribution (simplified)
            threshold = torch.quantile(ood_scores, threshold_percentile / 100.0)
            ood_flags = ood_scores > threshold
            
        return ood_scores, ood_flags


class UncertaintyCalibrationTrainer:
    """Trainer for uncertainty calibration."""
    
    def __init__(
        self,
        model: BayesianDiffusion,
        calibration_data: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4
    ):
        """Initialize calibration trainer.
        
        Args:
            model: Bayesian diffusion model
            calibration_data: Calibration dataset
            learning_rate: Learning rate for calibration
        """
        self.model = model
        self.calibration_data = calibration_data
        
        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.Adam([self.temperature], lr=learning_rate)
        
    def calibrate_temperature(self, max_epochs: int = 100) -> float:
        """Calibrate temperature scaling for better uncertainty calibration.
        
        Args:
            max_epochs: Maximum calibration epochs
            
        Returns:
            Optimal temperature value
        """
        self.model.eval()
        best_nll = float('inf')
        
        for epoch in range(max_epochs):
            total_nll = 0.0
            n_batches = 0
            
            for batch in self.calibration_data:
                inputs, targets = batch['inputs'], batch['targets']
                
                # Get predictions with uncertainty
                with torch.no_grad():
                    predictions, uncertainty_dict = self.model.predict_with_uncertainty(inputs)
                    
                # Apply temperature scaling to uncertainties
                scaled_var = uncertainty_dict['total_variance'] * (self.temperature ** 2)
                
                # Compute negative log-likelihood with scaled uncertainties
                nll = torch.mean(
                    0.5 * torch.log(2 * math.pi * scaled_var) +
                    (predictions - targets)**2 / (2 * scaled_var)
                )
                
                self.optimizer.zero_grad()
                nll.backward()
                self.optimizer.step()
                
                total_nll += nll.item()
                n_batches += 1
                
            avg_nll = total_nll / n_batches
            if avg_nll < best_nll:
                best_nll = avg_nll
                
        return float(self.temperature.item())