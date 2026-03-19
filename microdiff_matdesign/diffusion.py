"""
DiffusionModel: DDPM in 2D image space, conditioned on a target property vector.

Implements the denoising diffusion probabilistic model (Ho et al., 2020) with
classifier-free guidance conditioning on material property targets.

Forward process: q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
Reverse process: p_θ(x_{t-1} | x_t, c) — learned via noise prediction network.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Smoother than linear; avoids too-noisy start/end.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.9999).float()


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time + condition injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = h + self.cond_proj(c_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Lightweight U-Net noise predictor for 64×64 single-channel images.

    Encoder: 3 downsampling stages (64→32→16→8).
    Bottleneck: 1 ResBlock at 8×8.
    Decoder: 3 upsampling stages with skip connections.
    Conditioning: timestep + property vector injected at every ResBlock.
    """

    def __init__(self, image_channels: int = 1, base_ch: int = 32, n_properties: int = 2):
        super().__init__()
        time_dim = base_ch * 4
        cond_dim = base_ch * 2

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_properties, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c = base_ch  # channel multiplier

        # Encoder
        self.init_conv = nn.Conv2d(image_channels, c, 3, padding=1)
        self.enc1 = ResBlock(c, c, time_dim, cond_dim)
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)

        self.enc2 = ResBlock(c * 2, c * 2, time_dim, cond_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)

        self.enc3 = ResBlock(c * 4, c * 4, time_dim, cond_dim)
        self.down3 = nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResBlock(c * 4, c * 4, time_dim, cond_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 4, stride=2, padding=1)
        self.dec3 = ResBlock(c * 8, c * 4, time_dim, cond_dim)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(c * 4, c * 2, time_dim, cond_dim)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = ResBlock(c * 2, c, time_dim, cond_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, image_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) — noisy image at timestep t
        t : (B,) — timestep indices
        conditions : (B, n_properties) — target property vector in [0, 1]

        Returns
        -------
        (B, 1, H, W) — predicted noise ε
        """
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(conditions)

        h0 = self.init_conv(x)
        h1 = self.enc1(h0, t_emb, c_emb)
        h2 = self.enc2(self.down1(h1), t_emb, c_emb)
        h3 = self.enc3(self.down2(h2), t_emb, c_emb)

        hb = self.bottleneck(self.down3(h3), t_emb, c_emb)

        d3 = self.dec3(torch.cat([self.up3(hb), h3], dim=1), t_emb, c_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), h2], dim=1), t_emb, c_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1), t_emb, c_emb)

        return self.final(d1)


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class DiffusionModel:
    """
    DDPM diffusion model for conditioned microstructure generation.

    Wraps a U-Net noise predictor with the standard DDPM forward/reverse
    process and classifier-free guidance for property conditioning.

    Parameters
    ----------
    image_size : int
        Spatial resolution (square images assumed).
    T : int
        Number of diffusion timesteps.
    n_properties : int
        Dimensionality of the conditioning property vector.
    device : str
        'cuda' or 'cpu'.
    guidance_scale : float
        Classifier-free guidance scale (1.0 = no guidance, 3–7 = typical).
    """

    def __init__(
        self,
        image_size: int = 64,
        T: int = 200,
        n_properties: int = 2,
        device: str = "cpu",
        guidance_scale: float = 3.0,
    ):
        self.image_size = image_size
        self.T = T
        self.n_properties = n_properties
        self.device = torch.device(device)
        self.guidance_scale = guidance_scale

        # Noise schedule
        betas = cosine_beta_schedule(T).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

        # Network
        self.net = UNet(n_properties=n_properties).to(self.device)

    def register(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean image x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        xt = sqrt_a * x0 + sqrt_1ma * noise
        return xt, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x0: torch.Tensor,
        conditions: torch.Tensor,
        uncond_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Simple DDPM noise-prediction loss with classifier-free guidance dropout.

        With probability `uncond_prob`, the condition is zeroed out so the
        model also learns unconditional generation.

        Parameters
        ----------
        x0 : (B, 1, H, W)
        conditions : (B, n_properties)
        uncond_prob : float
            Fraction of training steps where condition is dropped (→ unconditional).
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, noise = self.q_sample(x0, t)

        # Classifier-free guidance dropout
        if uncond_prob > 0:
            mask = (torch.rand(B, device=self.device) > uncond_prob).float()
            cond_in = conditions * mask[:, None]
        else:
            cond_in = conditions

        noise_pred = self.net(xt, t, cond_in)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Sampling (reverse process)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: int,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse-process step with classifier-free guidance."""
        t_tensor = torch.full((xt.size(0),), t, device=self.device, dtype=torch.long)

        # Conditional + unconditional forward pass
        uncond = torch.zeros_like(conditions)
        noise_pred_cond = self.net(xt, t_tensor, conditions)
        noise_pred_uncond = self.net(xt, t_tensor, uncond)

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        beta_t = self.betas[t]
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = (1.0 / (1.0 - beta_t).sqrt())

        mean = sqrt_recip_alpha * (xt - beta_t / sqrt_one_minus_ac * noise_pred)

        if t > 0:
            variance = self.posterior_variance[t]
            noise = torch.randn_like(xt)
            return mean + variance.sqrt() * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        conditions: np.ndarray | torch.Tensor,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """
        Generate microstructure images from target property conditions.

        Parameters
        ----------
        conditions : array-like (B, n_properties) or (n_properties,)
            Target property values in [0, 1].
        n_steps : int or None
            Number of reverse diffusion steps (defaults to full T).

        Returns
        -------
        np.ndarray
            Shape (B, 1, H, W), values clipped to [0, 1].
        """
        self.net.eval()
        if isinstance(conditions, np.ndarray):
            conditions = torch.from_numpy(conditions).float()
        if conditions.ndim == 1:
            conditions = conditions.unsqueeze(0)
        conditions = conditions.to(self.device)
        B = conditions.size(0)

        steps = n_steps or self.T
        step_indices = list(range(self.T - 1, -1, -1))
        if steps < self.T:
            stride = self.T // steps
            step_indices = step_indices[::stride]

        x = torch.randn(B, 1, self.image_size, self.image_size, device=self.device)
        for t in step_indices:
            x = self.p_sample(x, t, conditions)

        x = (x.clamp(-1, 1) + 1) / 2  # map [-1,1] → [0,1]
        return x.cpu().numpy()

    # ------------------------------------------------------------------
    # Quick training loop for demo
    # ------------------------------------------------------------------

    def train_quick(
        self,
        dataset: np.ndarray,
        conditions: np.ndarray,
        n_epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> list[float]:
        """
        Minimal training loop — useful for smoke-test and demos.

        Parameters
        ----------
        dataset : (N, 1, H, W) float32 microstructure images in [0, 1]
        conditions : (N, n_properties) float32 property vectors
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        N = dataset.shape[0]
        losses = []

        for epoch in range(n_epochs):
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, N, batch_size):
                batch_idx = idx[start: start + batch_size]
                x0 = torch.from_numpy(dataset[batch_idx]).to(self.device)
                # Map [0,1] → [-1,1] (standard DDPM convention)
                x0 = x0 * 2 - 1
                cond = torch.from_numpy(conditions[batch_idx]).to(self.device)
                optimizer.zero_grad()
                loss = self.loss(x0, cond)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            avg = epoch_loss / steps
            losses.append(avg)
            if verbose:
                print(f"  Epoch {epoch + 1}/{n_epochs}  loss={avg:.4f}")
        return losses
