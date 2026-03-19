"""
MaterialPropertyPredictor: CNN that predicts material properties from microstructure images.

Properties predicted (as normalized proxy values in [0, 1]):
  - hardness: proxy for grain-boundary density and phase hardness (Hall-Petch effect)
  - conductivity: proxy inversely related to grain boundaries and secondary phase fraction
"""

import torch
import torch.nn as nn
import numpy as np


class MaterialPropertyPredictor(nn.Module):
    """
    Small convolutional neural network that predicts two material property proxies
    from a single-channel 2D microstructure image.

    Architecture: 3-layer conv backbone → global average pool → MLP head.
    Lightweight enough to run on CPU during demo.

    Parameters
    ----------
    image_size : int
        Spatial dimension of input images (assumed square).
    n_properties : int
        Number of property outputs. Default = 2 (hardness, conductivity).
    """

    def __init__(self, image_size: int = 64, n_properties: int = 2):
        super().__init__()
        self.image_size = image_size
        self.n_properties = n_properties

        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → (B, 64, 1, 1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_properties),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, n_properties), values in [0, 1].
        """
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def predict_numpy(self, images: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper accepting numpy arrays.

        Parameters
        ----------
        images : np.ndarray
            Shape (B, 1, H, W) or (1, H, W) for a single image.

        Returns
        -------
        np.ndarray
            Shape (B, n_properties).
        """
        if images.ndim == 3:
            images = images[np.newaxis]
        x = torch.from_numpy(images).float()
        self.eval()
        return self(x).cpu().numpy()

    # ------------------------------------------------------------------
    # Physics-inspired heuristic for untrained model warm-start
    # ------------------------------------------------------------------

    @staticmethod
    def heuristic_properties(image: np.ndarray) -> dict[str, float]:
        """
        Compute interpretable property proxies directly from image statistics.
        Useful for validating that generated microstructures have expected trends.

        Physics basis:
          - Hardness ∝ grain boundary density (Hall-Petch) + secondary phase fraction
          - Conductivity ∝ 1 − (grain boundary density + secondary phase fraction)

        Parameters
        ----------
        image : np.ndarray
            Shape (1, H, W) or (H, W), values in [0, 1].

        Returns
        -------
        dict with keys 'hardness' and 'conductivity', each in [0, 1].
        """
        img = image.squeeze()
        # Grain boundary pixels are near 0.5; secondary phase near 1.0
        boundary_density = float(np.mean((img > 0.35) & (img < 0.65)))
        phase2_fraction = float(np.mean(img > 0.65))
        disorder = boundary_density + 0.5 * phase2_fraction
        hardness = float(np.clip(disorder * 2.5, 0.0, 1.0))
        conductivity = float(np.clip(1.0 - disorder * 2.0, 0.0, 1.0))
        return {"hardness": hardness, "conductivity": conductivity}
