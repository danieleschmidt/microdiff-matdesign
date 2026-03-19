"""
microdiff-matdesign: Diffusion models for inverse material microstructure design.

Generate synthetic microstructures conditioned on target material properties
using denoising diffusion probabilistic models (DDPM).
"""

from .generator import MicrostructureGenerator
from .predictor import MaterialPropertyPredictor
from .diffusion import DiffusionModel
from .designer import InverseDesigner

__version__ = "0.1.0"
__all__ = [
    "MicrostructureGenerator",
    "MaterialPropertyPredictor",
    "DiffusionModel",
    "InverseDesigner",
]
