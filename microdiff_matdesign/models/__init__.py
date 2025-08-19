"""Neural network models for diffusion-based inverse design."""

from .diffusion import DiffusionModel, DiffusionUNet3D, DiffusionTransformer
from .encoders import MicrostructureEncoder, FeatureExtractor
from .decoders import ParameterDecoder
from .conditioning import ConditionalDiffusion
from .quantum_enhanced import (
    QuantumEnhancedDiffusion, QuantumAdaptiveDiffusion, 
    QuantumAttentionMechanism, QuantumMaterialsOptimizer
)
from .consciousness_aware import (
    ConsciousnessDrivenDiffusion, SelfAwarenessModule, 
    CreativeInsightGenerator, ConsciousMaterialsExplorer
)
from .adaptive_intelligence import (
    AdaptiveIntelligenceSystem, NeuralPlasticityModule,
    MetaLearningController, AdaptiveNeuralArchitecture
)

__all__ = [
    "DiffusionModel",
    "DiffusionUNet3D", 
    "DiffusionTransformer",
    "MicrostructureEncoder",
    "FeatureExtractor",
    "ParameterDecoder",
    "ConditionalDiffusion",
    "QuantumEnhancedDiffusion",
    "QuantumAdaptiveDiffusion",
    "QuantumAttentionMechanism",
    "QuantumMaterialsOptimizer",
    "ConsciousnessDrivenDiffusion",
    "SelfAwarenessModule",
    "CreativeInsightGenerator",
    "ConsciousMaterialsExplorer",
    "AdaptiveIntelligenceSystem",
    "NeuralPlasticityModule",
    "MetaLearningController",
    "AdaptiveNeuralArchitecture",
]