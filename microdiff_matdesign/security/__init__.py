"""Security module for MicroDiff-MatDesign.

This module provides comprehensive security features for production deployment
of diffusion models in materials science applications.

Features:
    - Adversarial input detection
    - Robustness verification 
    - Security monitoring
    - Gradient-based attack detection
"""

from .adversarial_defense import (
    AdversarialDetector,
    GradientAnalyzer,
    RobustnessVerifier,
    SecurityMonitor
)

__all__ = [
    'AdversarialDetector',
    'GradientAnalyzer',
    'RobustnessVerifier',
    'SecurityMonitor'
]