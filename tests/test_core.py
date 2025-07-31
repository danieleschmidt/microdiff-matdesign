"""Tests for core diffusion functionality."""

import pytest
import numpy as np
from microdiff_matdesign.core import MicrostructureDiffusion


class TestMicrostructureDiffusion:
    """Test cases for MicrostructureDiffusion class."""
    
    def test_init(self):
        """Test model initialization."""
        model = MicrostructureDiffusion()
        assert model.alloy == "Ti-6Al-4V"
        assert model.process == "laser_powder_bed_fusion"
        assert model.pretrained is True
        
    def test_custom_init(self):
        """Test model initialization with custom parameters."""
        model = MicrostructureDiffusion(
            alloy="Inconel 718",
            process="electron_beam_melting",
            pretrained=False
        )
        assert model.alloy == "Inconel 718"
        assert model.process == "electron_beam_melting"
        assert model.pretrained is False
        
    def test_inverse_design(self):
        """Test inverse design functionality."""
        model = MicrostructureDiffusion()
        microstructure = np.random.rand(64, 64, 64)
        
        params = model.inverse_design(microstructure)
        
        assert isinstance(params, dict)
        assert "laser_power" in params
        assert "scan_speed" in params
        assert "layer_thickness" in params
        assert "hatch_spacing" in params