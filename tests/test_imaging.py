"""Tests for imaging functionality."""

import pytest
import numpy as np
from microdiff_matdesign.imaging import MicroCTProcessor


class TestMicroCTProcessor:
    """Test cases for MicroCTProcessor class."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = MicroCTProcessor()
        assert processor is not None
        
    def test_load_image(self):
        """Test image loading."""
        processor = MicroCTProcessor()
        image = processor.load_image("dummy_path.tif")
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        
    def test_load_volume(self):
        """Test volume loading."""
        processor = MicroCTProcessor()
        volume = processor.load_volume("dummy_dir/")
        
        assert isinstance(volume, np.ndarray)
        assert len(volume.shape) == 3
        
    def test_preprocess(self):
        """Test volume preprocessing."""
        processor = MicroCTProcessor()
        volume = np.random.rand(32, 32, 32)
        
        processed = processor.preprocess(volume)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == volume.shape
        
    def test_segment_phases(self):
        """Test phase segmentation."""
        processor = MicroCTProcessor()
        volume = np.random.rand(32, 32, 32)
        
        phases = processor.segment_phases(volume, num_phases=3)
        
        assert isinstance(phases, np.ndarray)
        assert phases.shape == volume.shape
        assert phases.dtype == int