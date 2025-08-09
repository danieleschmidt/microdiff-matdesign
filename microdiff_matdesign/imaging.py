"""Micro-CT image processing functionality."""

import os
import glob
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any

import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing
from skimage import io, filters, morphology, segmentation, measure, restoration
from skimage.morphology import disk, ball, remove_small_objects, remove_small_holes
from skimage.segmentation import watershed, chan_vese, morphological_geodesic_active_contour
from skimage.filters import threshold_otsu, threshold_multiotsu, sobel, gaussian
import warnings

from .utils.preprocessing import (
    preprocess_microct_volume, segment_phases, extract_microstructure_features,
    normalize_microstructure, denoise_volume, enhance_volume_contrast
)


class MicroCTProcessor:
    """Processor for micro-CT images and microstructure analysis."""
    
    def __init__(self, voxel_size: float = 0.5, cache_enabled: bool = True):
        """Initialize the processor.
        
        Args:
            voxel_size: Physical size of voxels in micrometers
            cache_enabled: Whether to enable caching of processed volumes
        """
        self.voxel_size = voxel_size
        self.cache_enabled = cache_enabled
        self._cache = {}
        
    def load_image(
        self,
        filepath: str,
        voxel_size: Optional[float] = None
    ) -> np.ndarray:
        """Load and process single micro-CT image file.
        
        Args:
            filepath: Path to image file (TIFF, PNG, or other formats)
            voxel_size: Voxel size override
            
        Returns:
            3D numpy array representing the microstructure
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
            
        # Check cache
        cache_key = f"image_{filepath}_{voxel_size}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        try:
            # Load image based on file extension
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                # Multi-page TIFF (common for CT data)
                image = io.imread(filepath)
                if image.ndim == 2:
                    # Single slice, create minimal 3D volume
                    image = np.expand_dims(image, axis=0)
                elif image.ndim == 3 and image.shape[0] == 1:
                    # Single slice with channel dimension
                    image = image.squeeze(0)
                    image = np.expand_dims(image, axis=0)
            else:
                # Other formats (PNG, JPG, etc.)
                image = io.imread(filepath)
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=0)
                elif image.ndim == 3 and image.shape[-1] in [3, 4]:
                    # RGB/RGBA image - convert to grayscale
                    if image.shape[-1] == 4:
                        # Remove alpha channel
                        image = image[..., :3]
                    # Convert to grayscale
                    image = np.mean(image, axis=-1)
                    image = np.expand_dims(image, axis=0)
            
            # Convert to float
            image = image.astype(np.float32)
            
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / np.max(image)
            
            # Apply preprocessing
            processed = self.preprocess(image, denoise_method="gaussian")
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = processed
                
            return processed
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {filepath}: {str(e)}")
        
    def load_volume(
        self,
        directory: str,
        file_pattern: str = "*.tif",
        slice_range: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Load volume from directory of slice images.
        
        Args:
            directory: Directory containing slice images
            file_pattern: Glob pattern for slice files
            slice_range: Optional range of slices to load (start, end)
            
        Returns:
            3D numpy array representing the volume
        """
        # Check cache
        cache_key = f"volume_{directory}_{file_pattern}_{slice_range}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all matching files
        pattern_path = directory_path / file_pattern
        files = sorted(glob.glob(str(pattern_path)))
        
        if not files:
            raise ValueError(f"No files found matching pattern {file_pattern} in {directory}")
        
        # Apply slice range if specified
        if slice_range is not None:
            start, end = slice_range
            files = files[start:end]
        
        print(f"Loading {len(files)} slices from {directory}")
        
        # Load first slice to get dimensions
        first_slice = io.imread(files[0])
        if first_slice.ndim == 3:
            # RGB image - convert to grayscale
            first_slice = np.mean(first_slice, axis=-1)
        
        height, width = first_slice.shape
        num_slices = len(files)
        
        # Initialize volume
        volume = np.zeros((num_slices, height, width), dtype=np.float32)
        
        # Load all slices
        for i, file_path in enumerate(files):
            try:
                slice_img = io.imread(file_path)
                
                # Handle different image formats
                if slice_img.ndim == 3:
                    # RGB/RGBA image
                    if slice_img.shape[-1] == 4:
                        slice_img = slice_img[..., :3]  # Remove alpha
                    slice_img = np.mean(slice_img, axis=-1)  # Convert to grayscale
                
                # Normalize
                if slice_img.max() > 1.0:
                    slice_img = slice_img.astype(np.float32) / np.max(slice_img)
                
                volume[i] = slice_img
                
            except Exception as e:
                warnings.warn(f"Failed to load slice {file_path}: {e}")
                # Use zeros for failed slices
                volume[i] = np.zeros((height, width))
        
        # Apply preprocessing
        processed = self.preprocess(volume)
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = processed
        
        return processed
        
    def preprocess(
        self,
        volume: np.ndarray,
        denoise_method: str = "bilateral",
        enhance_contrast: bool = True,
        remove_artifacts: bool = True
    ) -> np.ndarray:
        """Preprocess 3D volume with comprehensive pipeline.
        
        Args:
            volume: Input 3D volume
            denoise_method: Denoising method ('gaussian', 'median', 'bilateral', 'tv')
            enhance_contrast: Whether to enhance contrast
            remove_artifacts: Whether to remove imaging artifacts
            
        Returns:
            Preprocessed 3D volume
        """
        return preprocess_microct_volume(
            volume, 
            self.voxel_size,
            denoise=True,
            enhance_contrast=enhance_contrast,
            remove_artifacts=remove_artifacts
        )
        
    def segment_phases(
        self,
        volume: np.ndarray,
        num_phases: int = 3,
        method: str = "watershed"
    ) -> np.ndarray:
        """Segment phases in microstructure.
        
        Args:
            volume: Input 3D volume
            num_phases: Number of phases to segment
            method: Segmentation method ('watershed', 'kmeans', 'threshold', 'random_walker')
            
        Returns:
            Segmented volume with integer phase labels
        """
        return segment_phases(volume, num_phases, method)
    
    def extract_features(
        self,
        volume: np.ndarray,
        segmented: Optional[np.ndarray] = None,
        feature_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract quantitative features from microstructure.
        
        Args:
            volume: Input 3D volume
            segmented: Pre-segmented volume (if None, will segment automatically)
            feature_types: List of feature types to extract
            
        Returns:
            Dictionary of extracted features
        """
        if feature_types is None:
            feature_types = [
                "grain_size_distribution",
                "phase_fractions", 
                "texture_coefficients",
                "porosity",
                "surface_roughness"
            ]
        
        if segmented is None:
            segmented = self.segment_phases(volume)
        
        features = extract_microstructure_features(volume, segmented)
        
        # Add additional feature types if requested
        if "grain_size_distribution" in feature_types:
            grain_features = self._extract_grain_size_distribution(segmented)
            features.update(grain_features)
        
        if "texture_coefficients" in feature_types:
            texture_features = self._extract_texture_coefficients(volume)
            features.update(texture_features)
        
        if "surface_roughness" in feature_types:
            roughness_features = self._extract_surface_roughness(volume)
            features.update(roughness_features)
        
        return features
    
    def _extract_grain_size_distribution(self, segmented: np.ndarray) -> Dict[str, float]:
        """Extract grain size distribution statistics."""
        
        features = {}
        
        # Get unique phases (excluding background)
        phases = np.unique(segmented)
        phases = phases[phases > 0]
        
        for phase in phases:
            phase_mask = segmented == phase
            
            # Label connected components (grains)
            labeled_grains = morphology.label(phase_mask)
            props = measure.regionprops(labeled_grains)
            
            if props:
                # Calculate equivalent spherical diameters
                volumes = [prop.area for prop in props]  # In 2D, this is area
                # Convert to equivalent sphere diameter
                diameters = [2 * (3 * vol / (4 * np.pi)) ** (1/3) * self.voxel_size for vol in volumes]
                
                # Statistics
                features[f'phase_{phase}_mean_grain_diameter'] = np.mean(diameters)
                features[f'phase_{phase}_median_grain_diameter'] = np.median(diameters)
                features[f'phase_{phase}_std_grain_diameter'] = np.std(diameters)
                features[f'phase_{phase}_min_grain_diameter'] = np.min(diameters)
                features[f'phase_{phase}_max_grain_diameter'] = np.max(diameters)
                features[f'phase_{phase}_grain_count'] = len(diameters)
                
                # Distribution percentiles
                features[f'phase_{phase}_d10'] = np.percentile(diameters, 10)
                features[f'phase_{phase}_d50'] = np.percentile(diameters, 50)
                features[f'phase_{phase}_d90'] = np.percentile(diameters, 90)
        
        return features
    
    def _extract_texture_coefficients(self, volume: np.ndarray) -> Dict[str, float]:
        """Extract texture analysis coefficients."""
        
        features = {}
        
        # Gray Level Co-occurrence Matrix (GLCM) features
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Convert to appropriate integer range for GLCM
            volume_int = (volume * 255).astype(np.uint8)
            
            # Calculate GLCM for middle slice (representative)
            middle_slice = volume_int[volume_int.shape[0] // 2]
            
            # GLCM for different directions and distances
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(middle_slice, distances, angles, 256, symmetric=True, normed=True)
            
            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                features[f'glcm_{prop}_mean'] = np.mean(values)
                features[f'glcm_{prop}_std'] = np.std(values)
            
        except ImportError:
            warnings.warn("GLCM texture analysis requires scikit-image")
        
        # Local Binary Pattern (LBP) features
        try:
            from skimage.feature import local_binary_pattern
            
            middle_slice = volume[volume.shape[0] // 2]
            
            # LBP with different parameters
            radius = 3
            n_points = 8 * radius
            
            lbp = local_binary_pattern(middle_slice, n_points, radius, method='uniform')
            
            # LBP histogram features
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            features['lbp_uniformity'] = np.sum(hist ** 2)
            features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-7))
            
        except ImportError:
            warnings.warn("LBP texture analysis requires scikit-image")
        
        return features
    
    def _extract_surface_roughness(self, volume: np.ndarray) -> Dict[str, float]:
        """Extract surface roughness metrics."""
        
        features = {}
        
        # Calculate gradient magnitude (represents edges/surfaces)
        gradient_z = np.gradient(volume, axis=0)
        gradient_y = np.gradient(volume, axis=1)
        gradient_x = np.gradient(volume, axis=2)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
        
        # Surface roughness metrics
        features['surface_roughness_mean'] = np.mean(gradient_magnitude)
        features['surface_roughness_std'] = np.std(gradient_magnitude)
        features['surface_roughness_max'] = np.max(gradient_magnitude)
        
        # Ra (arithmetic average roughness)
        features['roughness_ra'] = np.mean(np.abs(gradient_magnitude - np.mean(gradient_magnitude)))
        
        # RMS roughness
        features['roughness_rms'] = np.sqrt(np.mean(gradient_magnitude**2))
        
        # Skewness and kurtosis of surface height distribution
        mean_height = np.mean(volume)
        centered = volume - mean_height
        std_height = np.std(volume)
        
        if std_height > 0:
            features['surface_skewness'] = np.mean((centered / std_height) ** 3)
            features['surface_kurtosis'] = np.mean((centered / std_height) ** 4)
        else:
            features['surface_skewness'] = 0.0
            features['surface_kurtosis'] = 3.0
        
        return features
    
    def analyze_porosity(
        self,
        volume: np.ndarray,
        threshold_method: str = "otsu"
    ) -> Dict[str, float]:
        """Analyze porosity in the microstructure.
        
        Args:
            volume: Input 3D volume
            threshold_method: Thresholding method ('otsu', 'manual', 'adaptive')
            
        Returns:
            Dictionary of porosity metrics
        """
        
        if threshold_method == "otsu":
            threshold = threshold_otsu(volume)
        elif threshold_method == "manual":
            threshold = 0.5  # Assume pre-normalized volume
        else:
            # Adaptive thresholding (slice by slice)
            binary_volume = np.zeros_like(volume, dtype=bool)
            for i in range(volume.shape[0]):
                slice_threshold = threshold_otsu(volume[i])
                binary_volume[i] = volume[i] < slice_threshold
        
        if threshold_method != "adaptive":
            binary_volume = volume < threshold
        
        # Remove small objects (noise)
        cleaned = remove_small_objects(binary_volume, min_size=64)
        
        # Calculate porosity metrics
        total_voxels = cleaned.size
        pore_voxels = np.sum(cleaned)
        
        porosity = pore_voxels / total_voxels
        
        # Analyze pore size distribution
        labeled_pores = morphology.label(cleaned)
        pore_props = measure.regionprops(labeled_pores)
        
        if pore_props:
            pore_volumes = [prop.area for prop in pore_props]
            
            # Convert to physical units
            pore_volumes_physical = [vol * (self.voxel_size ** 3) for vol in pore_volumes]
            
            metrics = {
                'total_porosity': porosity,
                'pore_count': len(pore_volumes),
                'mean_pore_volume': np.mean(pore_volumes_physical),
                'median_pore_volume': np.median(pore_volumes_physical),
                'max_pore_volume': np.max(pore_volumes_physical),
                'pore_volume_std': np.std(pore_volumes_physical),
                'closed_porosity_ratio': self._calculate_closed_porosity_ratio(cleaned)
            }
        else:
            metrics = {
                'total_porosity': porosity,
                'pore_count': 0,
                'mean_pore_volume': 0.0,
                'median_pore_volume': 0.0,
                'max_pore_volume': 0.0,
                'pore_volume_std': 0.0,
                'closed_porosity_ratio': 0.0
            }
        
        return metrics
    
    def _calculate_closed_porosity_ratio(self, binary_pores: np.ndarray) -> float:
        """Calculate ratio of closed to total porosity."""
        
        # Fill holes to find closed porosity
        filled = ndimage.binary_fill_holes(binary_pores)
        
        closed_pores = filled & (~binary_pores)
        total_pores = np.sum(binary_pores)
        closed_pore_volume = np.sum(closed_pores)
        
        if total_pores > 0:
            return closed_pore_volume / total_pores
        else:
            return 0.0
    
    def visualize_features(
        self,
        volume: np.ndarray,
        segmented: Optional[np.ndarray] = None,
        slice_index: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate visualization of extracted features.
        
        Args:
            volume: Input 3D volume
            segmented: Segmented volume
            slice_index: Index of slice to visualize (if None, uses middle slice)
            
        Returns:
            Dictionary of visualization arrays
        """
        
        if slice_index is None:
            slice_index = volume.shape[0] // 2
        
        if segmented is None:
            segmented = self.segment_phases(volume)
        
        # Get slice
        volume_slice = volume[slice_index]
        segmented_slice = segmented[slice_index]
        
        visualizations = {
            'original': volume_slice,
            'segmented': segmented_slice,
        }
        
        # Edge detection
        edges = sobel(volume_slice)
        visualizations['edges'] = edges
        
        # Gradient magnitude
        gy, gx = np.gradient(volume_slice)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        visualizations['gradient'] = gradient_mag
        
        return visualizations
    
    def clear_cache(self):
        """Clear the processing cache."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache."""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }