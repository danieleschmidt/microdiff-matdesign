"""Preprocessing utilities for microstructures and parameters."""

from typing import Tuple, Optional, Union, Dict, Any, List
import numpy as np
import warnings
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology, filters, segmentation, measure
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral


def normalize_microstructure(microstructure: np.ndarray, 
                            method: str = "standardize",
                            clip_percentiles: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Normalize microstructure data for model input."""
    
    # Ensure float type
    if not np.issubdtype(microstructure.dtype, np.floating):
        microstructure = microstructure.astype(np.float32)
    
    # Handle different normalization methods
    if method == "standardize":
        # Z-score normalization
        mean = np.mean(microstructure)
        std = np.std(microstructure)
        
        if std < 1e-8:
            warnings.warn("Very small standard deviation - using min-max normalization instead")
            method = "minmax"
        else:
            normalized = (microstructure - mean) / std
    
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(microstructure)
        max_val = np.max(microstructure)
        
        if max_val - min_val < 1e-8:
            warnings.warn("Constant microstructure values detected")
            normalized = np.zeros_like(microstructure)
        else:
            normalized = (microstructure - min_val) / (max_val - min_val)
    
    elif method == "robust":
        # Robust normalization using percentiles
        p_low, p_high = np.percentile(microstructure, clip_percentiles)
        
        if p_high - p_low < 1e-8:
            warnings.warn("Very small percentile range - using standardization")
            return normalize_microstructure(microstructure, "standardize")
        
        # Clip outliers
        clipped = np.clip(microstructure, p_low, p_high)
        
        # Normalize to [0, 1]
        normalized = (clipped - p_low) / (p_high - p_low)
    
    elif method == "zscore_robust":
        # Robust z-score using median and MAD
        median = np.median(microstructure)
        mad = np.median(np.abs(microstructure - median))
        
        if mad < 1e-8:
            warnings.warn("Very small MAD - using standard normalization")
            return normalize_microstructure(microstructure, "standardize")
        
        normalized = (microstructure - median) / (1.4826 * mad)  # 1.4826 for normal distribution
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def denormalize_parameters(normalized_params: np.ndarray, 
                          param_means: Optional[np.ndarray] = None,
                          param_stds: Optional[np.ndarray] = None) -> np.ndarray:
    """Denormalize parameters from model output to physical values."""
    
    # Default parameter statistics for typical LPBF processes
    if param_means is None:
        param_means = np.array([200.0, 800.0, 30.0, 120.0, 80.0])  # Power, speed, layer, hatch, temp
    
    if param_stds is None:
        param_stds = np.array([50.0, 200.0, 10.0, 30.0, 20.0])
    
    # Ensure correct shape
    if normalized_params.ndim == 1:
        normalized_params = normalized_params.reshape(1, -1)
    
    # Trim to available statistics
    num_params = min(normalized_params.shape[-1], len(param_means))
    
    # Denormalize
    denormalized = normalized_params[..., :num_params] * param_stds[:num_params] + param_means[:num_params]
    
    return denormalized


def preprocess_microct_volume(volume: np.ndarray,
                             voxel_size: float = 0.5,
                             denoise: bool = True,
                             enhance_contrast: bool = True,
                             remove_artifacts: bool = True) -> np.ndarray:
    """Comprehensive preprocessing of micro-CT volumes."""
    
    processed = volume.copy().astype(np.float32)
    
    # Remove artifacts (ring artifacts, beam hardening)
    if remove_artifacts:
        processed = remove_ct_artifacts(processed)
    
    # Denoise
    if denoise:
        processed = denoise_volume(processed, method="bilateral")
    
    # Enhance contrast
    if enhance_contrast:
        processed = enhance_volume_contrast(processed)
    
    # Normalize
    processed = normalize_microstructure(processed, method="robust")
    
    return processed


def denoise_volume(volume: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Denoise 3D volume using various methods."""
    
    if method == "gaussian":
        return gaussian_filter(volume, sigma=1.0)
    
    elif method == "median":
        return median_filter(volume, size=3)
    
    elif method == "bilateral":
        # Apply bilateral filter slice by slice (3D bilateral is computationally expensive)
        denoised = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            denoised[i] = denoise_bilateral(volume[i], sigma_color=0.1, sigma_spatial=1.0)
        return denoised
    
    elif method == "tv":
        # Total variation denoising
        return denoise_tv_chambolle(volume, weight=0.1)
    
    elif method == "morphological":
        # Morphological denoising
        # Opening followed by closing
        selem = morphology.ball(1)
        opened = morphology.opening(volume, selem)
        return morphology.closing(opened, selem)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def enhance_volume_contrast(volume: np.ndarray, method: str = "gamma") -> np.ndarray:
    """Enhance contrast in 3D volumes."""
    
    if method == "clahe":
        # Contrast Limited Adaptive Histogram Equalization
        # Apply slice by slice
        enhanced = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            # Ensure volume slice is in correct range for CLAHE
            slice_data = volume[i]
            if slice_data.max() > 1.0 or slice_data.min() < 0.0:
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            try:
                enhanced[i] = filters.rank.enhance_contrast(slice_data, morphology.disk(30))
            except (ValueError, ImportError):
                # Fallback to simple contrast enhancement if filters.rank not available
                enhanced[i] = np.clip(slice_data * 1.2, 0, 1)
        return enhanced
    
    elif method == "histogram_eq":
        # Histogram equalization
        enhanced = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            slice_data = volume[i]
            if slice_data.max() > 1.0 or slice_data.min() < 0.0:
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            try:
                enhanced[i] = filters.rank.equalize(slice_data, morphology.disk(30))
            except (ValueError, ImportError):
                # Fallback to histogram equalization
                enhanced[i] = slice_data
        return enhanced
    
    elif method == "gamma":
        # Gamma correction
        gamma = 0.7
        return np.power(volume / np.max(volume), gamma) * np.max(volume)
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


def remove_ct_artifacts(volume: np.ndarray) -> np.ndarray:
    """Remove common CT artifacts."""
    
    # Remove ring artifacts (simplified approach)
    # In practice, more sophisticated methods like Fourier-based filtering would be used
    corrected = volume.copy()
    
    # Apply median filter in polar coordinates to reduce ring artifacts
    for i in range(volume.shape[0]):
        slice_2d = volume[i]
        
        # Simple ring artifact reduction using median filtering
        median_filtered = median_filter(slice_2d, size=3)
        
        # Blend original and filtered based on local variance
        local_var = ndimage.generic_filter(slice_2d, np.var, size=5)
        blend_factor = np.clip(local_var / np.mean(local_var), 0, 1)
        
        corrected[i] = blend_factor * slice_2d + (1 - blend_factor) * median_filtered
    
    return corrected


def segment_phases(volume: np.ndarray, 
                  num_phases: int = 3,
                  method: str = "watershed") -> np.ndarray:
    """Segment microstructure into distinct phases."""
    
    if method == "threshold":
        # Multi-level thresholding
        thresholds = filters.threshold_multiotsu(volume, classes=num_phases)
        segmented = np.digitize(volume, thresholds)
    
    elif method == "watershed":
        # Watershed segmentation
        # Compute gradient
        gradient = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            gradient[i] = filters.sobel(volume[i])
        
        # Find local minima as markers
        markers = np.zeros_like(volume, dtype=int)
        local_minima = morphology.local_minima(volume)
        markers[local_minima] = np.arange(1, np.sum(local_minima) + 1)
        
        # Watershed
        segmented = segmentation.watershed(gradient, markers)
        
        # Reduce to desired number of phases
        if np.max(segmented) > num_phases:
            # Merge similar regions
            segmented = merge_similar_regions(segmented, volume, num_phases)
    
    elif method == "kmeans":
        # K-means clustering
        from sklearn.cluster import KMeans
        
        # Reshape for clustering
        original_shape = volume.shape
        flattened = volume.reshape(-1, 1)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=num_phases, random_state=42)
        labels = kmeans.fit_predict(flattened)
        
        # Reshape back
        segmented = labels.reshape(original_shape)
    
    elif method == "random_walker":
        # Random walker segmentation (requires scikit-image)
        try:
            from skimage.segmentation import random_walker
            
            # Create markers
            markers = np.zeros_like(volume, dtype=int)
            
            # Use intensity-based markers
            for phase in range(1, num_phases + 1):
                percentile = phase * 100 / (num_phases + 1)
                threshold = np.percentile(volume, percentile)
                
                # Create sparse markers
                mask = (volume > threshold - 0.1) & (volume < threshold + 0.1)
                indices = np.where(mask)
                
                # Sample subset of points as markers
                sample_size = min(100, len(indices[0]))
                if sample_size > 0:
                    sample_indices = np.random.choice(len(indices[0]), sample_size, replace=False)
                    markers[indices[0][sample_indices], 
                           indices[1][sample_indices], 
                           indices[2][sample_indices]] = phase
            
            segmented = random_walker(volume, markers)
            
        except ImportError:
            warnings.warn("Random walker requires scikit-image, falling back to watershed")
            return segment_phases(volume, num_phases, "watershed")
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return segmented.astype(np.int32)


def merge_similar_regions(segmented: np.ndarray, intensity: np.ndarray, 
                         target_phases: int) -> np.ndarray:
    """Merge similar regions to reduce number of phases."""
    
    unique_labels = np.unique(segmented)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    if len(unique_labels) <= target_phases:
        return segmented
    
    # Compute mean intensity for each region
    region_intensities = []
    for label in unique_labels:
        mask = segmented == label
        mean_intensity = np.mean(intensity[mask])
        region_intensities.append(mean_intensity)
    
    # Hierarchical clustering of regions by intensity
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    # Compute pairwise distances
    distances = pdist(np.array(region_intensities).reshape(-1, 1))
    linkage_matrix = linkage(distances, method='ward')
    
    # Get clusters
    clusters = fcluster(linkage_matrix, target_phases, criterion='maxclust')
    
    # Create new segmentation
    merged = np.zeros_like(segmented)
    for i, label in enumerate(unique_labels):
        mask = segmented == label
        merged[mask] = clusters[i]
    
    return merged


def extract_microstructure_features(volume: np.ndarray, 
                                   segmented: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Extract quantitative microstructure features."""
    
    if segmented is None:
        segmented = segment_phases(volume, num_phases=3)
    
    features = {}
    
    # Phase fractions
    unique_phases = np.unique(segmented)
    unique_phases = unique_phases[unique_phases > 0]
    
    total_volume = np.sum(segmented > 0)
    for phase in unique_phases:
        phase_volume = np.sum(segmented == phase)
        features[f'phase_{phase}_fraction'] = phase_volume / total_volume
    
    # Porosity (assuming phase 0 is pores/voids)
    porosity = np.sum(segmented == 0) / segmented.size
    features['porosity'] = porosity
    
    # Grain size analysis (simplified)
    if len(unique_phases) > 0:
        # Use largest phase as matrix
        largest_phase = unique_phases[np.argmax([np.sum(segmented == p) for p in unique_phases])]
        matrix_mask = segmented == largest_phase
        
        # Connected component analysis for grain size
        labeled_grains = morphology.label(matrix_mask)
        props = measure.regionprops(labeled_grains)
        
        if props:
            grain_areas = [prop.area for prop in props]
            features['mean_grain_size'] = np.mean(grain_areas)
            features['grain_size_std'] = np.std(grain_areas)
            features['num_grains'] = len(grain_areas)
    
    # Surface roughness (simplified - based on gradient)
    gradient_magnitude = np.sqrt(
        np.sum([ndimage.sobel(volume, axis=i)**2 for i in range(3)], axis=0)
    )
    features['surface_roughness'] = np.mean(gradient_magnitude)
    
    # Connectivity analysis
    features['euler_number'] = measure_euler_number(segmented > 0)
    
    return features


def measure_euler_number(binary_volume: np.ndarray) -> float:
    """Compute Euler number as a measure of connectivity."""
    
    # Simplified Euler number calculation
    # In practice, would use more sophisticated topological analysis
    
    # Count connected components
    labeled = morphology.label(binary_volume)
    num_components = np.max(labeled)
    
    # Estimate cavities (simplified)
    inverted = ~binary_volume
    labeled_cavities = morphology.label(inverted)
    num_cavities = np.max(labeled_cavities) - 1  # Subtract exterior
    
    # Simplified Euler number
    euler = num_components - num_cavities
    
    return float(euler)


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int],
                 method: str = "linear") -> np.ndarray:
    """Resize 3D volume to target shape."""
    
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_factors = [target / current for target, current in zip(target_shape, volume.shape)]
    
    if method == "linear":
        order = 1
    elif method == "cubic":
        order = 3
    elif method == "nearest":
        order = 0
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Apply zoom
    resized = zoom(volume, zoom_factors, order=order)
    
    return resized


def augment_microstructure(volume: np.ndarray, 
                          rotation_angles: Tuple[float, float, float] = (0, 0, 0),
                          flip_axes: Optional[List[int]] = None,
                          noise_level: float = 0.0) -> np.ndarray:
    """Apply data augmentation to microstructure."""
    
    augmented = volume.copy()
    
    # Rotation
    if any(angle != 0 for angle in rotation_angles):
        for axis, angle in enumerate(rotation_angles):
            if angle != 0:
                augmented = ndimage.rotate(augmented, angle, axes=(axis, (axis+1)%3), 
                                         reshape=False, order=1)
    
    # Flipping
    if flip_axes:
        for axis in flip_axes:
            if axis < augmented.ndim:
                augmented = np.flip(augmented, axis=axis)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(augmented), augmented.shape)
        augmented = augmented + noise
    
    return augmented