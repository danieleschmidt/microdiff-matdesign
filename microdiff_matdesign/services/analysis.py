"""Analysis service for microstructure and parameter analysis."""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import warnings
from dataclasses import dataclass
from pathlib import Path

from ..core import ProcessParameters
from ..imaging import MicroCTProcessor
from ..utils.preprocessing import extract_microstructure_features


@dataclass
class AnalysisReport:
    """Container for analysis results."""
    microstructure_features: Dict[str, float]
    parameter_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    warnings: List[str]
    confidence_scores: Dict[str, float]


class AnalysisService:
    """Service for comprehensive microstructure and parameter analysis."""
    
    def __init__(self, processor: Optional[MicroCTProcessor] = None):
        """Initialize analysis service.
        
        Args:
            processor: MicroCT processor for image analysis
        """
        self.processor = processor or MicroCTProcessor()
        self._feature_extractors = {}
        self._quality_assessors = {}
        self._parameter_validators = {}
        
        # Initialize analysis modules
        self._initialize_feature_extractors()
        self._initialize_quality_assessors()
        self._initialize_parameter_validators()
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction methods."""
        
        self._feature_extractors = {
            'grain_analysis': self._analyze_grain_structure,
            'phase_analysis': self._analyze_phase_distribution,
            'porosity_analysis': self._analyze_porosity_detailed,
            'texture_analysis': self._analyze_texture_features,
            'defect_analysis': self._analyze_defects,
            'surface_analysis': self._analyze_surface_characteristics
        }
    
    def _initialize_quality_assessors(self):
        """Initialize quality assessment methods."""
        
        self._quality_assessors = {
            'density_quality': self._assess_density_quality,
            'microstructure_quality': self._assess_microstructure_quality,
            'defect_severity': self._assess_defect_severity,
            'uniformity': self._assess_uniformity,
            'printability': self._assess_printability
        }
    
    def _initialize_parameter_validators(self):
        """Initialize parameter validation methods."""
        
        self._parameter_validators = {
            'energy_density': self._validate_energy_density,
            'thermal_history': self._validate_thermal_history,
            'melt_pool': self._validate_melt_pool_characteristics,
            'layer_bonding': self._validate_layer_bonding,
            'scan_strategy': self._validate_scan_strategy
        }
    
    def analyze_microstructure(self, 
                             volume: np.ndarray,
                             parameters: Optional[ProcessParameters] = None,
                             analysis_types: Optional[List[str]] = None) -> AnalysisReport:
        """Comprehensive microstructure analysis.
        
        Args:
            volume: 3D microstructure volume
            parameters: Process parameters used to create the microstructure
            analysis_types: Types of analysis to perform
            
        Returns:
            Comprehensive analysis report
        """
        
        if analysis_types is None:
            analysis_types = list(self._feature_extractors.keys())
        
        # Extract basic features
        microstructure_features = self.processor.extract_features(volume)
        
        # Perform specialized analyses
        for analysis_type in analysis_types:
            if analysis_type in self._feature_extractors:
                try:
                    specialized_features = self._feature_extractors[analysis_type](volume)
                    microstructure_features.update(specialized_features)
                except Exception as e:
                    warnings.warn(f"Failed to perform {analysis_type}: {str(e)}")
        
        # Parameter analysis
        parameter_analysis = {}
        if parameters is not None:
            parameter_analysis = self._analyze_parameters(parameters, volume)
        
        # Quality assessment
        quality_metrics = self._assess_quality(volume, microstructure_features, parameters)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            microstructure_features, parameter_analysis, quality_metrics
        )
        
        # Confidence scoring
        confidence_scores = self._compute_confidence_scores(
            microstructure_features, quality_metrics
        )
        
        # Collect warnings
        analysis_warnings = self._collect_warnings(
            microstructure_features, parameter_analysis, quality_metrics
        )
        
        return AnalysisReport(
            microstructure_features=microstructure_features,
            parameter_analysis=parameter_analysis,
            quality_metrics=quality_metrics,
            recommendations=recommendations,
            warnings=analysis_warnings,
            confidence_scores=confidence_scores
        )
    
    def compare_microstructures(self, 
                              volumes: List[np.ndarray],
                              labels: Optional[List[str]] = None,
                              parameters: Optional[List[ProcessParameters]] = None) -> Dict[str, Any]:
        """Compare multiple microstructures.
        
        Args:
            volumes: List of 3D microstructure volumes
            labels: Labels for each volume
            parameters: Process parameters for each volume
            
        Returns:
            Comparison analysis results
        """
        
        if labels is None:
            labels = [f"Sample_{i+1}" for i in range(len(volumes))]
        
        # Analyze each microstructure
        analyses = []
        for i, volume in enumerate(volumes):
            params = parameters[i] if parameters and i < len(parameters) else None
            analysis = self.analyze_microstructure(volume, params)
            analyses.append(analysis)
        
        # Compare features
        feature_comparison = self._compare_features(analyses, labels)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(analyses, labels)
        
        # Ranking
        ranking = self._rank_microstructures(analyses, labels)
        
        return {
            'individual_analyses': {label: analysis for label, analysis in zip(labels, analyses)},
            'feature_comparison': feature_comparison,
            'statistical_analysis': statistical_analysis,
            'ranking': ranking,
            'summary': self._generate_comparison_summary(analyses, labels)
        }
    
    def _analyze_grain_structure(self, volume: np.ndarray) -> Dict[str, float]:
        """Detailed grain structure analysis."""
        
        features = {}
        
        # Segment phases for grain analysis
        segmented = self.processor.segment_phases(volume, num_phases=3)
        
        # Focus on the matrix phase (usually the largest)
        unique_phases = np.unique(segmented)
        unique_phases = unique_phases[unique_phases > 0]
        
        if len(unique_phases) > 0:
            # Find matrix phase (largest by volume)
            phase_volumes = [np.sum(segmented == phase) for phase in unique_phases]
            matrix_phase = unique_phases[np.argmax(phase_volumes)]
            
            matrix_mask = segmented == matrix_phase
            
            # Grain size analysis using morphological operations
            from skimage import morphology, measure
            
            # Clean up the mask
            cleaned = morphology.remove_small_objects(matrix_mask, min_size=100)
            filled = morphology.remove_small_holes(cleaned, area_threshold=50)
            
            # Label grains
            labeled_grains = morphology.label(filled)
            grain_props = measure.regionprops(labeled_grains)
            
            if grain_props:
                # Extract grain metrics
                grain_areas = [prop.area for prop in grain_props]
                grain_perimeters = [prop.perimeter for prop in grain_props]
                grain_eccentricities = [prop.eccentricity for prop in grain_props]
                grain_solidities = [prop.solidity for prop in grain_props]
                
                # Convert to physical units
                voxel_volume = self.processor.voxel_size ** 3
                grain_volumes = [area * voxel_volume for area in grain_areas]
                
                # Equivalent sphere diameters
                grain_diameters = [(6 * vol / np.pi) ** (1/3) for vol in grain_volumes]
                
                # Grain statistics
                features.update({
                    'grain_count': len(grain_props),
                    'mean_grain_diameter': np.mean(grain_diameters),
                    'median_grain_diameter': np.median(grain_diameters),
                    'std_grain_diameter': np.std(grain_diameters),
                    'grain_diameter_cv': np.std(grain_diameters) / np.mean(grain_diameters),
                    'min_grain_diameter': np.min(grain_diameters),
                    'max_grain_diameter': np.max(grain_diameters),
                    'grain_size_d10': np.percentile(grain_diameters, 10),
                    'grain_size_d50': np.percentile(grain_diameters, 50),
                    'grain_size_d90': np.percentile(grain_diameters, 90),
                    'mean_grain_eccentricity': np.mean(grain_eccentricities),
                    'mean_grain_solidity': np.mean(grain_solidities),
                    'grain_density': len(grain_props) / (np.sum(matrix_mask) * voxel_volume)
                })
                
                # Grain boundary analysis
                features.update(self._analyze_grain_boundaries(labeled_grains))
        
        return features
    
    def _analyze_grain_boundaries(self, labeled_grains: np.ndarray) -> Dict[str, float]:
        """Analyze grain boundary characteristics."""
        
        features = {}
        
        # Find grain boundaries
        from skimage import segmentation
        boundaries = segmentation.find_boundaries(labeled_grains, mode='thick')
        
        # Grain boundary density
        total_volume = labeled_grains.size
        boundary_volume = np.sum(boundaries)
        
        features['grain_boundary_density'] = boundary_volume / total_volume
        
        # Analyze boundary connectivity
        from skimage import morphology
        boundary_skeleton = morphology.skeletonize_3d(boundaries)
        
        features['grain_boundary_connectivity'] = np.sum(boundary_skeleton) / np.sum(boundaries)
        
        return features
    
    def _analyze_phase_distribution(self, volume: np.ndarray) -> Dict[str, float]:
        """Analyze phase distribution and morphology."""
        
        features = {}
        
        # Segment phases
        segmented = self.processor.segment_phases(volume, num_phases=4)
        
        unique_phases = np.unique(segmented)
        unique_phases = unique_phases[unique_phases > 0]
        
        total_volume = np.sum(segmented > 0)
        
        for phase in unique_phases:
            phase_mask = segmented == phase
            phase_volume = np.sum(phase_mask)
            
            if phase_volume > 0:
                # Basic phase metrics
                phase_fraction = phase_volume / total_volume
                features[f'phase_{phase}_fraction'] = phase_fraction
                
                # Phase morphology
                from skimage import measure, morphology
                
                labeled_regions = morphology.label(phase_mask)
                regions = measure.regionprops(labeled_regions)
                
                if regions:
                    # Shape analysis
                    eccentricities = [region.eccentricity for region in regions]
                    solidities = [region.solidity for region in regions]
                    aspect_ratios = [region.major_axis_length / (region.minor_axis_length + 1e-6) 
                                   for region in regions]
                    
                    features.update({
                        f'phase_{phase}_mean_eccentricity': np.mean(eccentricities),
                        f'phase_{phase}_mean_solidity': np.mean(solidities),
                        f'phase_{phase}_mean_aspect_ratio': np.mean(aspect_ratios),
                        f'phase_{phase}_num_regions': len(regions)
                    })
                    
                    # Connectivity analysis
                    features[f'phase_{phase}_connectivity'] = self._compute_phase_connectivity(phase_mask)
        
        # Inter-phase analysis
        features.update(self._analyze_phase_interfaces(segmented))
        
        return features
    
    def _compute_phase_connectivity(self, phase_mask: np.ndarray) -> float:
        """Compute phase connectivity metric."""
        
        from skimage import morphology
        
        # Label connected components
        labeled = morphology.label(phase_mask)
        num_components = np.max(labeled)
        
        if num_components == 0:
            return 0.0
        
        # Euler number based connectivity
        # Higher connectivity = more connected structure
        total_volume = np.sum(phase_mask)
        largest_component_volume = np.max([np.sum(labeled == i) for i in range(1, num_components + 1)])
        
        connectivity = largest_component_volume / total_volume if total_volume > 0 else 0.0
        
        return connectivity
    
    def _analyze_phase_interfaces(self, segmented: np.ndarray) -> Dict[str, float]:
        """Analyze interfaces between phases."""
        
        features = {}
        
        # Find phase boundaries
        from skimage import segmentation, filters
        
        boundaries = segmentation.find_boundaries(segmented, mode='thick')
        
        # Interface density
        total_volume = segmented.size
        interface_volume = np.sum(boundaries)
        
        features['interface_density'] = interface_volume / total_volume
        
        # Interface roughness (based on gradient)
        gradient_magnitude = np.sqrt(np.sum([filters.sobel(segmented.astype(float), axis=i)**2 
                                           for i in range(3)], axis=0))
        
        features['interface_roughness'] = np.mean(gradient_magnitude[boundaries])
        
        return features
    
    def _analyze_porosity_detailed(self, volume: np.ndarray) -> Dict[str, float]:
        """Detailed porosity analysis."""
        
        # Use the processor's porosity analysis
        porosity_metrics = self.processor.analyze_porosity(volume)
        
        # Add advanced porosity metrics
        features = porosity_metrics.copy()
        
        # Pore shape analysis
        from skimage import filters, morphology, measure
        
        # Threshold for pores
        threshold = filters.threshold_otsu(volume)
        pore_mask = volume < threshold
        
        # Clean up
        cleaned_pores = morphology.remove_small_objects(pore_mask, min_size=10)
        
        # Label pores
        labeled_pores = morphology.label(cleaned_pores)
        pore_regions = measure.regionprops(labeled_pores)
        
        if pore_regions:
            # Pore shape characteristics
            pore_eccentricities = [region.eccentricity for region in pore_regions]
            pore_solidities = [region.solidity for region in pore_regions]
            pore_sphericities = [self._compute_sphericity(region) for region in pore_regions]
            
            features.update({
                'mean_pore_eccentricity': np.mean(pore_eccentricities),
                'mean_pore_solidity': np.mean(pore_solidities),
                'mean_pore_sphericity': np.mean(pore_sphericities),
                'pore_shape_uniformity': 1.0 - np.std(pore_sphericities)
            })
            
            # Pore distribution analysis
            features.update(self._analyze_pore_distribution(labeled_pores))
        
        return features
    
    def _compute_sphericity(self, region) -> float:
        """Compute sphericity of a 3D region."""
        
        # Sphericity = (π^(1/3) * (6*Volume)^(2/3)) / Surface_Area
        # For 2D approximation, use equivalent circle
        
        area = region.area
        perimeter = region.perimeter
        
        if perimeter > 0:
            sphericity = 4 * np.pi * area / (perimeter ** 2)
            return min(sphericity, 1.0)
        else:
            return 0.0
    
    def _analyze_pore_distribution(self, labeled_pores: np.ndarray) -> Dict[str, float]:
        """Analyze spatial distribution of pores."""
        
        features = {}
        
        # Pore spacing analysis
        from scipy.spatial.distance import pdist
        from skimage import measure
        
        # Get pore centroids
        regions = measure.regionprops(labeled_pores)
        
        if len(regions) > 1:
            centroids = np.array([region.centroid for region in regions])
            
            # Calculate nearest neighbor distances
            distances = pdist(centroids)
            
            features.update({
                'mean_pore_spacing': np.mean(distances),
                'min_pore_spacing': np.min(distances),
                'pore_spacing_uniformity': 1.0 - (np.std(distances) / np.mean(distances))
            })
            
            # Clustering analysis
            features['pore_clustering_index'] = self._compute_clustering_index(centroids)
        
        return features
    
    def _compute_clustering_index(self, points: np.ndarray) -> float:
        """Compute clustering index for point distribution."""
        
        from scipy.spatial.distance import pdist
        
        if len(points) < 2:
            return 0.0
        
        # Compute average nearest neighbor distance
        distances = pdist(points)
        avg_distance = np.mean(distances)
        
        # Compare to expected distance for random distribution
        volume = np.prod(np.max(points, axis=0) - np.min(points, axis=0))
        density = len(points) / volume
        expected_distance = 1 / (2 * np.sqrt(density))
        
        # Clustering index: < 1 = clustered, > 1 = dispersed
        clustering_index = avg_distance / expected_distance if expected_distance > 0 else 1.0
        
        return clustering_index
    
    def _analyze_texture_features(self, volume: np.ndarray) -> Dict[str, float]:
        """Advanced texture analysis."""
        
        # Use processor's texture analysis as base
        texture_features = self.processor._extract_texture_coefficients(volume)
        
        # Add Haralick features
        features = texture_features.copy()
        
        # Multi-scale texture analysis
        for scale in [1, 2, 4]:
            if scale > 1:
                # Downsample volume
                from skimage.transform import rescale
                downsampled = rescale(volume[::scale, ::scale, ::scale], 1.0, anti_aliasing=True)
            else:
                downsampled = volume
            
            # Local variance and gradients
            local_variance = self._compute_local_variance(downsampled)
            gradient_features = self._compute_gradient_features(downsampled)
            
            features.update({
                f'local_variance_scale_{scale}': local_variance,
                f'gradient_magnitude_scale_{scale}': gradient_features['magnitude'],
                f'gradient_direction_scale_{scale}': gradient_features['direction_std']
            })
        
        return features
    
    def _compute_local_variance(self, volume: np.ndarray, window_size: int = 5) -> float:
        """Compute local variance as texture measure."""
        
        from scipy.ndimage import generic_filter
        
        # Compute local variance using sliding window
        local_var = generic_filter(volume, np.var, size=window_size)
        
        return np.mean(local_var)
    
    def _compute_gradient_features(self, volume: np.ndarray) -> Dict[str, float]:
        """Compute gradient-based texture features."""
        
        # Compute gradients
        grad_z, grad_y, grad_x = np.gradient(volume)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Gradient directions
        directions = np.arctan2(grad_y, grad_x)
        
        return {
            'magnitude': np.mean(magnitude),
            'direction_std': np.std(directions)
        }
    
    def _analyze_defects(self, volume: np.ndarray) -> Dict[str, float]:
        """Analyze defects in microstructure."""
        
        features = {}
        
        # Different types of defects
        
        # 1. Cracks (high aspect ratio, low intensity regions)
        crack_features = self._detect_cracks(volume)
        features.update(crack_features)
        
        # 2. Voids (spherical low intensity regions)
        void_features = self._detect_voids(volume)
        features.update(void_features)
        
        # 3. Inclusions (high intensity particles)
        inclusion_features = self._detect_inclusions(volume)
        features.update(inclusion_features)
        
        # 4. Surface roughness defects
        surface_features = self._analyze_surface_defects(volume)
        features.update(surface_features)
        
        return features
    
    def _detect_cracks(self, volume: np.ndarray) -> Dict[str, float]:
        """Detect crack-like defects."""
        
        from skimage import filters, morphology, measure
        
        # Edge detection to find potential cracks
        edges = filters.sobel(volume)
        
        # Threshold to get strong edges
        threshold = filters.threshold_otsu(edges)
        crack_candidates = edges > threshold
        
        # Morphological processing to enhance crack-like structures
        crack_enhanced = morphology.closing(crack_candidates, morphology.disk(2))
        crack_cleaned = morphology.remove_small_objects(crack_enhanced, min_size=50)
        
        # Analyze crack properties
        labeled_cracks = morphology.label(crack_cleaned)
        crack_regions = measure.regionprops(labeled_cracks)
        
        features = {}
        
        if crack_regions:
            # Filter for crack-like shapes (high aspect ratio)
            crack_like = [region for region in crack_regions 
                         if region.major_axis_length / (region.minor_axis_length + 1e-6) > 3]
            
            features['crack_count'] = len(crack_like)
            features['crack_density'] = len(crack_like) / volume.size
            
            if crack_like:
                crack_lengths = [region.major_axis_length for region in crack_like]
                features['mean_crack_length'] = np.mean(crack_lengths)
                features['max_crack_length'] = np.max(crack_lengths)
        else:
            features.update({
                'crack_count': 0,
                'crack_density': 0.0,
                'mean_crack_length': 0.0,
                'max_crack_length': 0.0
            })
        
        return features
    
    def _detect_voids(self, volume: np.ndarray) -> Dict[str, float]:
        """Detect void-like defects."""
        
        from skimage import filters, morphology, measure
        
        # Low intensity regions
        threshold = filters.threshold_otsu(volume)
        void_candidates = volume < threshold * 0.8  # Lower threshold for voids
        
        # Clean up
        void_cleaned = morphology.remove_small_objects(void_candidates, min_size=20)
        void_filled = morphology.remove_small_holes(void_cleaned, area_threshold=10)
        
        # Analyze void properties
        labeled_voids = morphology.label(void_filled)
        void_regions = measure.regionprops(labeled_voids)
        
        features = {}
        
        if void_regions:
            # Filter for void-like shapes (circular/spherical)
            void_like = [region for region in void_regions 
                        if region.eccentricity < 0.8]  # More circular
            
            features['void_count'] = len(void_like)
            features['void_density'] = len(void_like) / volume.size
            
            if void_like:
                void_areas = [region.area for region in void_like]
                features['mean_void_size'] = np.mean(void_areas)
                features['max_void_size'] = np.max(void_areas)
        else:
            features.update({
                'void_count': 0,
                'void_density': 0.0,
                'mean_void_size': 0.0,
                'max_void_size': 0.0
            })
        
        return features
    
    def _detect_inclusions(self, volume: np.ndarray) -> Dict[str, float]:
        """Detect inclusion-like defects."""
        
        from skimage import filters, morphology, measure
        
        # High intensity regions
        threshold = filters.threshold_otsu(volume)
        inclusion_candidates = volume > threshold * 1.2  # Higher threshold for inclusions
        
        # Clean up
        inclusion_cleaned = morphology.remove_small_objects(inclusion_candidates, min_size=10)
        
        # Analyze inclusion properties
        labeled_inclusions = morphology.label(inclusion_cleaned)
        inclusion_regions = measure.regionprops(labeled_inclusions)
        
        features = {
            'inclusion_count': len(inclusion_regions),
            'inclusion_density': len(inclusion_regions) / volume.size
        }
        
        if inclusion_regions:
            inclusion_areas = [region.area for region in inclusion_regions]
            features['mean_inclusion_size'] = np.mean(inclusion_areas)
            features['max_inclusion_size'] = np.max(inclusion_areas)
        else:
            features.update({
                'mean_inclusion_size': 0.0,
                'max_inclusion_size': 0.0
            })
        
        return features
    
    def _analyze_surface_defects(self, volume: np.ndarray) -> Dict[str, float]:
        """Analyze surface-related defects."""
        
        # Surface roughness analysis (already in processor)
        surface_features = self.processor._extract_surface_roughness(volume)
        
        # Additional surface defect analysis
        features = surface_features.copy()
        
        # Surface waviness (low frequency roughness)
        from scipy.ndimage import gaussian_filter
        
        # Low-pass filter to get waviness
        smoothed = gaussian_filter(volume, sigma=5)
        waviness = volume - smoothed
        
        features['surface_waviness'] = np.std(waviness)
        
        # Surface defect density (sharp local variations)
        from skimage import filters
        
        # High-pass filter for defects
        high_freq = volume - gaussian_filter(volume, sigma=2)
        defect_threshold = np.std(high_freq) * 3
        surface_defects = np.abs(high_freq) > defect_threshold
        
        features['surface_defect_density'] = np.sum(surface_defects) / volume.size
        
        return features
    
    def _analyze_surface_characteristics(self, volume: np.ndarray) -> Dict[str, float]:
        """Comprehensive surface analysis."""
        
        features = {}
        
        # Use processor's surface analysis
        surface_features = self.processor._extract_surface_roughness(volume)
        features.update(surface_features)
        
        # Additional surface metrics
        # Fractal dimension of surface
        features['surface_fractal_dimension'] = self._compute_surface_fractal_dimension(volume)
        
        # Surface area to volume ratio
        features['surface_area_ratio'] = self._compute_surface_area_ratio(volume)
        
        return features
    
    def _compute_surface_fractal_dimension(self, volume: np.ndarray) -> float:
        """Compute fractal dimension of surface."""
        
        # Simplified box-counting method
        from skimage import measure
        
        # Get surface using marching cubes (simplified to edge detection)
        from skimage import filters
        edges = filters.sobel(volume)
        
        # Box counting at different scales
        scales = [2, 4, 8, 16]
        counts = []
        
        for scale in scales:
            # Downsample
            downsampled = edges[::scale, ::scale, ::scale]
            # Count non-zero pixels
            count = np.sum(downsampled > 0)
            counts.append(count)
        
        # Fit power law
        if len(counts) > 1 and all(c > 0 for c in counts):
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            # Linear fit
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dimension = -slope
            
            # Clamp to reasonable range
            return np.clip(fractal_dimension, 1.0, 3.0)
        else:
            return 2.0  # Default for 2D surface
    
    def _compute_surface_area_ratio(self, volume: np.ndarray) -> float:
        """Compute surface area to volume ratio."""
        
        from skimage import measure
        
        # Threshold volume
        threshold = filters.threshold_otsu(volume)
        binary_volume = volume > threshold
        
        # Estimate surface area using voxel faces
        # Count boundary voxels (simplified approach)
        from scipy.ndimage import binary_erosion
        
        eroded = binary_erosion(binary_volume)
        surface_voxels = binary_volume & ~eroded
        
        surface_area = np.sum(surface_voxels)
        total_volume = np.sum(binary_volume)
        
        if total_volume > 0:
            return surface_area / total_volume
        else:
            return 0.0
    
    def _analyze_parameters(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze process parameters in context of microstructure."""
        
        analysis = {}
        
        # Parameter validation
        for validator_name, validator in self._parameter_validators.items():
            try:
                validation_result = validator(parameters, volume)
                analysis[validator_name] = validation_result
            except Exception as e:
                warnings.warn(f"Parameter validation {validator_name} failed: {str(e)}")
        
        # Parameter relationships
        analysis['parameter_relationships'] = self._analyze_parameter_relationships(parameters)
        
        # Process window analysis
        analysis['process_window'] = self._analyze_process_window(parameters)
        
        return analysis
    
    def _validate_energy_density(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Validate energy density parameters."""
        
        # Calculate volumetric energy density
        energy_density = parameters.laser_power / (
            parameters.scan_speed * parameters.hatch_spacing * parameters.layer_thickness / 1000
        )
        
        # Extract density from microstructure
        porosity_metrics = self.processor.analyze_porosity(volume)
        achieved_density = 1.0 - porosity_metrics['total_porosity'] / 100
        
        return {
            'calculated_energy_density': energy_density,
            'achieved_density': achieved_density,
            'optimal_range': (60, 100),  # J/mm³ for Ti-6Al-4V
            'in_optimal_range': 60 <= energy_density <= 100,
            'efficiency': achieved_density / (energy_density / 80)  # Normalized efficiency
        }
    
    def _validate_thermal_history(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Validate thermal history indicators."""
        
        # Simplified thermal history assessment based on parameters
        
        # Cooling rate indicator (higher scan speed = faster cooling)
        cooling_rate_indicator = parameters.scan_speed / parameters.laser_power
        
        # Thermal gradient indicator
        thermal_gradient_indicator = parameters.laser_power / (parameters.layer_thickness ** 2)
        
        # Extract microstructural indicators of thermal history
        features = self.processor.extract_features(volume)
        
        return {
            'cooling_rate_indicator': cooling_rate_indicator,
            'thermal_gradient_indicator': thermal_gradient_indicator,
            'grain_size': features.get('mean_grain_size', 0),
            'thermal_history_assessment': 'acceptable' if 0.1 < cooling_rate_indicator < 2.0 else 'needs_review'
        }
    
    def _validate_melt_pool_characteristics(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Validate melt pool related parameters."""
        
        # Melt pool dimensions estimation
        melt_pool_width = 0.1 * (parameters.laser_power / parameters.scan_speed) ** 0.5  # Simplified model
        melt_pool_depth = 0.05 * (parameters.laser_power / parameters.scan_speed) ** 0.3
        
        # Overlap calculations
        hatch_overlap = (melt_pool_width - parameters.hatch_spacing) / melt_pool_width
        layer_overlap = (melt_pool_depth - parameters.layer_thickness) / melt_pool_depth
        
        return {
            'estimated_melt_pool_width': melt_pool_width,
            'estimated_melt_pool_depth': melt_pool_depth,
            'hatch_overlap': hatch_overlap,
            'layer_overlap': layer_overlap,
            'adequate_overlap': hatch_overlap > 0.1 and layer_overlap > 0.2
        }
    
    def _validate_layer_bonding(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Validate layer bonding quality."""
        
        # Layer bonding assessment based on defects and uniformity
        defect_features = self._analyze_defects(volume)
        
        # Look for layer-related defects (horizontal cracks, delamination)
        crack_density = defect_features.get('crack_density', 0)
        void_density = defect_features.get('void_density', 0)
        
        # Energy per layer
        energy_per_layer = parameters.laser_power / (parameters.scan_speed * parameters.hatch_spacing)
        
        return {
            'energy_per_layer': energy_per_layer,
            'crack_density': crack_density,
            'void_density': void_density,
            'bonding_quality': 'good' if crack_density < 0.001 and void_density < 0.01 else 'poor'
        }
    
    def _validate_scan_strategy(self, parameters: ProcessParameters, volume: np.ndarray) -> Dict[str, Any]:
        """Validate scan strategy effectiveness."""
        
        # Assess scan strategy based on achieved uniformity
        features = self.processor.extract_features(volume)
        
        # Uniformity indicators
        porosity_uniformity = 1.0 - (features.get('pore_volume_std', 0) / (features.get('mean_pore_volume', 1) + 1e-6))
        
        # Hatch spacing to layer thickness ratio
        aspect_ratio = parameters.hatch_spacing / parameters.layer_thickness
        
        return {
            'hatch_to_layer_ratio': aspect_ratio,
            'porosity_uniformity': porosity_uniformity,
            'optimal_aspect_ratio': 2 <= aspect_ratio <= 6,
            'scan_strategy_effectiveness': 'good' if porosity_uniformity > 0.8 and 2 <= aspect_ratio <= 6 else 'needs_optimization'
        }
    
    def _analyze_parameter_relationships(self, parameters: ProcessParameters) -> Dict[str, float]:
        """Analyze relationships between parameters."""
        
        relationships = {}
        
        # Energy density
        energy_density = parameters.laser_power / (
            parameters.scan_speed * parameters.hatch_spacing * parameters.layer_thickness / 1000
        )
        relationships['volumetric_energy_density'] = energy_density
        
        # Line energy
        line_energy = parameters.laser_power / parameters.scan_speed
        relationships['line_energy'] = line_energy
        
        # Area energy
        area_energy = parameters.laser_power / (parameters.scan_speed * parameters.hatch_spacing)
        relationships['area_energy'] = area_energy
        
        # Build rate
        build_rate = parameters.scan_speed * parameters.hatch_spacing * parameters.layer_thickness / 1000
        relationships['build_rate'] = build_rate
        
        return relationships
    
    def _analyze_process_window(self, parameters: ProcessParameters) -> Dict[str, Any]:
        """Analyze if parameters are within optimal process window."""
        
        # Define process windows for different objectives
        windows = {
            'high_density': {
                'energy_density_range': (70, 110),
                'scan_speed_range': (600, 1200),
                'description': 'Parameters optimized for high density parts'
            },
            'high_strength': {
                'energy_density_range': (80, 120),
                'scan_speed_range': (800, 1400),
                'description': 'Parameters optimized for high strength'
            },
            'good_surface': {
                'layer_thickness_range': (20, 40),
                'hatch_spacing_range': (80, 140),
                'description': 'Parameters optimized for surface quality'
            }
        }
        
        # Calculate current energy density
        energy_density = parameters.laser_power / (
            parameters.scan_speed * parameters.hatch_spacing * parameters.layer_thickness / 1000
        )
        
        # Check which windows the parameters fall into
        compatible_windows = []
        
        for window_name, window_def in windows.items():
            compatible = True
            
            if 'energy_density_range' in window_def:
                ed_min, ed_max = window_def['energy_density_range']
                if not (ed_min <= energy_density <= ed_max):
                    compatible = False
            
            if 'scan_speed_range' in window_def:
                ss_min, ss_max = window_def['scan_speed_range']
                if not (ss_min <= parameters.scan_speed <= ss_max):
                    compatible = False
            
            if 'layer_thickness_range' in window_def:
                lt_min, lt_max = window_def['layer_thickness_range']
                if not (lt_min <= parameters.layer_thickness <= lt_max):
                    compatible = False
            
            if 'hatch_spacing_range' in window_def:
                hs_min, hs_max = window_def['hatch_spacing_range']
                if not (hs_min <= parameters.hatch_spacing <= hs_max):
                    compatible = False
            
            if compatible:
                compatible_windows.append(window_name)
        
        return {
            'energy_density': energy_density,
            'compatible_windows': compatible_windows,
            'window_definitions': windows,
            'in_any_window': len(compatible_windows) > 0
        }
    
    def _assess_quality(self, volume: np.ndarray, features: Dict[str, float], 
                       parameters: Optional[ProcessParameters] = None) -> Dict[str, float]:
        """Assess overall quality metrics."""
        
        quality_metrics = {}
        
        # Run quality assessors
        for assessor_name, assessor in self._quality_assessors.items():
            try:
                quality_score = assessor(volume, features, parameters)
                quality_metrics[assessor_name] = quality_score
            except Exception as e:
                warnings.warn(f"Quality assessment {assessor_name} failed: {str(e)}")
        
        # Overall quality score
        if quality_metrics:
            quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _assess_density_quality(self, volume: np.ndarray, features: Dict[str, float], 
                               parameters: Optional[ProcessParameters] = None) -> float:
        """Assess density quality score."""
        
        porosity = features.get('porosity', features.get('total_porosity', 5.0))
        density = 1.0 - porosity / 100 if porosity <= 100 else 1.0 - porosity
        
        # Score based on density (higher is better)
        if density >= 0.995:
            return 1.0
        elif density >= 0.99:
            return 0.8
        elif density >= 0.95:
            return 0.6
        elif density >= 0.90:
            return 0.4
        else:
            return 0.2
    
    def _assess_microstructure_quality(self, volume: np.ndarray, features: Dict[str, float], 
                                     parameters: Optional[ProcessParameters] = None) -> float:
        """Assess microstructure quality score."""
        
        score_components = []
        
        # Grain size uniformity
        if 'grain_diameter_cv' in features:
            cv = features['grain_diameter_cv']
            uniformity_score = max(0.0, 1.0 - cv)  # Lower CV is better
            score_components.append(uniformity_score)
        
        # Phase distribution
        if 'phase_1_fraction' in features:
            phase_balance = min(features.get(f'phase_{i}_fraction', 0) for i in range(1, 4))
            balance_score = min(phase_balance * 5, 1.0)  # Reasonable phase balance
            score_components.append(balance_score)
        
        # Defect assessment
        crack_density = features.get('crack_density', 0)
        void_density = features.get('void_density', 0)
        defect_score = max(0.0, 1.0 - 10 * (crack_density + void_density))
        score_components.append(defect_score)
        
        return np.mean(score_components) if score_components else 0.5
    
    def _assess_defect_severity(self, volume: np.ndarray, features: Dict[str, float], 
                              parameters: Optional[ProcessParameters] = None) -> float:
        """Assess defect severity score."""
        
        # Lower scores for more severe defects
        crack_penalty = features.get('crack_density', 0) * 100
        void_penalty = features.get('void_density', 0) * 50
        inclusion_penalty = features.get('inclusion_density', 0) * 20
        
        total_penalty = crack_penalty + void_penalty + inclusion_penalty
        
        return max(0.0, 1.0 - total_penalty)
    
    def _assess_uniformity(self, volume: np.ndarray, features: Dict[str, float], 
                          parameters: Optional[ProcessParameters] = None) -> float:
        """Assess uniformity score."""
        
        uniformity_indicators = []
        
        # Porosity uniformity
        if 'pore_spacing_uniformity' in features:
            uniformity_indicators.append(features['pore_spacing_uniformity'])
        
        # Grain size uniformity
        if 'grain_diameter_cv' in features:
            cv = features['grain_diameter_cv']
            grain_uniformity = max(0.0, 1.0 - cv)
            uniformity_indicators.append(grain_uniformity)
        
        # Texture uniformity
        if 'local_variance_scale_1' in features:
            variance = features['local_variance_scale_1']
            texture_uniformity = max(0.0, 1.0 - variance / 0.1)  # Normalized
            uniformity_indicators.append(texture_uniformity)
        
        return np.mean(uniformity_indicators) if uniformity_indicators else 0.5
    
    def _assess_printability(self, volume: np.ndarray, features: Dict[str, float], 
                           parameters: Optional[ProcessParameters] = None) -> float:
        """Assess printability score."""
        
        printability_factors = []
        
        # Density (higher is more printable)
        density_score = self._assess_density_quality(volume, features, parameters)
        printability_factors.append(density_score)
        
        # Surface quality
        surface_roughness = features.get('surface_roughness_mean', 10.0)
        surface_score = max(0.0, 1.0 - surface_roughness / 20.0)  # Normalize by acceptable roughness
        printability_factors.append(surface_score)
        
        # Defect level
        defect_score = self._assess_defect_severity(volume, features, parameters)
        printability_factors.append(defect_score)
        
        # Process parameter compatibility
        if parameters is not None:
            energy_density = parameters.laser_power / (
                parameters.scan_speed * parameters.hatch_spacing * parameters.layer_thickness / 1000
            )
            
            if 60 <= energy_density <= 100:
                param_score = 1.0
            else:
                param_score = max(0.0, 1.0 - 0.01 * abs(energy_density - 80))
            
            printability_factors.append(param_score)
        
        return np.mean(printability_factors)
    
    def _generate_recommendations(self, microstructure_features: Dict[str, float],
                                parameter_analysis: Dict[str, Any],
                                quality_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        # Density recommendations
        density_quality = quality_metrics.get('density_quality', 0.5)
        if density_quality < 0.8:
            porosity = microstructure_features.get('porosity', microstructure_features.get('total_porosity', 0))
            if porosity > 2.0:
                recommendations.append(
                    f"High porosity detected ({porosity:.1f}%). Consider increasing energy density "
                    "by reducing scan speed or increasing laser power."
                )
        
        # Defect recommendations
        defect_severity = quality_metrics.get('defect_severity', 1.0)
        if defect_severity < 0.7:
            crack_density = microstructure_features.get('crack_density', 0)
            if crack_density > 0.001:
                recommendations.append(
                    "Crack formation detected. Consider optimizing thermal gradients "
                    "by preheating powder bed or adjusting scan strategy."
                )
            
            void_density = microstructure_features.get('void_density', 0)
            if void_density > 0.01:
                recommendations.append(
                    "Excessive void formation. Check powder quality and consider "
                    "increasing energy density or improving powder spreading."
                )
        
        # Microstructure recommendations
        microstructure_quality = quality_metrics.get('microstructure_quality', 0.5)
        if microstructure_quality < 0.7:
            grain_cv = microstructure_features.get('grain_diameter_cv', 0)
            if grain_cv > 0.5:
                recommendations.append(
                    "High grain size variability detected. Consider optimizing "
                    "cooling rate through scan strategy or powder bed temperature."
                )
        
        # Parameter-specific recommendations
        if 'energy_density' in parameter_analysis:
            energy_analysis = parameter_analysis['energy_density']
            if not energy_analysis.get('in_optimal_range', True):
                energy_density = energy_analysis['calculated_energy_density']
                if energy_density < 60:
                    recommendations.append(
                        f"Energy density ({energy_density:.1f} J/mm³) is below optimal range. "
                        "Consider increasing laser power or reducing scan speed."
                    )
                elif energy_density > 100:
                    recommendations.append(
                        f"Energy density ({energy_density:.1f} J/mm³) is above optimal range. "
                        "Consider reducing laser power or increasing scan speed."
                    )
        
        # Process window recommendations
        if 'process_window' in parameter_analysis:
            window_analysis = parameter_analysis['process_window']
            if not window_analysis.get('in_any_window', True):
                recommendations.append(
                    "Parameters are outside established process windows. "
                    "Consider adjusting parameters to match proven process recipes."
                )
        
        # Surface quality recommendations
        surface_roughness = microstructure_features.get('surface_roughness_mean', 0)
        if surface_roughness > 15:
            recommendations.append(
                f"High surface roughness ({surface_roughness:.1f} μm) detected. "
                "Consider reducing layer thickness or optimizing scan parameters."
            )
        
        # Uniformity recommendations
        uniformity = quality_metrics.get('uniformity', 1.0)
        if uniformity < 0.7:
            recommendations.append(
                "Poor uniformity detected. Consider implementing scan pattern rotation "
                "or adjusting hatch spacing for more consistent melting."
            )
        
        # General recommendations
        overall_quality = quality_metrics.get('overall_quality', 0.5)
        if overall_quality < 0.6:
            recommendations.append(
                "Overall quality is below acceptable threshold. "
                "Comprehensive parameter optimization recommended."
            )
        
        return recommendations
    
    def _compute_confidence_scores(self, microstructure_features: Dict[str, float],
                                 quality_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute confidence scores for analysis results."""
        
        confidence_scores = {}
        
        # Base confidence on data quality and completeness
        feature_completeness = len(microstructure_features) / 20  # Assume 20 expected features
        confidence_scores['feature_extraction'] = min(feature_completeness, 1.0)
        
        # Quality assessment confidence
        quality_consistency = 1.0 - np.std(list(quality_metrics.values())) if quality_metrics else 0.5
        confidence_scores['quality_assessment'] = quality_consistency
        
        # Defect detection confidence
        defect_features = [k for k in microstructure_features.keys() if 'crack' in k or 'void' in k or 'inclusion' in k]
        defect_confidence = len(defect_features) / 10  # Assume 10 defect-related features
        confidence_scores['defect_detection'] = min(defect_confidence, 1.0)
        
        # Overall confidence
        confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        
        return confidence_scores
    
    def _collect_warnings(self, microstructure_features: Dict[str, float],
                         parameter_analysis: Dict[str, Any],
                         quality_metrics: Dict[str, float]) -> List[str]:
        """Collect analysis warnings."""
        
        analysis_warnings = []
        
        # Data quality warnings
        if len(microstructure_features) < 10:
            analysis_warnings.append("Limited feature extraction - analysis may be incomplete")
        
        # Extreme values warnings
        porosity = microstructure_features.get('porosity', microstructure_features.get('total_porosity', 0))
        if porosity > 10:
            analysis_warnings.append(f"Extremely high porosity ({porosity:.1f}%) detected")
        
        # Quality warnings
        overall_quality = quality_metrics.get('overall_quality', 1.0)
        if overall_quality < 0.4:
            analysis_warnings.append("Very low overall quality - results may not be reliable")
        
        # Missing data warnings
        if not parameter_analysis:
            analysis_warnings.append("No process parameters provided - parameter analysis unavailable")
        
        return analysis_warnings
    
    def _compare_features(self, analyses: List[AnalysisReport], labels: List[str]) -> Dict[str, Any]:
        """Compare features across multiple analyses."""
        
        # Extract common features
        all_features = set()
        for analysis in analyses:
            all_features.update(analysis.microstructure_features.keys())
        
        common_features = all_features
        for analysis in analyses:
            common_features &= set(analysis.microstructure_features.keys())
        
        # Compare common features
        feature_comparison = {}
        
        for feature in common_features:
            values = [analysis.microstructure_features[feature] for analysis in analyses]
            
            feature_comparison[feature] = {
                'values': dict(zip(labels, values)),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        
        return feature_comparison
    
    def _perform_statistical_analysis(self, analyses: List[AnalysisReport], labels: List[str]) -> Dict[str, Any]:
        """Perform statistical analysis of multiple samples."""
        
        statistical_analysis = {}
        
        # Extract quality metrics for all samples
        quality_metrics = {}
        for metric_name in analyses[0].quality_metrics.keys():
            values = [analysis.quality_metrics.get(metric_name, 0) for analysis in analyses]
            quality_metrics[metric_name] = values
        
        # Statistical tests
        if len(analyses) > 2:
            # ANOVA test for differences
            try:
                from scipy import stats
                
                for metric_name, values in quality_metrics.items():
                    if len(set(values)) > 1:  # Check for variance
                        # Simplified one-way ANOVA (assuming groups)
                        group_size = len(values) // 2
                        group1 = values[:group_size]
                        group2 = values[group_size:]
                        
                        if len(group1) > 0 and len(group2) > 0:
                            statistic, p_value = stats.ttest_ind(group1, group2)
                            
                            statistical_analysis[f'{metric_name}_ttest'] = {
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            except ImportError:
                statistical_analysis['note'] = "Statistical tests require scipy"
        
        # Correlation analysis
        if len(analyses) > 3:
            correlation_matrix = self._compute_correlation_matrix(analyses)
            statistical_analysis['correlations'] = correlation_matrix
        
        return statistical_analysis
    
    def _compute_correlation_matrix(self, analyses: List[AnalysisReport]) -> Dict[str, Any]:
        """Compute correlation matrix for features."""
        
        # Extract common numerical features
        common_features = set(analyses[0].microstructure_features.keys())
        for analysis in analyses[1:]:
            common_features &= set(analysis.microstructure_features.keys())
        
        # Create feature matrix
        feature_matrix = []
        feature_names = list(common_features)
        
        for analysis in analyses:
            row = [analysis.microstructure_features[feature] for feature in feature_names]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        
        # Compute correlation matrix
        if feature_matrix.shape[0] > 2 and feature_matrix.shape[1] > 1:
            correlation_matrix = np.corrcoef(feature_matrix.T)
            
            return {
                'matrix': correlation_matrix.tolist(),
                'feature_names': feature_names,
                'shape': correlation_matrix.shape
            }
        else:
            return {'note': 'Insufficient data for correlation analysis'}
    
    def _rank_microstructures(self, analyses: List[AnalysisReport], labels: List[str]) -> Dict[str, Any]:
        """Rank microstructures by quality."""
        
        # Extract overall quality scores
        quality_scores = []
        for i, analysis in enumerate(analyses):
            overall_quality = analysis.quality_metrics.get('overall_quality', 0.5)
            quality_scores.append((labels[i], overall_quality))
        
        # Sort by quality
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Multi-criteria ranking
        criteria_rankings = {}
        
        for criterion in ['density_quality', 'microstructure_quality', 'defect_severity', 'uniformity']:
            criterion_scores = []
            for i, analysis in enumerate(analyses):
                score = analysis.quality_metrics.get(criterion, 0.5)
                criterion_scores.append((labels[i], score))
            
            criterion_scores.sort(key=lambda x: x[1], reverse=True)
            criteria_rankings[criterion] = criterion_scores
        
        return {
            'overall_ranking': quality_scores,
            'criteria_rankings': criteria_rankings,
            'best_overall': quality_scores[0][0] if quality_scores else None,
            'worst_overall': quality_scores[-1][0] if quality_scores else None
        }
    
    def _generate_comparison_summary(self, analyses: List[AnalysisReport], labels: List[str]) -> Dict[str, Any]:
        """Generate summary of comparison analysis."""
        
        summary = {
            'num_samples': len(analyses),
            'labels': labels
        }
        
        # Overall quality statistics
        overall_qualities = [analysis.quality_metrics.get('overall_quality', 0.5) for analysis in analyses]
        
        summary['quality_statistics'] = {
            'mean': np.mean(overall_qualities),
            'std': np.std(overall_qualities),
            'min': min(overall_qualities),
            'max': max(overall_qualities),
            'range': max(overall_qualities) - min(overall_qualities)
        }
        
        # Common issues
        all_warnings = []
        for analysis in analyses:
            all_warnings.extend(analysis.warnings)
        
        # Count warning frequency
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        summary['common_issues'] = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.recommendations)
        
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        summary['common_recommendations'] = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return summary