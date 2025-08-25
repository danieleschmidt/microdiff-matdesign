#!/usr/bin/env python3
"""
Robust Quantum Materials Discovery System
Generation 2: MAKE IT ROBUST - Comprehensive Error Handling & Validation

Enhanced quantum materials discovery with enterprise-grade robustness,
comprehensive error handling, input validation, and fault tolerance.
"""

import sys
import time
import json
import math
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import random
from abc import ABC, abstractmethod
from enum import Enum
import contextlib


# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quantum_materials_robust.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class QuantumError(Exception):
    """Custom exception for quantum simulation errors."""
    pass


class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass


class SystemState(Enum):
    """System operation states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class QuantumMaterialsConfig:
    """Robust configuration with validation."""
    
    num_qubits: int = field(default=8)
    quantum_layers: int = field(default=3)
    diffusion_steps: int = field(default=1000)
    material_dimensions: int = field(default=6)
    target_properties: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    batch_size: int = field(default=16)
    max_iterations: int = field(default=100)
    
    # Robustness parameters
    max_retry_attempts: int = field(default=3)
    timeout_seconds: float = field(default=30.0)
    error_threshold: float = field(default=0.1)
    validation_enabled: bool = field(default=True)
    safety_checks: bool = field(default=True)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self.validate()
    
    def validate(self) -> None:
        """Comprehensive configuration validation."""
        logger.info("Validating configuration parameters")
        
        if self.num_qubits < 1 or self.num_qubits > 32:
            raise ValidationError(f"num_qubits must be between 1 and 32, got {self.num_qubits}")
        
        if self.quantum_layers < 1 or self.quantum_layers > 10:
            raise ValidationError(f"quantum_layers must be between 1 and 10, got {self.quantum_layers}")
        
        if self.diffusion_steps < 10 or self.diffusion_steps > 10000:
            raise ValidationError(f"diffusion_steps must be between 10 and 10000, got {self.diffusion_steps}")
        
        if self.learning_rate <= 0 or self.learning_rate > 0.1:
            raise ValidationError(f"learning_rate must be between 0 and 0.1, got {self.learning_rate}")
        
        if self.batch_size < 1 or self.batch_size > 1024:
            raise ValidationError(f"batch_size must be between 1 and 1024, got {self.batch_size}")
        
        if self.max_retry_attempts < 0 or self.max_retry_attempts > 10:
            raise ValidationError(f"max_retry_attempts must be between 0 and 10, got {self.max_retry_attempts}")
        
        if self.timeout_seconds <= 0 or self.timeout_seconds > 300:
            raise ValidationError(f"timeout_seconds must be between 0 and 300, got {self.timeout_seconds}")
        
        logger.info("✅ Configuration validation passed")


@dataclass
class MaterialsResult:
    """Enhanced results with error tracking."""
    
    parameters: Dict[str, float] = field(default_factory=dict)
    predicted_properties: Dict[str, float] = field(default_factory=dict)
    quantum_advantage: float = field(default=0.0)
    confidence: float = field(default=0.0)
    generation_time: float = field(default=0.0)
    breakthrough_score: float = field(default=0.0)
    
    # Robustness metrics
    retry_count: int = field(default=0)
    error_messages: List[str] = field(default_factory=list)
    validation_status: str = field(default="pending")
    quality_score: float = field(default=0.0)
    
    def add_error(self, error_msg: str) -> None:
        """Add error message to result."""
        self.error_messages.append(error_msg)
        logger.warning(f"Result error added: {error_msg}")
    
    def validate(self) -> bool:
        """Validate result data."""
        try:
            # Check parameter ranges
            for param, value in self.parameters.items():
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    self.add_error(f"Invalid parameter {param}: {value}")
                    return False
            
            # Check property ranges
            for prop, value in self.predicted_properties.items():
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    self.add_error(f"Invalid property {prop}: {value}")
                    return False
            
            # Check metric ranges
            if not (0 <= self.confidence <= 1):
                self.add_error(f"Invalid confidence: {self.confidence}")
                return False
            
            if not (0 <= self.quantum_advantage <= 1):
                self.add_error(f"Invalid quantum_advantage: {self.quantum_advantage}")
                return False
            
            self.validation_status = "passed"
            return True
            
        except Exception as e:
            self.add_error(f"Validation error: {str(e)}")
            self.validation_status = "failed"
            return False


class RobustValidator:
    """Comprehensive input and output validation."""
    
    @staticmethod
    def validate_target_properties(properties: Dict[str, float]) -> None:
        """Validate target material properties."""
        logger.debug(f"Validating target properties: {properties}")
        
        if not isinstance(properties, dict):
            raise ValidationError("Target properties must be a dictionary")
        
        if len(properties) == 0:
            raise ValidationError("Target properties cannot be empty")
        
        valid_properties = {
            'tensile_strength': (400.0, 2000.0),  # MPa
            'elongation': (1.0, 30.0),            # %
            'density': (0.70, 1.0),               # relative
            'grain_size': (1.0, 500.0),           # micrometers
            'yield_strength': (300.0, 1800.0),    # MPa
            'hardness': (100.0, 800.0),           # HV
        }
        
        for prop, value in properties.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Property {prop} must be numeric, got {type(value)}")
            
            if math.isnan(value) or math.isinf(value):
                raise ValidationError(f"Property {prop} cannot be NaN or infinity")
            
            if prop in valid_properties:
                min_val, max_val = valid_properties[prop]
                if not (min_val <= value <= max_val):
                    raise ValidationError(f"Property {prop} = {value} outside valid range [{min_val}, {max_val}]")
        
        logger.debug("✅ Target properties validation passed")
    
    @staticmethod
    def validate_process_parameters(params: Dict[str, float]) -> None:
        """Validate process parameters."""
        logger.debug(f"Validating process parameters: {params}")
        
        valid_ranges = {
            'laser_power': (50.0, 500.0),          # Watts
            'scan_speed': (100.0, 2000.0),         # mm/s
            'layer_thickness': (10.0, 100.0),      # micrometers
            'hatch_spacing': (20.0, 300.0),        # micrometers
            'powder_bed_temp': (20.0, 200.0),      # Celsius
            'scan_strategy_angle': (0.0, 90.0),    # degrees
        }
        
        for param, value in params.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Parameter {param} must be numeric, got {type(value)}")
            
            if math.isnan(value) or math.isinf(value):
                raise ValidationError(f"Parameter {param} cannot be NaN or infinity")
            
            if param in valid_ranges:
                min_val, max_val = valid_ranges[param]
                if not (min_val <= value <= max_val):
                    raise ValidationError(f"Parameter {param} = {value} outside valid range [{min_val}, {max_val}]")
        
        logger.debug("✅ Process parameters validation passed")


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle error with logging and recovery attempt."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.error(f"Error in {context}: {error_type} - {str(error)}")
        logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        # Determine if error is recoverable
        recoverable_errors = [ValidationError, ProcessingError]
        if type(error) in recoverable_errors:
            logger.info(f"Error {error_type} is recoverable, attempting recovery")
            return True
        
        logger.error(f"Error {error_type} is not recoverable")
        return False
    
    def should_retry(self, error_type: str, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_retries:
            logger.warning(f"Max retries ({self.max_retries}) reached for {error_type}")
            return False
        
        if self.error_counts.get(error_type, 0) > 10:
            logger.error(f"Too many errors of type {error_type}, giving up")
            return False
        
        return True


def with_timeout(timeout_seconds: float):
    """Decorator for function timeout (simplified version)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Simplified timeout implementation
            start_time = time.time()
            result = func(*args, **kwargs)
            
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Function {func.__name__} exceeded timeout of {timeout_seconds}s")
            
            return result
        return wrapper
    return decorator


class RobustQuantumStateSimulator:
    """Robust quantum state simulator with error handling."""
    
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.coherence_time = 100e-6
        self.error_handler = ErrorHandler()
        
        logger.info(f"Initialized robust quantum simulator with {num_qubits} qubits")
    
    @with_timeout(10.0)
    def prepare_superposition(self, classical_data: List[float]) -> List[complex]:
        """Robust superposition preparation with validation."""
        logger.debug(f"Preparing superposition for {len(classical_data)} data points")
        
        try:
            # Input validation
            if not isinstance(classical_data, list):
                raise ValidationError("Classical data must be a list")
            
            if len(classical_data) == 0:
                raise ValidationError("Classical data cannot be empty")
            
            for i, value in enumerate(classical_data):
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Data point {i} must be numeric, got {type(value)}")
                
                if math.isnan(value) or math.isinf(value):
                    raise ValidationError(f"Data point {i} cannot be NaN or infinity")
            
            # Normalize input data safely
            max_val = max(abs(x) for x in classical_data)
            if max_val == 0:
                logger.warning("All input values are zero, using default normalization")
                max_val = 1.0
            
            normalized = [x / max_val for x in classical_data]
            
            # Create quantum superposition with error checking
            quantum_state = []
            for i, value in enumerate(normalized[:self.num_qubits]):
                try:
                    phase = (value * math.pi) % (2 * math.pi)
                    amplitude = math.sqrt(abs(value)) if abs(value) <= 1.0 else 1.0
                    
                    real_part = amplitude * math.cos(phase)
                    imag_part = amplitude * math.sin(phase)
                    
                    if math.isnan(real_part) or math.isnan(imag_part):
                        logger.warning(f"NaN detected in qubit {i}, using default values")
                        real_part, imag_part = 1.0, 0.0
                    
                    quantum_state.append(complex(real_part, imag_part))
                    
                except Exception as e:
                    logger.error(f"Error processing qubit {i}: {e}")
                    quantum_state.append(complex(1.0, 0.0))  # Default value
            
            # Pad with zeros if needed
            while len(quantum_state) < self.num_qubits:
                quantum_state.append(complex(0.0, 0.0))
            
            # Normalize quantum state safely
            norm = math.sqrt(sum(abs(amp)**2 for amp in quantum_state))
            if norm < 1e-10:
                logger.warning("Quantum state norm too small, using equal superposition")
                quantum_state = [complex(1.0/math.sqrt(self.num_qubits), 0.0) for _ in range(self.num_qubits)]
            else:
                quantum_state = [amp / norm for amp in quantum_state]
            
            # Validate final state
            final_norm = sum(abs(amp)**2 for amp in quantum_state)
            if abs(final_norm - 1.0) > 0.01:
                logger.error(f"Quantum state not properly normalized: norm = {final_norm}")
                raise QuantumError("Failed to create valid quantum state")
            
            logger.debug(f"Successfully prepared quantum state with norm {final_norm:.6f}")
            return quantum_state
            
        except Exception as e:
            if self.error_handler.handle_error(e, "prepare_superposition"):
                logger.info("Attempting recovery with default quantum state")
                return [complex(1.0/math.sqrt(self.num_qubits), 0.0) for _ in range(self.num_qubits)]
            else:
                raise
    
    @with_timeout(15.0)
    def apply_quantum_evolution(self, quantum_state: List[complex], 
                              evolution_steps: int = 10) -> List[complex]:
        """Robust quantum evolution with error recovery."""
        logger.debug(f"Applying quantum evolution for {evolution_steps} steps")
        
        try:
            # Input validation
            if not isinstance(quantum_state, list):
                raise ValidationError("Quantum state must be a list")
            
            if len(quantum_state) != self.num_qubits:
                raise ValidationError(f"Quantum state length {len(quantum_state)} != {self.num_qubits}")
            
            if evolution_steps <= 0 or evolution_steps > 100:
                raise ValidationError(f"Evolution steps must be between 1 and 100, got {evolution_steps}")
            
            # Validate initial state
            initial_norm = sum(abs(amp)**2 for amp in quantum_state)
            if abs(initial_norm - 1.0) > 0.1:
                raise QuantumError(f"Invalid initial quantum state norm: {initial_norm}")
            
            evolved_state = quantum_state.copy()
            
            for step in range(evolution_steps):
                try:
                    new_state = []
                    
                    # Apply rotation gates with safety checks
                    for i in range(self.num_qubits):
                        angle = (step + 1) * math.pi / evolution_steps
                        
                        # Clamp angle to prevent overflow
                        angle = max(-10*math.pi, min(10*math.pi, angle))
                        
                        old_amp = evolved_state[i]
                        cos_angle = math.cos(angle)
                        sin_angle = math.sin(angle)
                        
                        real_part = old_amp.real * cos_angle - old_amp.imag * sin_angle
                        imag_part = old_amp.real * sin_angle + old_amp.imag * cos_angle
                        
                        # Check for NaN/infinity
                        if math.isnan(real_part) or math.isnan(imag_part) or \
                           math.isinf(real_part) or math.isinf(imag_part):
                            logger.warning(f"Invalid amplitude in step {step}, qubit {i}")
                            real_part, imag_part = old_amp.real, old_amp.imag  # Keep old value
                        
                        new_state.append(complex(real_part, imag_part))
                    
                    # Apply entanglement with safety checks
                    for i in range(self.num_qubits - 1):
                        entanglement_strength = 0.1
                        
                        control = new_state[i]
                        target = new_state[i + 1]
                        
                        new_target = target + entanglement_strength * control
                        
                        # Check for overflow
                        if abs(new_target) > 10.0:
                            logger.warning(f"Entanglement overflow in step {step}, using reduced strength")
                            new_target = target + 0.01 * control
                        
                        new_state[i + 1] = new_target
                    
                    # Normalize with safety checks
                    norm = math.sqrt(sum(abs(amp)**2 for amp in new_state))
                    if norm < 1e-10:
                        logger.error(f"Quantum state collapsed in step {step}")
                        break
                    
                    evolved_state = [amp / norm for amp in new_state]
                    
                    # Apply decoherence
                    decoherence_factor = math.exp(-step * 0.01)
                    decoherence_factor = max(0.1, decoherence_factor)  # Prevent complete decoherence
                    evolved_state = [amp * decoherence_factor for amp in evolved_state]
                    
                    # Validate step result
                    step_norm = sum(abs(amp)**2 for amp in evolved_state)
                    if step_norm < 1e-10:
                        logger.error(f"Quantum state too small after step {step}")
                        break
                
                except Exception as step_error:
                    logger.error(f"Error in evolution step {step}: {step_error}")
                    # Continue with current state
                    break
            
            # Final validation
            final_norm = sum(abs(amp)**2 for amp in evolved_state)
            if final_norm < 1e-10:
                logger.error("Final quantum state is degenerate")
                raise QuantumError("Quantum evolution failed - degenerate state")
            
            # Renormalize final state
            evolved_state = [amp / math.sqrt(final_norm) for amp in evolved_state]
            
            logger.debug(f"Quantum evolution completed successfully")
            return evolved_state
            
        except Exception as e:
            if self.error_handler.handle_error(e, "apply_quantum_evolution"):
                logger.info("Attempting recovery with original state")
                return quantum_state
            else:
                raise
    
    @with_timeout(5.0)
    def measure_quantum_state(self, quantum_state: List[complex]) -> List[float]:
        """Robust quantum measurement with validation."""
        logger.debug("Measuring quantum state")
        
        try:
            # Input validation
            if not isinstance(quantum_state, list):
                raise ValidationError("Quantum state must be a list")
            
            if len(quantum_state) != self.num_qubits:
                raise ValidationError(f"Quantum state length mismatch: {len(quantum_state)} != {self.num_qubits}")
            
            # Calculate measurement probabilities with safety checks
            probabilities = []
            for i, amp in enumerate(quantum_state):
                if not isinstance(amp, complex):
                    logger.warning(f"Non-complex amplitude at qubit {i}, converting")
                    amp = complex(amp)
                
                prob = abs(amp)**2
                
                if math.isnan(prob) or math.isinf(prob):
                    logger.warning(f"Invalid probability at qubit {i}, using default")
                    prob = 1.0 / self.num_qubits
                
                probabilities.append(prob)
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob < 1e-10:
                logger.error("Total probability too small, using uniform distribution")
                probabilities = [1.0 / self.num_qubits for _ in range(self.num_qubits)]
            else:
                probabilities = [p / total_prob for p in probabilities]
            
            # Convert probabilities to classical parameters with bounds checking
            classical_params = []
            for prob in probabilities:
                try:
                    if prob > 0.8:
                        value = 1.0
                    elif prob > 0.5:
                        value = 0.5 + (prob - 0.5) * 1.0
                    elif prob > 0.2:
                        value = prob * 2.5
                    else:
                        value = prob * 0.5
                    
                    # Clamp to valid range
                    value = max(0.0, min(1.0, value))
                    
                    if math.isnan(value) or math.isinf(value):
                        logger.warning(f"Invalid measurement value, using default")
                        value = 0.5
                    
                    classical_params.append(value)
                    
                except Exception as param_error:
                    logger.error(f"Error converting probability to parameter: {param_error}")
                    classical_params.append(0.5)  # Default value
            
            logger.debug(f"Successfully measured quantum state to {len(classical_params)} parameters")
            return classical_params
            
        except Exception as e:
            if self.error_handler.handle_error(e, "measure_quantum_state"):
                logger.info("Attempting recovery with default measurements")
                return [0.5 for _ in range(self.num_qubits)]
            else:
                raise


class RobustQuantumMaterialsDiscovery:
    """Robust quantum materials discovery with comprehensive error handling."""
    
    def __init__(self, config: Optional[QuantumMaterialsConfig] = None):
        if config is None:
            config = QuantumMaterialsConfig()
        
        self.config = config
        self.state = SystemState.INITIALIZING
        self.error_handler = ErrorHandler(config.max_retry_attempts)
        self.validator = RobustValidator()
        
        try:
            self.quantum_simulator = RobustQuantumStateSimulator(config.num_qubits)
            self.state = SystemState.READY
            
            logger.info(f"🚀 Robust Quantum Materials Discovery System Initialized")
            logger.info(f"   Qubits: {config.num_qubits}")
            logger.info(f"   Quantum Layers: {config.quantum_layers}")
            logger.info(f"   Material Dimensions: {config.material_dimensions}")
            logger.info(f"   Max Retries: {config.max_retry_attempts}")
            logger.info(f"   Timeout: {config.timeout_seconds}s")
            
        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def discover_materials(self, target_properties: Dict[str, float],
                         num_candidates: int = 5) -> List[MaterialsResult]:
        """Robust materials discovery with comprehensive error handling."""
        
        if self.state != SystemState.READY:
            raise SystemError(f"System not ready, current state: {self.state}")
        
        self.state = SystemState.PROCESSING
        start_time = time.time()
        
        try:
            logger.info(f"🔬 Starting robust quantum materials discovery...")
            logger.info(f"   Target Properties: {target_properties}")
            logger.info(f"   Generating {num_candidates} candidates")
            
            # Validate inputs
            if self.config.validation_enabled:
                self.validator.validate_target_properties(target_properties)
            
            if num_candidates < 1 or num_candidates > 100:
                raise ValidationError(f"num_candidates must be between 1 and 100, got {num_candidates}")
            
            discovered_materials = []
            successful_candidates = 0
            total_attempts = 0
            
            while successful_candidates < num_candidates and total_attempts < num_candidates * 5:
                total_attempts += 1
                
                try:
                    logger.info(f"\n🧬 Generating Candidate {successful_candidates + 1}/{num_candidates} (Attempt {total_attempts})")
                    
                    result = self._generate_single_candidate(
                        target_properties, 
                        candidate_id=successful_candidates + 1
                    )
                    
                    if result and result.validate():
                        discovered_materials.append(result)
                        successful_candidates += 1
                        logger.info(f"   ✅ Candidate {successful_candidates} generated successfully")
                    else:
                        logger.warning(f"   ❌ Candidate generation failed validation")
                
                except Exception as candidate_error:
                    logger.error(f"   ❌ Candidate generation failed: {candidate_error}")
                    if not self.error_handler.handle_error(candidate_error, f"candidate_{total_attempts}"):
                        break
            
            if len(discovered_materials) == 0:
                raise ProcessingError("Failed to generate any valid materials candidates")
            
            # Sort by breakthrough score
            discovered_materials.sort(key=lambda x: x.breakthrough_score, reverse=True)
            
            total_time = time.time() - start_time
            logger.info(f"\n🎯 Robust discovery complete in {total_time:.2f}s")
            logger.info(f"   Successfully generated {len(discovered_materials)}/{num_candidates} candidates")
            logger.info(f"   Best breakthrough score: {discovered_materials[0].breakthrough_score:.3f}")
            logger.info(f"   Total attempts: {total_attempts}")
            
            self.state = SystemState.READY
            return discovered_materials
            
        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"❌ Materials discovery failed: {e}")
            raise
        finally:
            if self.state == SystemState.PROCESSING:
                self.state = SystemState.READY
    
    def _generate_single_candidate(self, target_properties: Dict[str, float],
                                 candidate_id: int) -> Optional[MaterialsResult]:
        """Generate a single materials candidate with retries."""
        
        result = MaterialsResult()
        
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                # Simulate microstructure generation
                microstructure_params = self._generate_microstructure_robust(target_properties)
                
                # Convert to quantum state
                classical_data = list(microstructure_params.values())
                quantum_state = self.quantum_simulator.prepare_superposition(classical_data)
                
                # Quantum evolution
                evolved_state = self.quantum_simulator.apply_quantum_evolution(
                    quantum_state, evolution_steps=15
                )
                
                # Measurement
                quantum_params = self.quantum_simulator.measure_quantum_state(evolved_state)
                
                # Convert to process parameters
                process_parameters = self._convert_to_process_parameters(quantum_params)
                
                # Validate process parameters
                if self.config.validation_enabled:
                    self.validator.validate_process_parameters(process_parameters)
                
                # Predict properties
                predicted_properties = self._predict_properties_robust(
                    process_parameters, microstructure_params
                )
                
                # Calculate metrics
                quantum_advantage = self._calculate_quantum_advantage(quantum_state, evolved_state)
                confidence = self._calculate_confidence_robust(predicted_properties, target_properties)
                breakthrough_score = self._calculate_breakthrough_score(
                    quantum_advantage, confidence, predicted_properties
                )
                quality_score = self._calculate_quality_score(
                    process_parameters, predicted_properties, confidence
                )
                
                # Populate result
                result.parameters = process_parameters
                result.predicted_properties = predicted_properties
                result.quantum_advantage = quantum_advantage
                result.confidence = confidence
                result.breakthrough_score = breakthrough_score
                result.quality_score = quality_score
                result.retry_count = attempt
                result.generation_time = time.time()
                
                logger.debug(f"Candidate {candidate_id} generated on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                result.add_error(f"Attempt {attempt + 1}: {str(e)}")
                logger.warning(f"Candidate {candidate_id} attempt {attempt + 1} failed: {e}")
                
                if attempt == self.config.max_retry_attempts:
                    logger.error(f"All attempts failed for candidate {candidate_id}")
                    return result
                
                # Brief pause before retry
                time.sleep(0.1 * (attempt + 1))
        
        return result
    
    def _generate_microstructure_robust(self, target_properties: Dict[str, float]) -> Dict[str, float]:
        """Generate microstructure parameters with robust error handling."""
        
        try:
            strength = target_properties.get('tensile_strength', 1000.0)
            ductility = target_properties.get('elongation', 10.0)
            density = target_properties.get('density', 0.95)
            grain_size = target_properties.get('grain_size', 50.0)
            
            # Clamp values to realistic ranges
            strength = max(400.0, min(2000.0, strength))
            ductility = max(1.0, min(30.0, ductility))
            density = max(0.70, min(1.0, density))
            grain_size = max(1.0, min(500.0, grain_size))
            
            microstructure_params = {
                'grain_size': grain_size * (0.8 + random.random() * 0.4),
                'phase_fraction': min(1.0, max(0.0, strength / 1200.0)),
                'porosity': max(0.001, min(0.3, 1.0 - density)),
                'texture_strength': max(0.0, min(1.0, ductility / 15.0)),
                'interface_density': max(10.0, min(10000.0, 1000.0 / grain_size))
            }
            
            # Validate microstructure parameters
            for param, value in microstructure_params.items():
                if math.isnan(value) or math.isinf(value):
                    logger.warning(f"Invalid microstructure parameter {param}, using default")
                    microstructure_params[param] = 0.5
            
            return microstructure_params
            
        except Exception as e:
            logger.error(f"Error generating microstructure: {e}")
            # Return default microstructure
            return {
                'grain_size': 50.0,
                'phase_fraction': 0.8,
                'porosity': 0.05,
                'texture_strength': 0.5,
                'interface_density': 20.0
            }
    
    def _convert_to_process_parameters(self, quantum_params: List[float]) -> Dict[str, float]:
        """Convert quantum parameters to process parameters with validation."""
        
        try:
            params = {}
            
            if len(quantum_params) >= 6:
                params['laser_power'] = max(50.0, min(500.0, 150.0 + quantum_params[0] * 100.0))
                params['scan_speed'] = max(100.0, min(2000.0, 600.0 + quantum_params[1] * 400.0))
                params['layer_thickness'] = max(10.0, min(100.0, 20.0 + quantum_params[2] * 20.0))
                params['hatch_spacing'] = max(20.0, min(300.0, 80.0 + quantum_params[3] * 80.0))
                params['powder_bed_temp'] = max(20.0, min(200.0, 60.0 + quantum_params[4] * 40.0))
                params['scan_strategy_angle'] = max(0.0, min(90.0, quantum_params[5] * 90.0))
            else:
                # Default parameters if not enough quantum params
                params = {
                    'laser_power': 200.0,
                    'scan_speed': 800.0,
                    'layer_thickness': 30.0,
                    'hatch_spacing': 120.0,
                    'powder_bed_temp': 80.0,
                    'scan_strategy_angle': 67.0
                }
            
            # Final validation and clamping
            for param, value in params.items():
                if math.isnan(value) or math.isinf(value):
                    logger.warning(f"Invalid process parameter {param}, using default")
                    params[param] = 200.0 if 'power' in param else 100.0
            
            return params
            
        except Exception as e:
            logger.error(f"Error converting to process parameters: {e}")
            return {
                'laser_power': 200.0,
                'scan_speed': 800.0,
                'layer_thickness': 30.0,
                'hatch_spacing': 120.0,
                'powder_bed_temp': 80.0,
                'scan_strategy_angle': 67.0
            }
    
    def _predict_properties_robust(self, process_params: Dict[str, float],
                                 microstructure: Dict[str, float]) -> Dict[str, float]:
        """Predict properties with robust error handling."""
        
        try:
            laser_power = process_params.get('laser_power', 200.0)
            scan_speed = process_params.get('scan_speed', 800.0)
            grain_size = microstructure.get('grain_size', 50.0)
            porosity = microstructure.get('porosity', 0.05)
            
            # Robust energy density calculation
            if scan_speed > 0:
                energy_density = laser_power / scan_speed
            else:
                energy_density = 0.25
                logger.warning("Scan speed is zero, using default energy density")
            
            # Property predictions with safety checks
            properties = {}
            
            # Tensile strength with Hall-Petch and porosity effects
            base_strength = 800.0
            hp_contribution = 500.0 / max(1.0, math.sqrt(grain_size))
            porosity_penalty = porosity * 2000.0
            
            properties['tensile_strength'] = max(400.0, min(2000.0, 
                base_strength + hp_contribution - porosity_penalty
            ))
            
            # Elongation
            base_elongation = 15.0
            porosity_effect = porosity * 100.0
            grain_effect = (grain_size - 50.0) * 0.1
            
            properties['elongation'] = max(1.0, min(30.0,
                base_elongation - porosity_effect + grain_effect
            ))
            
            # Density
            properties['density'] = max(0.70, min(1.0, 0.99 - porosity))
            
            # Grain size (affected by cooling rate)
            cooling_rate_factor = min(5.0, energy_density / 0.25)
            properties['grain_size'] = max(1.0, min(500.0,
                grain_size * (1.0 + cooling_rate_factor * 0.2)
            ))
            
            # Validate all properties
            for prop, value in properties.items():
                if math.isnan(value) or math.isinf(value):
                    logger.warning(f"Invalid predicted property {prop}, using fallback")
                    properties[prop] = {
                        'tensile_strength': 1000.0,
                        'elongation': 10.0,
                        'density': 0.95,
                        'grain_size': 50.0
                    }.get(prop, 1.0)
            
            return properties
            
        except Exception as e:
            logger.error(f"Error predicting properties: {e}")
            return {
                'tensile_strength': 1000.0,
                'elongation': 10.0,
                'density': 0.95,
                'grain_size': 50.0
            }
    
    def _calculate_confidence_robust(self, predicted: Dict[str, float],
                                   target: Dict[str, float]) -> float:
        """Calculate confidence with robust error handling."""
        
        try:
            if not predicted or not target:
                return 0.0
            
            total_error = 0.0
            num_properties = 0
            
            for prop, target_value in target.items():
                if prop in predicted:
                    predicted_value = predicted[prop]
                    
                    # Handle edge cases
                    if target_value == 0:
                        if predicted_value == 0:
                            error = 0.0
                        else:
                            error = 1.0
                    else:
                        relative_error = abs(predicted_value - target_value) / abs(target_value)
                        error = min(1.0, relative_error)  # Cap at 100% error
                    
                    total_error += error
                    num_properties += 1
            
            if num_properties == 0:
                return 0.0
            
            average_error = total_error / num_properties
            confidence = 1.0 / (1.0 + average_error)
            
            # Ensure valid confidence range
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _calculate_quantum_advantage(self, initial_state: List[complex],
                                   evolved_state: List[complex]) -> float:
        """Calculate quantum advantage with error handling."""
        
        try:
            if not initial_state or not evolved_state:
                return 0.0
            
            # Calculate entanglement increase safely
            initial_entanglement = sum(abs(amp)**2 for amp in initial_state)
            evolved_entanglement = sum(abs(amp)**2 for amp in evolved_state)
            
            entanglement_increase = abs(evolved_entanglement - initial_entanglement)
            
            # Calculate coherence preservation
            initial_coherence = abs(sum(initial_state))
            evolved_coherence = abs(sum(evolved_state))
            
            if initial_coherence < 1e-10:
                coherence_ratio = 1.0
            else:
                coherence_ratio = evolved_coherence / initial_coherence
            
            coherence_ratio = min(2.0, max(0.0, coherence_ratio))  # Clamp ratio
            
            # Quantum advantage score
            quantum_advantage = 0.5 * entanglement_increase + 0.5 * (coherence_ratio / 2.0)
            
            return min(1.0, max(0.0, quantum_advantage))
            
        except Exception as e:
            logger.error(f"Error calculating quantum advantage: {e}")
            return 0.0
    
    def _calculate_breakthrough_score(self, quantum_advantage: float,
                                    confidence: float,
                                    properties: Dict[str, float]) -> float:
        """Calculate breakthrough score with validation."""
        
        try:
            # Novelty score based on properties combination
            novelty = 0.0
            if 'tensile_strength' in properties and 'elongation' in properties:
                strength_score = min(1.0, max(0.0, properties['tensile_strength'] / 1200.0))
                ductility_score = min(1.0, max(0.0, properties['elongation'] / 15.0))
                novelty = strength_score * ductility_score
            
            # Ensure all components are valid
            quantum_advantage = min(1.0, max(0.0, quantum_advantage))
            confidence = min(1.0, max(0.0, confidence))
            novelty = min(1.0, max(0.0, novelty))
            
            # Combined breakthrough score
            breakthrough_score = (
                0.3 * quantum_advantage +
                0.4 * confidence +
                0.3 * novelty
            )
            
            return min(1.0, max(0.0, breakthrough_score))
            
        except Exception as e:
            logger.error(f"Error calculating breakthrough score: {e}")
            return 0.0
    
    def _calculate_quality_score(self, parameters: Dict[str, float],
                               properties: Dict[str, float],
                               confidence: float) -> float:
        """Calculate overall quality score."""
        
        try:
            # Parameter consistency score
            param_score = 1.0
            if 'laser_power' in parameters and 'scan_speed' in parameters:
                energy_density = parameters['laser_power'] / parameters['scan_speed']
                if 0.1 <= energy_density <= 1.0:  # Reasonable range
                    param_score = 1.0
                else:
                    param_score = 0.5
            
            # Property realism score
            prop_score = 1.0
            if 'tensile_strength' in properties and 'elongation' in properties:
                # Check for realistic strength-ductility trade-off
                strength = properties['tensile_strength']
                ductility = properties['elongation']
                
                # Typical inverse relationship
                expected_ductility = max(2.0, 25.0 - (strength - 800.0) / 40.0)
                ductility_error = abs(ductility - expected_ductility) / expected_ductility
                
                prop_score = 1.0 / (1.0 + ductility_error)
            
            # Combined quality score
            quality_score = 0.4 * param_score + 0.4 * prop_score + 0.2 * confidence
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0


def run_robust_quantum_materials():
    """Run the robust quantum materials discovery demonstration."""
    
    print("=" * 80)
    print("🛡️  ROBUST QUANTUM MATERIALS DISCOVERY")
    print("   Autonomous SDLC Generation 2: MAKE IT ROBUST")
    print("   Comprehensive Error Handling & Validation")
    print("=" * 80)
    
    try:
        # Initialize robust system
        config = QuantumMaterialsConfig(
            num_qubits=8,
            quantum_layers=3,
            material_dimensions=6,
            target_properties=4,
            max_retry_attempts=3,
            timeout_seconds=30.0,
            validation_enabled=True,
            safety_checks=True
        )
        
        discovery_system = RobustQuantumMaterialsDiscovery(config)
        
        # Define challenging target properties
        target_properties = {
            'tensile_strength': 1300.0,  # High strength
            'elongation': 15.0,          # Good ductility
            'density': 0.98,             # High density
            'grain_size': 35.0           # Fine grain
        }
        
        # Discover materials with error handling
        logger.info("Starting robust materials discovery...")
        results = discovery_system.discover_materials(target_properties, num_candidates=5)
        
        # Display results
        print("\n" + "=" * 80)
        print("🏆 ROBUST BREAKTHROUGH RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results[:3]):  # Show top 3
            print(f"\n🥇 Candidate {i + 1} (Score: {result.breakthrough_score:.3f})")
            print(f"   Quality Score: {result.quality_score:.3f}")
            print(f"   Retry Count: {result.retry_count}")
            print(f"   Validation: {result.validation_status}")
            
            print("   Process Parameters:")
            for param, value in result.parameters.items():
                print(f"      {param}: {value:.2f}")
            
            print("   Predicted Properties:")
            for prop, value in result.predicted_properties.items():
                print(f"      {prop}: {value:.2f}")
            
            print(f"   Quantum Advantage: {result.quantum_advantage:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            
            if result.error_messages:
                print(f"   Errors Encountered: {len(result.error_messages)}")
                for error in result.error_messages[:2]:  # Show first 2 errors
                    print(f"      - {error}")
        
        # Save robust results
        robust_data = {
            'timestamp': datetime.now().isoformat(),
            'generation': 2,
            'target_properties': target_properties,
            'config': {
                'num_qubits': config.num_qubits,
                'max_retry_attempts': config.max_retry_attempts,
                'timeout_seconds': config.timeout_seconds,
                'validation_enabled': config.validation_enabled
            },
            'results': [
                {
                    'parameters': result.parameters,
                    'predicted_properties': result.predicted_properties,
                    'quantum_advantage': result.quantum_advantage,
                    'confidence': result.confidence,
                    'breakthrough_score': result.breakthrough_score,
                    'quality_score': result.quality_score,
                    'retry_count': result.retry_count,
                    'validation_status': result.validation_status,
                    'error_count': len(result.error_messages)
                }
                for result in results
            ],
            'summary': {
                'total_candidates': len(results),
                'best_score': results[0].breakthrough_score,
                'average_retries': sum(r.retry_count for r in results) / len(results),
                'success_rate': len([r for r in results if r.validation_status == 'passed']) / len(results)
            }
        }
        
        with open('robust_quantum_breakthrough.json', 'w') as f:
            json.dump(robust_data, f, indent=2)
        
        print(f"\n💾 Robust results saved to robust_quantum_breakthrough.json")
        print(f"🎯 Best breakthrough score: {results[0].breakthrough_score:.3f}")
        print(f"🛡️  Average retries per candidate: {robust_data['summary']['average_retries']:.1f}")
        print(f"✅ Success rate: {robust_data['summary']['success_rate']:.1%}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Critical error in robust discovery: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    try:
        results = run_robust_quantum_materials()
        print("\n✅ GENERATION 2: MAKE IT ROBUST - SUCCESS!")
        print("🛡️  Robust quantum materials discovery with comprehensive error handling operational!")
        
    except Exception as e:
        print(f"\n❌ Critical failure: {e}")
        sys.exit(1)