"""Helper utility functions."""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import yaml


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, save_path: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint with metadata."""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metadata': metadata or {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_model_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                         optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metadata': checkpoint.get('metadata', {})
    }


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(prefer_gpu: bool = True) -> torch.device:
    """Setup compute device."""
    
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    
    result = {
        'system_memory_total_gb': memory.total / (1024**3),
        'system_memory_used_gb': memory.used / (1024**3),
        'system_memory_percent': memory.percent
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        result['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
    
    return result


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """Create directory structure from nested dictionary."""
    
    def _create_dirs(path: Path, struct: Dict[str, Any]) -> None:
        for name, content in struct.items():
            current_path = path / name
            
            if isinstance(content, dict):
                current_path.mkdir(exist_ok=True)
                _create_dirs(current_path, content)
            else:
                # Create file
                current_path.parent.mkdir(parents=True, exist_ok=True)
                if not current_path.exists():
                    current_path.touch()
    
    base = Path(base_path)
    base.mkdir(exist_ok=True)
    _create_dirs(base, structure)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def progress_bar(current: int, total: int, prefix: str = '', 
                suffix: str = '', length: int = 50) -> str:
    """Create a progress bar string."""
    
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    return f'\r{prefix} |{bar}| {percent:.1f}% {suffix}'


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with dot notation keys."""
    
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray],
               default: float = 0.0) -> Union[float, np.ndarray]:
    """Safe division with default value for zero denominator."""
    
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    else:
        return numerator / denominator if denominator != 0 else default


def moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average."""
    
    if window_size >= len(data):
        return [np.mean(data)] * len(data)
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window = data[start_idx:end_idx]
        result.append(np.mean(window))
    
    return result


def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
    """Calculate exponential moving average."""
    
    if not data:
        return []
    
    result = [data[0]]
    
    for i in range(1, len(data)):
        ema = alpha * data[i] + (1 - alpha) * result[i-1]
        result.append(ema)
    
    return result


def cosine_schedule(current_step: int, total_steps: int, 
                   min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Cosine annealing schedule."""
    
    if current_step >= total_steps:
        return min_value
    
    progress = current_step / total_steps
    cosine_value = 0.5 * (1 + np.cos(np.pi * progress))
    
    return min_value + (max_value - min_value) * cosine_value


def linear_schedule(current_step: int, total_steps: int,
                   start_value: float = 1.0, end_value: float = 0.0) -> float:
    """Linear schedule."""
    
    if current_step >= total_steps:
        return end_value
    
    progress = current_step / total_steps
    return start_value + (end_value - start_value) * progress


def warmup_schedule(current_step: int, warmup_steps: int, max_value: float = 1.0) -> float:
    """Linear warmup schedule."""
    
    if current_step >= warmup_steps:
        return max_value
    
    return max_value * (current_step / warmup_steps)


def ensure_tensor(data: Union[np.ndarray, torch.Tensor, List], 
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """Ensure data is a PyTorch tensor."""
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    else:
        data = data.to(dtype=dtype)
    
    if device is not None:
        data = data.to(device)
    
    return data


def ensure_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensure data is a NumPy array."""
    
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def batch_to_device(batch: Union[torch.Tensor, List, Dict], 
                   device: torch.device) -> Union[torch.Tensor, List, Dict]:
    """Move batch to device, handling nested structures."""
    
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    else:
        return batch


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} took {format_time(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None