import torch
from typing import Union, Optional
from ..configs import model_config


def get_device() -> torch.device:
    """Get the appropriate device based on configuration and availability."""
    if model_config.ENABLE_CUDA and torch.cuda.is_available():
        return torch.device(model_config.CUDA_DEVICE)
    return torch.device(model_config.CPU_DEVICE)


def get_dtype() -> torch.dtype:
    """Get the appropriate dtype based on configuration."""
    device = get_device()
    if device.type == "cpu":
        return torch.float32
    if model_config.DTYPE == "float16":
        return torch.float16
    return torch.float32


def move_to_device(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None
) -> torch.nn.Module:
    """Move model to specified device with proper configuration."""
    if device is None:
        device = get_device()
    
    dtype = get_dtype()
    model = model.to(device=device, dtype=dtype)
    
    if model_config.ATTENTION_SLICING:
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing()
    
    # Only enable xformers on CUDA devices
    if device.type == "cuda" and model_config.ENABLE_XFORMERS:
        try:
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                model.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            print("xformers not found, skipping memory optimization")
    
    return model


def clear_memory():
    """Clear CUDA memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_stats() -> dict:
    """Get current memory statistics."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
        "cached": torch.cuda.memory_reserved() / 1024**2,      # MB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
    } 