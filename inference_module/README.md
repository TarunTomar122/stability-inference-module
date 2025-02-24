# Stable Diffusion Inference Module

A clean, modular implementation for running inference with Stable Diffusion models. This module provides a simple interface for loading and running inference with different versions of Stable Diffusion models (SD1.x, SD2.x, SDXL, and SD3).

## Features

- Support for multiple Stable Diffusion versions (SD1.x, SD2.x, SDXL, SD3)
- Automatic model type detection
- Custom VAE support
- Memory optimization features
- Device management (CPU/CUDA)
- Support for both safetensors and PyTorch model formats

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the model paths in `configs/model_config.py` to point to your model files.

## Usage

Basic usage example:

```python
from models.sd_loader import StableDiffusionLoader

# Initialize model loader
sd_loader = StableDiffusionLoader(
    model_path="path/to/model.safetensors",
    vae_path="path/to/vae.safetensors",  # Optional
    model_type="sd_xl"  # Optional, will auto-detect if not specified
)

# Generate image
image = sd_loader.generate(
    prompt="A beautiful sunset over mountains",
    negative_prompt="blurry, low quality",
    num_inference_steps=30,
    guidance_scale=7.5
)

# Save the image
image.save("output.png")

# Clean up
sd_loader.unload()
```

See `example.py` for a complete example.

## Configuration

The module can be configured through `configs/model_config.py`. Key settings include:

- Model paths
- Device settings (CUDA/CPU)
- Memory optimization options
- Default inference parameters

## Memory Optimization Features

The module includes several memory optimization features that can be enabled in the config:

- Attention slicing
- VAE tiling
- xFormers memory efficient attention
- Model offloading
- Sequential loading

## Supported Model Types

- SD1.x (Stable Diffusion v1.x)
- SD2.x (Stable Diffusion v2.x)
- SDXL (Stable Diffusion XL)
- SD3 (Stable Diffusion 3)

## Contributing

Feel free to submit issues and enhancement requests! 