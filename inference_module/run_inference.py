import argparse
import time
from pathlib import Path
from typing import Optional
import torch
from PIL import Image

from models.sd_loader import StableDiffusionLoader
from utils.device_utils import get_memory_stats, clear_memory
from configs import model_config


def setup_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inference Script")
    
    # Model configuration
    parser.add_argument(
        "--model", type=str,
        default=str(Path(model_config.SD_MODELS_PATH) / "v1-5-pruned.safetensors"),
        help="Path to the model file"
    )
    parser.add_argument(
        "--vae", type=str,
        default=None,
        help="Path to custom VAE file"
    )
    parser.add_argument(
        "--model-type", type=str,
        choices=["sd1", "sd2", "sd_xl", "sd3"],
        default=None,
        help="Model type (will auto-detect if not specified)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--prompt", type=str,
        required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative-prompt", type=str,
        default="",
        help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--steps", type=int,
        default=model_config.NUM_INFERENCE_STEPS,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float,
        default=model_config.GUIDANCE_SCALE,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--seed", type=int,
        default=None,
        help="Random seed for generation"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", type=str,
        default="output.png",
        help="Output image path"
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=1,
        help="Number of images to generate"
    )
    
    # Device configuration
    parser.add_argument(
        "--device", type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--dtype", type=str,
        choices=["float16", "float32"],
        default=model_config.DTYPE,
        help="Data type for inference"
    )
    
    return parser.parse_args()


def generate_image(
    model: StableDiffusionLoader,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: Optional[int] = None,
    **kwargs
) -> Image.Image:
    """Generate a single image with the given parameters."""
    if seed is not None:
        torch.manual_seed(seed)
        
    start_time = time.time()
    
    image = model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        **kwargs
    )
    
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")
    
    if torch.cuda.is_available():
        stats = get_memory_stats()
        print("CUDA Memory Stats (MB):")
        print(f"  Allocated: {stats['allocated']:.2f}")
        print(f"  Cached: {stats['cached']:.2f}")
        print(f"  Max Allocated: {stats['max_allocated']:.2f}")
    
    return image


def main():
    args = setup_args()
    
    print("Initializing model...")
    model = StableDiffusionLoader(
        model_path=args.model,
        vae_path=args.vae,
        model_type=args.model_type,
        device=args.device,
        dtype=torch.float16 if args.dtype == "float16" else torch.float32
    )
    
    output_path = Path(args.output)
    if args.batch_size > 1:
        # If generating multiple images, create numbered outputs
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        for i in range(args.batch_size):
            print(f"\nGenerating image {i+1}/{args.batch_size}")
            print(f"Prompt: {args.prompt}")
            if args.negative_prompt:
                print(f"Negative prompt: {args.negative_prompt}")
            
            image = generate_image(
                model=model,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed
            )
            
            # Save the image
            if args.batch_size > 1:
                save_path = output_path / f"{stem}_{i+1}{suffix}"
            else:
                save_path = output_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            image.save(save_path)
            print(f"Saved image to: {save_path}")
            
            # Clear some memory between generations
            clear_memory()
    
    finally:
        print("\nCleaning up...")
        model.unload()
        clear_memory()


if __name__ == "__main__":
    main() 