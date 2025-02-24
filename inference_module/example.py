from pathlib import Path
import os
import argparse
from .models.sd_loader import StableDiffusionLoader


def setup_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A beautiful sunset over mountains, "
            "highly detailed digital art"
        ),
        help="Text prompt for image generation"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, worst quality",
        help="Negative prompt for image generation"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width of generated image"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated image"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image filename"
    )
    
    return parser.parse_args()


def main():
    args = setup_args()
    
    # Get absolute paths
    current_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    model_dir = current_dir / "stable-diffusion-webui/models/Stable-diffusion"
    model_path = model_dir / "realisticVisionV51_v51VAE.safetensors"
    
    # Initialize model loader
    sd_loader = StableDiffusionLoader(
        model_path=model_path,
        model_type="sd1"  # or "sd_xl", "sd2", "sd3"
    )
    
    # Generate image
    print(f"Generating image with prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Image size: {args.width}x{args.height}")
    
    image = sd_loader.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
    )
    
    # Save the generated image
    output_path = current_dir / args.output
    image.save(output_path)
    print(f"Image saved to {output_path}")
    
    # Clean up
    sd_loader.unload()


if __name__ == "__main__":
    main() 