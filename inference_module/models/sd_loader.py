from pathlib import Path
from typing import Optional, Union, Dict
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
)

from inference_module.models.base_loader import BaseModelLoader
from inference_module.configs import model_config


class StableDiffusionLoader(BaseModelLoader):
    def __init__(
        self,
        model_path: Union[str, Path],
        vae_path: Optional[Union[str, Path]] = None,
        model_type: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_path, device, dtype)
        self.vae_path = vae_path
        self.model_type = model_type or model_config.DEFAULT_MODEL_TYPE
        self.pipeline = None
        
    def _determine_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Determine model type from state dict if not specified."""
        if self.model_type:
            return self.model_type
            
        # Check for SDXL specific keys
        if "conditioner.embedders.1.model.ln_final.weight" in state_dict:
            return "sd_xl"
        # Check for SD2.x specific keys
        elif ("cond_stage_model.model.transformer.resblocks.0."
              "attn.in_proj_weight") in state_dict:
            return "sd2"
        # Check for SD3 specific keys
        elif "model.diffusion_model.x_embedder.proj.weight" in state_dict:
            return "sd3"
        # Default to SD1.x
        return "sd1"
        
    def _create_model(self) -> torch.nn.Module:
        """Create appropriate pipeline based on model type."""
        if self.model_type == "sd_xl":
            return StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=self.dtype,
                use_safetensors=self.model_path.suffix == '.safetensors'
            )
        return StableDiffusionPipeline.from_single_file(
            self.model_path,
            torch_dtype=self.dtype,
            use_safetensors=self.model_path.suffix == '.safetensors'
        )
        
    def _post_load_processing(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply post-processing configurations."""
        if model_config.VAE_TILING:
            model.vae.enable_tiling()
            
        if self.vae_path:
            vae = AutoencoderKL.from_single_file(
                self.vae_path,
                torch_dtype=self.dtype
            )
            model.vae = vae
            
        return model
        
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate image from prompt."""
        model = self.load_model()
        
        # Set default parameters from config if not specified
        if num_inference_steps is None:
            num_inference_steps = model_config.NUM_INFERENCE_STEPS
        if guidance_scale is None:
            guidance_scale = model_config.GUIDANCE_SCALE
            
        with torch.no_grad():
            output = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]
            
        return output 