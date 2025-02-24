from pathlib import Path
import os

# Base paths
MODELS_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "stable-diffusion-webui"
SD_MODELS_PATH = MODELS_ROOT / "models"
VAE_MODELS_PATH = SD_MODELS_PATH / "VAE"
CLIP_MODELS_PATH = SD_MODELS_PATH / "CLIP"
UPSCALER_MODELS_PATH = SD_MODELS_PATH / "ESRGAN"
CODEFORMER_MODELS_PATH = SD_MODELS_PATH / "CodeFormer"

# Model URLs and filenames
CLIP_G_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors"
CLIP_L_URL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors"
T5_URL = "https://huggingface.co/DeepFloyd/IF-I-XL-v1.0/resolve/main/text_encoder.safetensors"

# Device configurations
ENABLE_CUDA = True
CUDA_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"
DTYPE = "float16"  # or float32

# Model specific settings
DEFAULT_MODEL_TYPE = "sd_xl"  # one of: sd1, sd2, sd_xl, sd3
VAE_TILING = False
ENABLE_XFORMERS = True
ENABLE_CUDA_GRAPH = True

# Inference settings
BATCH_SIZE = 1
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Memory optimization
SEQUENTIAL_OFFLOAD = False
MODEL_CPU_OFFLOAD = False
ATTENTION_SLICING = True
VAE_SLICING = True 