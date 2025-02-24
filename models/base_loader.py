import os
from pathlib import Path
from typing import Optional, Union

from utils.device_utils import move_to_device, get_device, get_dtype
from configs.model_config import ModelConfig

import torch
from safetensors.torch import load_file as load_safetensors 