from pathlib import Path
from typing import Optional, Union, Dict

import torch
from safetensors.torch import load_file as load_safetensors

from inference_module.utils.device_utils import (
    move_to_device,
    get_device,
    get_dtype,
)


class BaseModelLoader:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or get_device()
        self.dtype = dtype or get_dtype()
        self.model = None
        
    def load_state_dict(self) -> dict:
        """Load state dict from file."""
        if self.model_path.suffix == '.safetensors':
            return load_safetensors(self.model_path)
        return torch.load(self.model_path, map_location='cpu')
    
    def _process_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process state dict before loading into model. Override if needed."""
        return state_dict
    
    def _create_model(self) -> torch.nn.Module:
        """Create model instance. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _post_load_processing(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply post-processing configurations."""
        return model
    
    def load_model(self) -> torch.nn.Module:
        """Load and prepare model for inference."""
        if self.model is not None:
            return self.model
            
        # Create and configure model
        model = self._create_model()
        model = move_to_device(model, self.device)
        model = self._post_load_processing(model)
        
        # Set to eval mode if applicable
        if hasattr(model, 'eval'):
            model.eval()
            
        self.model = model
        return model
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() 