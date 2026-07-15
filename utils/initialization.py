import math
import torch
import torch.nn as nn

def initialize_weights(model, config):
    """
    Uniformly initializes linear, embedding, and normalization layers in a model.
    Also handles special scaling of residual layers based on the number of transformer layers.
    """
    initializer_range = getattr(config, "initializer_range", 0.02)
    num_layers = getattr(config, "num_layers", 4)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Normal weight initialization
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            
            # Special scaling for residual output projections (e.g. out_proj, w_down, w_out)
            # This balances variance across deep layers.
            if any(proj in name for proj in ["out_proj", "w_down", "w_out"]):
                with torch.no_grad():
                    module.weight.data.copy_(module.weight.data / math.sqrt(2.0 * num_layers))
                    
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            
        elif hasattr(module, "weight") and module.weight is not None:
            # For custom Norm classes and PyTorch norm layers
            # Check if name contains norm
            if "norm" in name or isinstance(module, (nn.LayerNorm, nn.modules.normalization.LayerNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
