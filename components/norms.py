import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) from scratch.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (LayerNorm) implemented from scratch for full transparency.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

def build_norm(config, dim):
    """
    Factory function to build a normalization layer based on configuration.
    """
    norm_type = getattr(config, "norm_type", "rmsnorm")
    if isinstance(config, dict):
        norm_type = config.get("norm_type", "rmsnorm")
        
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "layernorm":
        return LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
