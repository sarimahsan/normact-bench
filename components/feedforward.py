import torch
import torch.nn as nn
from transformer_research.components.activations import GELU, ReLUSquared, SiLU

class GELUFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_in = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = GELU()
        self.w_out = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_out(self.act(self.w_in(x)))

class ReLU2FFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_in = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = ReLUSquared()
        self.w_out = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_out(self.act(self.w_in(x)))

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = SiLU()
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))

class FeedForward(nn.Module):
    """
    Unified FeedForward wrapper class.
    """
    def __init__(self, config, hidden_dim=None):
        super().__init__()
        self.net = build_ffn(config, hidden_dim)

    def forward(self, x):
        return self.net(x)

def build_ffn(config, hidden_dim=None):
    """
    Factory function to build the correct FeedForward module type based on configuration.
    """
    activation_type = getattr(config, "activation_type", "swiglu")
    if isinstance(config, dict):
        activation_type = config.get("activation_type", "swiglu")
        
    h_dim = hidden_dim
    if h_dim is None:
        h_dim = getattr(config, "hidden_dim", 256)
        if isinstance(config, dict):
            h_dim = config.get("hidden_dim", 256)
            
    if activation_type == "swiglu":
        return SwiGLUFFN(h_dim)
    elif activation_type == "gelu":
        return GELUFFN(h_dim)
    elif activation_type == "relu2":
        return ReLU2FFN(h_dim)
    else:
        raise ValueError(f"Unknown activation_type: {activation_type}")
