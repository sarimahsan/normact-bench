import torch
import torch.nn as nn
from transformer_research.components.attention import Qwen3Attention
from transformer_research.components.norms import build_norm
from transformer_research.components.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, config=None):
        super().__init__()
        if config is None:
            # Fallback configuration
            class DummyConfig:
                norm_type = "rmsnorm"
                activation_type = "swiglu"
            config = DummyConfig()

        self.attn_norm = build_norm(config, hidden_dim)
        self.attn = Qwen3Attention(hidden_dim, num_heads, num_kv_heads, config=config)

        self.ffn_norm = build_norm(config, hidden_dim)
        self.ffn = FeedForward(config, hidden_dim)

        # Diagnostic attributes to track hidden state magnitudes (RMS)
        self.last_rms_attn_in = 0.0
        self.last_rms_attn_out = 0.0
        self.last_rms_ffn_in = 0.0
        self.last_rms_ffn_out = 0.0

    def forward(self, x, cos, sin):
        # x: (batch, seq, hidden_dim)
        
        # Calculate RMS before attention norm
        with torch.no_grad():
            self.last_rms_attn_in = torch.sqrt(x.pow(2).mean()).item()

        # Attention block
        residual = x
        x_normed = self.attn_norm(x)
        
        # Calculate RMS after attention norm
        with torch.no_grad():
            self.last_rms_attn_out = torch.sqrt(x_normed.pow(2).mean()).item()
            
        attn_out = self.attn(x_normed, cos, sin)
        x = residual + attn_out

        # Calculate RMS before FFN norm
        with torch.no_grad():
            self.last_rms_ffn_in = torch.sqrt(x.pow(2).mean()).item()

        # FFN block
        residual = x
        x_normed = self.ffn_norm(x)

        # Calculate RMS after FFN norm
        with torch.no_grad():
            self.last_rms_ffn_out = torch.sqrt(x_normed.pow(2).mean()).item()

        ffn_out = self.ffn(x_normed)
        x = residual + ffn_out

        return x
