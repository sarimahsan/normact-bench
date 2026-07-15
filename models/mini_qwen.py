import torch
import torch.nn as nn
from transformer_research.components.block import TransformerBlock
from transformer_research.components.norms import build_norm
from transformer_research.components.rope import build_rope_cache

class MiniQwen(nn.Module):
    """
    MiniQwen model configured via the ModelConfig class/dictionary.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_dim = getattr(config, "hidden_dim", 256)
        if isinstance(config, dict):
            self.hidden_dim = config.get("hidden_dim", 256)
            
        self.max_seq_len = getattr(config, "max_seq_len", 256)
        if isinstance(config, dict):
            self.max_seq_len = config.get("max_seq_len", 256)
            
        vocab_size = getattr(config, "vocab_size", 50257)
        if isinstance(config, dict):
            vocab_size = config.get("vocab_size", 50257)
            
        num_layers = getattr(config, "num_layers", 4)
        if isinstance(config, dict):
            num_layers = config.get("num_layers", 4)
            
        num_heads = getattr(config, "num_heads", 8)
        if isinstance(config, dict):
            num_heads = config.get("num_heads", 8)
            
        use_gqa = getattr(config, "gqa", True)
        if isinstance(config, dict):
            use_gqa = config.get("gqa", True)

        num_kv_heads = getattr(config, "num_kv_heads", 2)
        if isinstance(config, dict):
            num_kv_heads = config.get("num_kv_heads", 2)

        if not use_gqa:
            num_kv_heads = num_heads
            
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if isinstance(config, dict):
            tie_word_embeddings = config.get("tie_word_embeddings", True)

        self.embed = nn.Embedding(vocab_size, self.hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_dim, num_heads, num_kv_heads, config=config)
            for _ in range(num_layers)
        ])

        self.norm = build_norm(config, self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, vocab_size, bias=False)

        if tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.last_cosine_similarities = []

    def forward(self, x):
        """
        x: (batch, seq)
        """
        import torch.nn.functional as F
        b, s = x.shape
        x = self.embed(x)

        # Dynamic RoPE Cache
        cos, sin = build_rope_cache(s, self.blocks[0].attn.head_dim, device=x.device)

        # Blocks forward pass
        cosine_sims = []
        for block in self.blocks:
            h_next = block(x, cos, sin)
            with torch.no_grad():
                cos_val = F.cosine_similarity(x, h_next, dim=-1).mean().item()
                cosine_sims.append(cos_val)
            x = h_next
        self.last_cosine_similarities = cosine_sims

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
