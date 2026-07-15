import torch

def count_parameters(model):
    """
    Returns total parameter count and trainable parameter count.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops_per_token(config, vocab_size):
    """
    Computes an analytical estimate of FLOPs per token for the forward and backward passes.
    Based on standard matrix multiplication sizes of attention projections, feed-forward layers, 
    and the vocabulary classification head.
    """
    h = getattr(config, "hidden_dim", 256)
    if isinstance(config, dict):
        h = config.get("hidden_dim", 256)
        
    L = getattr(config, "num_layers", 4)
    if isinstance(config, dict):
        L = config.get("num_layers", 4)
        
    num_heads = getattr(config, "num_heads", 8)
    if isinstance(config, dict):
        num_heads = config.get("num_heads", 8)
        
    num_kv_heads = getattr(config, "num_kv_heads", 2)
    if isinstance(config, dict):
        num_kv_heads = config.get("num_kv_heads", 2)
        
    act_type = getattr(config, "activation_type", "swiglu")
    if isinstance(config, dict):
        act_type = config.get("activation_type", "swiglu")
        
    head_dim = h // num_heads
    
    # 1. Attention Block (Forward FLOPs per token)
    # Q_proj: 2 * h * (num_heads * head_dim) = 2 * h^2
    # K_proj: 2 * h * (num_kv_heads * head_dim)
    # V_proj: 2 * h * (num_kv_heads * head_dim)
    qkv_proj_flops = 2 * h * h + 4 * h * (num_kv_heads * head_dim)
    # Output projection: 2 * h^2
    attn_out_proj_flops = 2 * h * h
    
    # 2. Feed-Forward Network Block (Forward FLOPs per token)
    ffn_dim = int(h * 4)
    if act_type == "swiglu":
        # SwiGLU has w_gate and w_up projections (2 * 2 * h * ffn_dim) plus down projection (2 * ffn_dim * h)
        ffn_flops = 4 * h * ffn_dim + 2 * ffn_dim * h
    else:
        # GELU/ReLU2 has input projection (2 * h * ffn_dim) plus output projection (2 * ffn_dim * h)
        ffn_flops = 2 * h * ffn_dim + 2 * ffn_dim * h

    # 3. Layer Normalizations (~5 FLOPs per activation per norm, 2 norms per block)
    norm_flops = 2 * (5 * h)
    
    # 4. Total Forward FLOPs per block
    block_flops = qkv_proj_flops + attn_out_proj_flops + ffn_flops + norm_flops
    
    # 5. Total Model Forward FLOPs per token (Layers * block + Embed + lm_head)
    # lm_head: 2 * h * vocab_size
    total_forward_flops = L * block_flops + 2 * h * vocab_size
    
    # 6. Backward pass is approximately 2 * Forward pass FLOPs. Total = 3 * Forward.
    total_training_flops = 3 * total_forward_flops
    
    return total_forward_flops, total_training_flops

def get_peak_memory():
    """
    Returns the peak GPU memory allocated in MB.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0
