import os
import pytest
import torch
import torch.nn as nn
import tempfile
import shutil

from transformer_research.components.norms import RMSNorm, LayerNorm, build_norm
from transformer_research.components.feedforward import FeedForward, build_ffn
from transformer_research.components.block import TransformerBlock
from transformer_research.models.builder import build_model
from transformer_research.utils.seed import set_seed
from transformer_research.utils.config_hash import get_config_hash

def test_rmsnorm():
    """
    Test RMSNorm output shape and mathematical output scaling.
    """
    x = torch.randn(4, 16, 32)
    norm = RMSNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    
    # Scale invariant test: RMSNorm output should have an RMS close to 1.0 (before multiplying by weight)
    # The default weight is 1s, so the output RMS should be approximately 1.0.
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    # It should be very close to 1.0
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_layernorm():
    """
    Test custom LayerNorm output shape and mean/variance scaling.
    """
    x = torch.randn(4, 16, 32) * 5 + 10.0
    norm = LayerNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    
    # LayerNorm outputs should have mean 0 and variance 1 (before scale/shift by parameters)
    # Weight defaults to 1s, bias defaults to 0s, so output mean should be 0, variance should be 1
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

def test_feedforward_shapes():
    """
    Test FFN shapes for all three activation types: GELU, ReLU^2, SwiGLU.
    """
    x = torch.randn(2, 8, 32)
    
    # SwiGLU FFN
    ffn_swiglu = FeedForward({"activation_type": "swiglu", "hidden_dim": 32})
    out = ffn_swiglu(x)
    assert out.shape == x.shape
    
    # GELU FFN
    ffn_gelu = FeedForward({"activation_type": "gelu", "hidden_dim": 32})
    out = ffn_gelu(x)
    assert out.shape == x.shape
    
    # ReLU2 FFN
    ffn_relu2 = FeedForward({"activation_type": "relu2", "hidden_dim": 32})
    out = ffn_relu2(x)
    assert out.shape == x.shape

def test_transformer_block_configurations():
    """
    Verify that TransformerBlock builds and forwards successfully for all 6 configurations.
    """
    batch_size = 2
    seq_len = 8
    hidden_dim = 16
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    cos = torch.randn(seq_len, hidden_dim // 4)
    sin = torch.randn(seq_len, hidden_dim // 4)
    
    norms = ["rmsnorm", "layernorm"]
    activations = ["gelu", "relu2", "swiglu"]
    
    for norm in norms:
        for act in activations:
            config = {
                "norm_type": norm,
                "activation_type": act,
                "hidden_dim": hidden_dim,
                "num_heads": 4,
                "num_kv_heads": 2
            }
            
            block = TransformerBlock(hidden_dim, num_heads=4, num_kv_heads=2, config=config)
            out = block(x, cos, sin)
            
            assert out.shape == x.shape
            # Verify hidden state diagnostics are tracking values
            assert block.last_rms_attn_in > 0
            assert block.last_rms_attn_out > 0
            assert block.last_rms_ffn_in > 0
            assert block.last_rms_ffn_out > 0
            assert block.attn.last_attn_entropy >= 0

def test_full_model_training_step():
    """
    Performs a full training step (forward, backward, optimizer update)
    on MiniQwen to ensure weight updates function and gradients flow.
    """
    config = {
        "vocab_size": 100,
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_seq_len": 16,
        "tie_word_embeddings": True,
        "norm_type": "rmsnorm",
        "activation_type": "swiglu"
    }
    
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Input batch of token IDs
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    
    # Forward pass
    logits = model(x)
    assert logits.shape == (2, 8, 100)
    
    # Compute loss
    loss = nn.CrossEntropyLoss()(logits.view(-1, 100), y.view(-1))
    assert loss.item() > 0
    
    # Save parameter baseline
    p_before = model.embed.weight.clone().detach()
    
    # Backward pass & update
    loss.backward()
    optimizer.step()
    
    # Verify weights actually updated
    p_after = model.embed.weight.detach()
    assert not torch.equal(p_before, p_after)

def test_model_serialization():
    """
    Test loading and saving states to ensure serialization consistency.
    """
    config = {
        "vocab_size": 50,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "num_kv_heads": 1,
        "max_seq_len": 8,
        "tie_word_embeddings": False,
        "norm_type": "layernorm",
        "activation_type": "gelu"
    }
    
    model1 = build_model(config)
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
    
    try:
        # Save state
        torch.save({"model_state_dict": model1.state_dict()}, checkpoint_path)
        
        # Load state into new model instance
        model2 = build_model(config)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint["model_state_dict"])
        
        # Verify weight equality
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    finally:
        shutil.rmtree(temp_dir)

def test_config_hashing():
    """
    Test that identical configurations yield identical hashes,
    and different configs yield different hashes.
    """
    config1 = {"vocab_size": 100, "hidden_dim": 32, "norm_type": "rmsnorm", "activation_type": "swiglu"}
    config2 = {"hidden_dim": 32, "vocab_size": 100, "activation_type": "swiglu", "norm_type": "rmsnorm"} # Reordered keys
    config3 = {"vocab_size": 100, "hidden_dim": 32, "norm_type": "layernorm", "activation_type": "swiglu"} # Different norm
    
    hash1 = get_config_hash(config1)
    hash2 = get_config_hash(config2)
    hash3 = get_config_hash(config3)
    
    assert hash1 == hash2
    assert hash1 != hash3

def test_qk_norm_rope_gqa_ablation_configs():
    """
    Test that qk_norm, rope, and gqa configuration flags successfully
    influence model instantiation and attention shapes.
    """
    base_config = {
        "vocab_size": 100,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_seq_len": 16,
        "tie_word_embeddings": True,
        "norm_type": "rmsnorm",
        "activation_type": "swiglu"
    }

    # 1. qk_norm = False
    cfg_no_qknorm = base_config.copy()
    cfg_no_qknorm["qk_norm"] = False
    model_no_qknorm = build_model(cfg_no_qknorm)
    assert not hasattr(model_no_qknorm.blocks[0].attn, "q_norm")
    assert not hasattr(model_no_qknorm.blocks[0].attn, "k_norm")

    # 2. qk_norm = True (default)
    cfg_qknorm = base_config.copy()
    cfg_qknorm["qk_norm"] = True
    model_qknorm = build_model(cfg_qknorm)
    assert hasattr(model_qknorm.blocks[0].attn, "q_norm")
    assert hasattr(model_qknorm.blocks[0].attn, "k_norm")

    # 3. rope = False
    cfg_no_rope = base_config.copy()
    cfg_no_rope["rope"] = False
    model_no_rope = build_model(cfg_no_rope)
    assert model_no_rope.blocks[0].attn.use_rope is False

    # 4. gqa = False (should set num_kv_heads to num_heads)
    cfg_no_gqa = base_config.copy()
    cfg_no_gqa["gqa"] = False
    model_no_gqa = build_model(cfg_no_gqa)
    assert model_no_gqa.blocks[0].attn.num_kv_heads == 4
    assert model_no_gqa.blocks[0].attn.n_rep == 1

def test_custom_optimizer_stats():
    """
    Verify that our custom AdamW optimizer computes last_stats correctly.
    """
    params = [nn.Parameter(torch.randn(2, 2))]
    from transformer_research.utils.optimizer import AdamW
    opt = AdamW(params, lr=1e-3, weight_decay=0.01)
    
    # Run a step
    params[0].grad = torch.randn(2, 2)
    opt.step()
    
    assert opt.last_stats is not None
    assert "mean_update_magnitude" in opt.last_stats
    assert "update_variance" in opt.last_stats
    assert "mean_denom" in opt.last_stats
    assert "var_denom" in opt.last_stats
    assert opt.last_stats["mean_update_magnitude"] > 0

def test_save_experiment_manifest():
    """
    Test that save_experiment_manifest successfully writes an experiment.json
    manifest with all the expected structure and keys.
    """
    import tempfile
    import json
    import shutil
    from transformer_research.experiments.run_all import save_experiment_manifest
    from transformer_research.models.builder import build_model
    from transformer_research.experiments.run_all import ModelConfig

    config_dict = {
        "vocab_size": 100,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_seq_len": 16,
        "norm_type": "rmsnorm",
        "activation_type": "swiglu",
        "batch_size": 4
    }
    config = ModelConfig(**config_dict)
    model = build_model(config)

    # Mock Trainer
    class DummyHistoryEntry:
        def __init__(self):
            self.history = [{"tokens_per_second": 100.0}]
    
    class DummyTrainer:
        def __init__(self):
            self.global_step = 10
            self.total_tokens_seen = 640
            self.train_start_time = 0.0
            self.validation_history = [
                {"epoch": 1, "step": 5, "val_loss": 2.5, "perplexity": 12.18},
                {"epoch": 2, "step": 10, "val_loss": 2.3, "perplexity": 9.97}
            ]
            self.callbacks = [DummyHistoryEntry()]
            
    trainer = DummyTrainer()
    temp_dir = tempfile.mkdtemp()
    
    try:
        save_experiment_manifest(
            run_dir=temp_dir,
            config=config,
            config_dict=config_dict,
            model=model,
            trainer=trainer,
            seed=42,
            config_hash="dummy_hash"
        )
        
        manifest_path = os.path.join(temp_dir, "experiment.json")
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        assert manifest["experiment_id"] == "dummy_hash"
        assert manifest["seed"] == 42
        assert manifest["activation_type"] == "swiglu"
        assert manifest["norm_type"] == "rmsnorm"
        assert manifest["dataset"]["total_tokens_seen"] == 640
        assert manifest["model_info"]["parameters_total"] > 0
        assert manifest["results"]["best_validation_loss"] == 2.3
        assert len(manifest["learning_dynamics"]) == 2
        
    finally:
        shutil.rmtree(temp_dir)
