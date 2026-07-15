import os
import sys
import yaml
import json
import time
import socket
import argparse
import platform
import subprocess
import torch

# Ensure the root folder is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformer_research.experiments.registry import EXPERIMENTS
from transformer_research.utils.config_hash import get_config_hash
from transformer_research.utils.seed import set_seed
from transformer_research.models.builder import build_model
from transformer_research.trainer.trainer import Trainer
from transformer_research.trainer.callbacks import LoggerCallback, CheckpointCallback, DiagnosticCallback, EarlyStoppingCallback

class ModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def to_dict(self):
        return self.__dict__.copy()

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "not_a_git_repository"

def get_cpu_info():
    try:
        import multiprocessing
        cpu_name = platform.processor() or "unknown_cpu"
        cores = multiprocessing.cpu_count()
        return f"{cpu_name}, {cores} cores"
    except Exception:
        return "unknown_cpu"

def get_flash_attention_status():
    try:
        return torch.backends.cuda.flash_sdp_enabled()
    except Exception:
        return False

def save_experiment_manifest(run_dir, config, config_dict, model, trainer, seed, config_hash):
    from transformer_research.analysis.metrics.profiler import count_parameters, estimate_flops_per_token
    total_params, trainable_params = count_parameters(model)
    flops_fwd, flops_train = estimate_flops_per_token(config, config_dict.get("vocab_size", 50257))
    
    gpu_model = "CPU"
    gpu_mem_total_mb = 0
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        gpu_mem_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        
    pytorch_commit = getattr(torch.version, "git_version", "unknown_commit")
    
    env_info = {
        "gpu_model": gpu_model,
        "gpu_memory_total_mb": gpu_mem_total_mb,
        "cpu_info": get_cpu_info(),
        "pytorch_version": torch.__version__,
        "pytorch_commit": pytorch_commit,
        "flash_attention_enabled": get_flash_attention_status()
    }
    
    throughputs = [r["tokens_per_second"] for r in trainer.callbacks[0].history]
    avg_throughput = sum(throughputs[2:]) / len(throughputs[2:]) if len(throughputs) > 2 else (sum(throughputs) / len(throughputs) if throughputs else 0.0)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    runtime_sec = time.time() - trainer.train_start_time if trainer.train_start_time else 0.0
    
    best_val_loss = float("inf")
    best_ppl = float("inf")
    for val_entry in trainer.validation_history:
        if val_entry["val_loss"] < best_val_loss:
            best_val_loss = val_entry["val_loss"]
            best_ppl = val_entry["perplexity"]
            
    manifest = {
        "experiment_id": config_hash,
        "experiment_name": f"{config_dict.get('activation_type', 'unknown')}_{config_dict.get('norm_type', 'unknown')}",
        "activation_type": config_dict.get("activation_type", "unknown"),
        "norm_type": config_dict.get("norm_type", "unknown"),
        "seed": seed,
        "dataset": {
            "dataset_name": config_dict.get("dataset_name", "roneneldan/TinyStories"),
            "tokenizer_name": config_dict.get("tokenizer_name", "gpt2"),
            "vocab_size": config_dict.get("vocab_size", 50257),
            "max_seq_len": config_dict.get("max_seq_len", 256),
            "total_tokens_seen": trainer.total_tokens_seen
        },
        "model_info": {
            "parameters_total": total_params,
            "parameters_trainable": trainable_params,
            "estimated_flops_forward": flops_fwd,
            "estimated_flops_training": flops_train
        },
        "environment": env_info,
        "results": {
            "steps": trainer.global_step,
            "best_validation_loss": best_val_loss if best_val_loss != float("inf") else None,
            "best_perplexity": best_ppl if best_ppl != float("inf") else None,
            "training_runtime_sec": runtime_sec,
            "average_throughput_tokens_per_sec": avg_throughput,
            "peak_memory_used_mb": peak_mem_mb,
            "git_commit": get_git_commit()
        },
        "learning_dynamics": trainer.validation_history
    }
    
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run_experiment(exp, args):
    base_config_path = "transformer_research/configs/base.yaml"
    with open(base_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Load experimental overrides
    with open(exp["config_file"], "r") as f:
        overrides = yaml.safe_load(f)
    config_dict.update(overrides)

    # Apply Quick Sweep overrides if requested
    if args.quick_sweep:
        print("\n>>> Applying QUICK SWEEP overrides for rapid execution...")
        config_dict.update({
            "max_samples": 200,
            "val_max_samples": 50,
            "epochs": 1,
            "val_interval": 2,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 1,
            "warmup_steps": 2
        })

    # Generate config hash
    config_hash = get_config_hash(config_dict)
    print(f"\n==========================================")
    print(f"Running Experiment: {exp['name']}")
    print(f"Config Hash: {config_hash}")
    print(f"Activation: {config_dict['activation_type']} | Norm: {config_dict['norm_type']}")
    print(f"==========================================")

    # Get parsed seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Lazy-load dataloader to prevent importing transformers/datasets during unit tests
    from transformer_research.utils.data import get_dataloader

    # Load datasets once per config hash (shared across seeds to minimize overhead)
    print("Loading datasets...")
    train_loader, tokenizer = get_dataloader(
        dataset_name=config_dict.get("dataset_name", "roneneldan/TinyStories"),
        dataset_config=config_dict.get("dataset_config", None),
        split="train",
        tokenizer_name=config_dict.get("tokenizer_name", "gpt2"),
        seq_len=config_dict.get("max_seq_len", 256),
        batch_size=config_dict.get("batch_size", 16),
        shuffle=True,
        max_samples=config_dict.get("max_samples", 150000)
    )

    val_loader, _ = get_dataloader(
        dataset_name=config_dict.get("dataset_name", "roneneldan/TinyStories"),
        dataset_config=config_dict.get("dataset_config", None),
        split="validation",
        tokenizer_name=config_dict.get("tokenizer_name", "gpt2"),
        seq_len=config_dict.get("max_seq_len", 256),
        batch_size=config_dict.get("batch_size", 16),
        shuffle=False,
        max_samples=config_dict.get("val_max_samples", 2000)
    )

    # Sync vocabulary size
    config_dict["vocab_size"] = len(tokenizer)
    
    # Instantiate config object
    config = ModelConfig(**config_dict)

    # Run for each seed
    for seed in seeds:
        print(f"\n---> Starting Run: Seed {seed}")
        set_seed(seed)
        config.seed = seed

        # Set up output directory: results/{config_hash}/seed{seed}/
        run_dir = os.path.join(args.results_dir, config_hash, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Save config.yaml
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Save reproducibility metadata
        repro_metadata = {
            "experiment_name": exp["name"],
            "config_hash": config_hash,
            "seed": seed,
            "git_commit": get_git_commit(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "hostname": socket.gethostname(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config_dict
        }
        with open(os.path.join(run_dir, "reproducibility.json"), "w") as f:
            json.dump(repro_metadata, f, indent=2)

        # Build model and optimizer
        model = build_model(config)
        
        if getattr(config, "use_custom_optimizer", False):
            # Custom AdamW implementation
            from transformer_research.utils.optimizer import AdamW
            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        else:
            # Native PyTorch AdamW
            use_fused = (torch.cuda.is_available() and config.precision != "fp32")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                fused=use_fused
            )

        # Callbacks
        callbacks = [
            LoggerCallback(run_dir, log_interval=2 if args.quick_sweep else 10),
            CheckpointCallback(run_dir),
            DiagnosticCallback(run_dir, log_interval=5 if args.quick_sweep else 100),
            EarlyStoppingCallback(patience=config.early_stopping_patience)
        ]

        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            callbacks=callbacks
        )

        # Execute training
        trainer.train()
        print(f"---> Run Complete: Seed {seed}")
        save_experiment_manifest(run_dir, config, config_dict, model, trainer, seed, config_hash)

def main():
    parser = argparse.ArgumentParser(description="Transformer Research Framework Experiment Manager")
    parser.add_argument("--quick_sweep", action="store_true", help="Run a small, fast subset sweep to test the pipeline")
    parser.add_argument("--results_dir", type=str, default="transformer_research/results", help="Directory to save experiment results")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of random seeds")
    parser.add_argument("--activation", type=str, default=None, choices=["gelu", "relu2", "swiglu"], help="Run only this activation type")
    parser.add_argument("--norm", type=str, default=None, choices=["layernorm", "rmsnorm"], help="Run only this norm type")
    args = parser.parse_args()

    experiments_to_run = EXPERIMENTS
    if args.activation:
        experiments_to_run = [e for e in experiments_to_run if e["activation_type"] == args.activation]
    if args.norm:
        experiments_to_run = [e for e in experiments_to_run if e["norm_type"] == args.norm]

    print(f"Starting experiment sweep over {len(experiments_to_run)} configurations...")
    for exp in experiments_to_run:
        run_experiment(exp, args)
        
    print("\nSweep Complete! Regenerating plots and statistical reports...")
    
    # Auto-generate plots and statistics after sweep completes
    try:
        from transformer_research.analysis import plot_all, compute_statistics
        
        plot_all(args.results_dir, os.path.join(args.results_dir, "plots"))
        compute_statistics(args.results_dir, args.results_dir)
        print("Plots and statistical analysis reports successfully updated.")
    except Exception as e:
        print(f"Error generating analysis: {e}")

if __name__ == "__main__":
    main()
