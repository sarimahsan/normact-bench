import os
import yaml
import json
import pandas as pd

def load_experiments(results_dir):
    """
    Scans the results directory for experiment config hashes and seed folders.
    Loads configurations and metrics, grouping runs by architecture configuration.
    Supports experiment.json manifest if present, falling back to config.yaml.
    """
    runs = []
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        return runs

    for root, dirs, files in os.walk(results_dir):
        if "plots" in dirs:
            dirs.remove("plots")
            
        seed_dirs = [d for d in dirs if d.startswith("seed")]
        if seed_dirs:
            config_hash = os.path.basename(root)
            for seed in seed_dirs:
                seed_path = os.path.join(root, seed)
                if not os.path.isdir(seed_path):
                    continue

            config_path = os.path.join(seed_path, "config.yaml")
            metrics_path = os.path.join(seed_path, "metrics.csv")
            act_path = os.path.join(seed_path, "activations.csv")
            opt_path = os.path.join(seed_path, "optimizer_stats.csv")
            manifest_path = os.path.join(seed_path, "experiment.json")

            if os.path.exists(metrics_path):
                try:
                    df_metrics = pd.read_csv(metrics_path)
                    df_act = pd.read_csv(act_path) if os.path.exists(act_path) else None
                    df_opt = pd.read_csv(opt_path) if os.path.exists(opt_path) else None
                    
                    manifest = None
                    if os.path.exists(manifest_path):
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                    
                    # Fallback config loading
                    config = {}
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = yaml.safe_load(f)
                    
                    runs.append({
                        "hash": config_hash,
                        "seed": seed,
                        "norm": manifest["norm_type"] if manifest else config.get("norm_type", "unknown"),
                        "activation": manifest["activation_type"] if manifest else config.get("activation_type", "unknown"),
                        "label": f"{manifest['activation_type'].upper()} + {manifest['norm_type'].upper()}" if manifest else f"{config.get('activation_type', 'unknown').upper()} + {config.get('norm_type', 'unknown').upper()}",
                        "metrics": df_metrics,
                        "activations": df_act,
                        "optimizer_stats": df_opt,
                        "config": config,
                        "manifest": manifest
                    })
                except Exception as e:
                    print(f"Error loading experiment run in '{seed_path}': {e}")
    return runs
