import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_research.analysis.loaders.experiment_loader import load_experiments

def plot_all(results_dir, output_dir):
    """
    Generates 9 publication-ready charts and reports from experiment results.
    """
    runs = load_experiments(results_dir)
    if not runs:
        print("No experiment runs found to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Reusable style configurations
    plt.rcParams.update({
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    # Group runs by label (configuration)
    grouped = {}
    for run in runs:
        label = run["label"]
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(run)

    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
    color_map = {label: colors[i] for i, label in enumerate(grouped.keys())}

    # 1. Validation Loss Curves
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        val_losses = []
        val_steps = None
        for run in group_runs:
            sub = run["metrics"].dropna(subset=["val_loss"])
            if len(sub) > 0:
                val_losses.append(sub["val_loss"].values)
                val_steps = sub["step"].values
        if val_losses:
            min_len = min(len(vl) for vl in val_losses)
            val_losses = np.array([vl[:min_len] for vl in val_losses])
            mean_loss = np.mean(val_losses, axis=0)
            std_loss = np.std(val_losses, axis=0)
            plt.plot(val_steps[:min_len], mean_loss, label=label, color=color, marker='o', markersize=4)
            plt.fill_between(val_steps[:min_len], mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)
    plt.title("Validation Loss Curves")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_loss_curves.png"), dpi=200)
    plt.close()

    # 2. Perplexity Curves
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        ppls = []
        val_steps = None
        for run in group_runs:
            sub = run["metrics"].dropna(subset=["val_loss"])
            if len(sub) > 0:
                ppls.append(sub["perplexity"].values)
                val_steps = sub["step"].values
        if ppls:
            min_len = min(len(p) for p in ppls)
            ppls = np.array([p[:min_len] for p in ppls])
            mean_ppl = np.mean(ppls, axis=0)
            std_ppl = np.std(ppls, axis=0)
            plt.plot(val_steps[:min_len], mean_ppl, label=label, color=color, marker='s', markersize=4)
            plt.fill_between(val_steps[:min_len], mean_ppl - std_ppl, mean_ppl + std_ppl, color=color, alpha=0.15)
    plt.title("Validation Perplexity Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_curves.png"), dpi=200)
    plt.close()

    # 3. Interaction Plot (Norm x Activation)
    plt.figure(figsize=(8, 5))
    # Collect final validation loss for each seed
    interaction_data = []
    for run in runs:
        sub = run["metrics"].dropna(subset=["val_loss"])
        if len(sub) > 0:
            final_loss = sub["val_loss"].values[-1]
            interaction_data.append({
                "norm": run["norm"].upper(),
                "activation": run["activation"].upper(),
                "val_loss": final_loss
            })
    if interaction_data:
        df_int = pd.DataFrame(interaction_data)
        # Pivot to get mean losses grouped by Norm and Activation
        means = df_int.groupby(["activation", "norm"])["val_loss"].mean().unstack()
        # Plot lines
        for norm in means.columns:
            plt.plot(means.index, means[norm], marker='o', label=norm, linewidth=2)
        plt.title("Interaction Plot (Normalization × Activation)")
        plt.xlabel("Activation Function")
        plt.ylabel("Mean Final Validation Loss")
        plt.legend(title="Normalization")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interaction_plot.png"), dpi=200)
    plt.close()

    # 4. Gradient Norm Curves Over Training
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        grad_norms = []
        steps = None
        for run in group_runs:
            df_g = run["activations"] # Note: gradients stats and activation stats are logged to activations/gradients.csv
            # We can read from gradients.csv directly by checking df_g or loading it if available
            grad_file = os.path.join(results_dir, run["hash"], run["seed"], "gradients.csv")
            if os.path.exists(grad_file):
                df_grad = pd.read_csv(grad_file)
                # Compute total grad norm at each step: sqrt(sum(grad_norm^2))
                # Group by step and calculate total grad norm
                total_grad = df_grad.groupby("step")["grad_norm"].apply(lambda g: np.sqrt(np.sum(g**2))).reset_index()
                grad_norms.append(total_grad["grad_norm"].values)
                steps = total_grad["step"].values
        if grad_norms:
            min_len = min(len(gn) for gn in grad_norms)
            grad_norms = np.array([gn[:min_len] for gn in grad_norms])
            mean_grad = np.mean(grad_norms, axis=0)
            std_grad = np.std(grad_norms, axis=0)
            plt.plot(steps[:min_len], mean_grad, label=label, color=color)
            plt.fill_between(steps[:min_len], mean_grad - std_grad, mean_grad + std_grad, color=color, alpha=0.1)
    plt.title("Total Gradient Norm Over Training")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_norm_curves.png"), dpi=200)
    plt.close()

    # 5. Hidden-State RMS by Layer
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        # Collect RMS values by layer at the final step
        layer_rms = {}
        for run in group_runs:
            df_act = run["activations"]
            if df_act is not None and len(df_act) > 0:
                # Get last logged step
                last_step = df_act["step"].max()
                df_last = df_act[df_act["step"] == last_step]
                for _, row in df_last.iterrows():
                    name = row["layer_name"]
                    if "rms_ffn_in" in name: # e.g. blocks.0.rms_ffn_in
                        try:
                            layer_idx = int(name.split(".")[1])
                            rms_val = float(row["act_rms"])
                            if layer_idx not in layer_rms:
                                layer_rms[layer_idx] = []
                            layer_rms[layer_idx].append(rms_val)
                        except ValueError:
                            pass
        if layer_rms:
            layers = sorted(layer_rms.keys())
            means = [np.mean(layer_rms[l]) for l in layers]
            stds = [np.std(layer_rms[l]) for l in layers]
            plt.errorbar(layers, means, yerr=stds, label=label, color=color, marker='o', capsize=4, elinewidth=1.5)
    plt.title("Hidden State RMS Magnitude by Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("RMS Magnitude (Final Step)")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hidden_state_rms_by_layer.png"), dpi=200)
    plt.close()

    # 6. Attention Entropy by Layer
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        layer_entropies = {}
        for run in group_runs:
            df_act = run["activations"]
            if df_act is not None and len(df_act) > 0:
                last_step = df_act["step"].max()
                df_last = df_act[df_act["step"] == last_step]
                for _, row in df_last.iterrows():
                    name = row["layer_name"]
                    if "attn.entropy" in name:
                        try:
                            layer_idx = int(name.split(".")[1])
                            ent_val = float(row["act_rms"])
                            if layer_idx not in layer_entropies:
                                layer_entropies[layer_idx] = []
                            layer_entropies[layer_idx].append(ent_val)
                        except ValueError:
                            pass
        if layer_entropies:
            layers = sorted(layer_entropies.keys())
            means = [np.mean(layer_entropies[l]) for l in layers]
            stds = [np.std(layer_entropies[l]) for l in layers]
            plt.errorbar(layers, means, yerr=stds, label=label, color=color, marker='x', capsize=4)
    plt.title("Attention Entropy by Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Entropy (Final Step)")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_entropy_by_layer.png"), dpi=200)
    plt.close()

    # 7. Training Throughput Comparison
    plt.figure(figsize=(8, 5))
    labels = []
    throughputs = []
    stds = []
    for label, group_runs in grouped.items():
        tps_runs = [run["metrics"]["tokens_per_second"].values for run in group_runs]
        avg_tps = [np.mean(tps[2:]) if len(tps) > 2 else np.mean(tps) for tps in tps_runs]
        labels.append(label)
        throughputs.append(np.mean(avg_tps))
        stds.append(np.std(avg_tps))
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, throughputs, yerr=stds, align='center', alpha=0.8, color=[color_map[l] for l in labels], capsize=10)
    plt.xticks(x_pos, labels, rotation=15)
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Training Throughput Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_throughput.png"), dpi=200)
    plt.close()

    # 8. Parameter Update Ratio Over Training
    plt.figure(figsize=(8, 5))
    for label, group_runs in grouped.items():
        color = color_map[label]
        update_ratios = []
        steps = None
        for run in group_runs:
            grad_file = os.path.join(results_dir, run["hash"], run["seed"], "gradients.csv")
            if os.path.exists(grad_file):
                df_grad = pd.read_csv(grad_file)
                # Average weight update ratio across parameters at each step
                avg_ratio = df_grad.groupby("step")["weight_update_ratio"].mean().reset_index()
                update_ratios.append(avg_ratio["weight_update_ratio"].values)
                steps = avg_ratio["step"].values
        if update_ratios:
            min_len = min(len(ur) for ur in update_ratios)
            update_ratios = np.array([ur[:min_len] for ur in update_ratios])
            mean_ur = np.mean(update_ratios, axis=0)
            std_ur = np.std(update_ratios, axis=0)
            plt.plot(steps[:min_len], mean_ur, label=label, color=color)
            plt.fill_between(steps[:min_len], mean_ur - std_ur, mean_ur + std_ur, color=color, alpha=0.1)
    plt.title("Parameter Update Ratio Over Training")
    plt.xlabel("Steps")
    plt.ylabel("Update Ratio (||dW|| / ||W||)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_update_ratio.png"), dpi=200)
    plt.close()

    # 9. Effect-Size Forest Plot (Cohen's d of RMSNorm vs LayerNorm)
    plt.figure(figsize=(8, 5))
    effect_sizes = []
    errors = []
    activations = []
    
    # We compare RMSNorm vs LayerNorm for each activation type
    act_groups = {}
    for run in runs:
        act = run["activation"].upper()
        if act not in act_groups:
            act_groups[act] = {"RMSNORM": [], "LAYERNORM": []}
        sub = run["metrics"].dropna(subset=["val_loss"])
        if len(sub) > 0:
            act_groups[act][run["norm"].upper()].append(sub["val_loss"].values[-1])
            
    for act, norms in act_groups.items():
        rms_vals = norms["RMSNORM"]
        ln_vals = norms["LAYERNORM"]
        if len(rms_vals) > 1 and len(ln_vals) > 1:
            mean_rms, mean_ln = np.mean(rms_vals), np.mean(ln_vals)
            var_rms, var_ln = np.var(rms_vals, ddof=1), np.var(ln_vals, ddof=1)
            n_rms, n_ln = len(rms_vals), len(ln_vals)
            
            pooled_std = np.sqrt(((n_rms - 1) * var_rms + (n_ln - 1) * var_ln) / (n_rms + n_ln - 2))
            cohens_d = (mean_rms - mean_ln) / (pooled_std + 1e-8)
            
            # Confidence interval error for Cohen's d: sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)))
            se_d = np.sqrt((n_rms + n_ln) / (n_rms * n_ln) + (cohens_d ** 2) / (2 * (n_rms + n_ln)))
            
            effect_sizes.append(cohens_d)
            errors.append(1.96 * se_d) # 95% Confidence Interval
            activations.append(act)
            
    if effect_sizes:
        y_pos = np.arange(len(activations))
        plt.errorbar(effect_sizes, y_pos, xerr=errors, fmt='o', color='black', capsize=5, markersize=8, elinewidth=2)
        plt.yticks(y_pos, activations)
        plt.axvline(x=0.0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel("Cohen's d (Negative indicates RMSNorm has lower loss)")
        plt.title("Effect-Size Forest Plot: RMSNorm vs LayerNorm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effect_size_forest_plot.png"), dpi=200)
    plt.close()
    
    print(f"All 9 publication plots successfully saved in: '{output_dir}'")
