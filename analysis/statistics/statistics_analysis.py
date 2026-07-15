import os
import yaml
import json
import numpy as np
import pandas as pd
import math
from transformer_research.analysis.loaders.experiment_loader import load_experiments

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def compute_statistics(results_dir, output_dir):
    """
    Computes ANOVA, t-tests, 95% confidence intervals, and effect sizes (Cohen's d)
    for the final validation losses of each configuration.
    """
    runs = load_experiments(results_dir)
    if not runs:
        print("No experiments found for statistical analysis.")
        return

    # Extract final validation loss for each run
    data = []
    for run in runs:
        sub = run["metrics"].dropna(subset=["val_loss"])
        if len(sub) == 0:
            continue
        final_val_loss = sub["val_loss"].values[-1]
        final_ppl = sub["perplexity"].values[-1]
        
        # Calculate tokens per second (throughput)
        tps = run["metrics"]["tokens_per_second"].values
        avg_tps = np.mean(tps[2:]) if len(tps) > 2 else np.mean(tps)
        
        # Calculate peak memory
        peak_mem = run["metrics"]["gpu_memory_mb"].values
        max_mem = np.max(peak_mem) if len(peak_mem) > 0 else 0.0

        data.append({
            "label": run["label"],
            "activation": run["activation"].upper(),
            "norm": run["norm"].upper(),
            "seed": run["seed"],
            "val_loss": final_val_loss,
            "perplexity": final_ppl,
            "throughput": avg_tps,
            "peak_memory": max_mem
        })

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid final validation loss metrics found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "statistical_analysis.md")
    summary_path = os.path.join(output_dir, "summary_table.md")
    
    # 1. Descriptive Statistics
    desc_stats = df.groupby("label").agg(
        mean_loss=("val_loss", "mean"),
        std_loss=("val_loss", "std"),
        mean_ppl=("perplexity", "mean"),
        std_ppl=("perplexity", "std"),
        mean_throughput=("throughput", "mean"),
        mean_memory=("peak_memory", "mean"),
        count=("val_loss", "count")
    ).reset_index()

    # Add 95% Confidence Intervals
    cis_loss = []
    cis_ppl = []
    for _, row in desc_stats.iterrows():
        n = row["count"]
        # loss CI
        if n > 1:
            se_loss = row["std_loss"] / math.sqrt(n)
            ci_l = 1.96 * se_loss
        else:
            ci_l = 0.0
        cis_loss.append(ci_l)
        
        # ppl CI
        if n > 1:
            se_ppl = row["std_ppl"] / math.sqrt(n)
            ci_p = 1.96 * se_ppl
        else:
            ci_p = 0.0
        cis_ppl.append(ci_p)

    desc_stats["ci_95_loss"] = cis_loss
    desc_stats["ci_95_ppl"] = cis_ppl

    # Write summary table first
    summary_lines = []
    summary_lines.append("# RESULTS SUMMARY TABLE\n")
    summary_lines.append("| Configuration | Final Val Loss | Final Perplexity | Throughput (tok/sec) | Peak Memory (MB) | Runs |")
    summary_lines.append("|---|---|---|---|---|---|")
    for _, row in desc_stats.iterrows():
        summary_lines.append(
            f"| {row['label']} | "
            f"{row['mean_loss']:.4f} ± {row['ci_95_loss']:.4f} | "
            f"{row['mean_ppl']:.2f} ± {row['ci_95_ppl']:.2f} | "
            f"{row['mean_throughput']:.1f} | "
            f"{row['mean_memory']:.1f} | "
            f"{int(row['count'])} |"
        )
    
    with open(summary_path, "w") as f:
        f.writelines("\n".join(summary_lines))
    print(f"Summary table saved to '{summary_path}'")

    # Now write the full report
    report = []
    report.append("# STATISTICAL ANALYSIS REPORT\n")
    report.append("## Descriptive Statistics (Mean ± 95% Confidence Interval)\n")
    report.append("| Configuration | Final Val Loss | Final Perplexity | Throughput (tok/sec) | Peak Memory (MB) | Runs |")
    report.append("|---|---|---|---|---|---|")
    
    for _, row in desc_stats.iterrows():
        report.append(
            f"| {row['label']} | "
            f"{row['mean_loss']:.4f} ± {row['ci_95_loss']:.4f} | "
            f"{row['mean_ppl']:.2f} ± {row['ci_95_ppl']:.2f} | "
            f"{row['mean_throughput']:.1f} | "
            f"{row['mean_memory']:.1f} | "
            f"{int(row['count'])} |"
        )
    report.append("\n")

    # 2. Factorial Two-Way ANOVA & Hypothesis testing (Requires SciPy)
    if SCIPY_AVAILABLE:
        report.append("## Factorial Analysis: Two-Way ANOVA (Normalization × Activation)\n")
        try:
            grand_mean = df['val_loss'].mean()
            norms = df['norm'].unique()
            activations = df['activation'].unique()
            
            N = len(df)
            I = len(norms)
            J = len(activations)
            
            ss_total = ((df['val_loss'] - grand_mean) ** 2).sum()
            
            ss_norm = 0.0
            for norm in norms:
                sub = df[df['norm'] == norm]
                ss_norm += len(sub) * ((sub['val_loss'].mean() - grand_mean) ** 2)
                
            ss_act = 0.0
            for act in activations:
                sub = df[df['activation'] == act]
                ss_act += len(sub) * ((sub['val_loss'].mean() - grand_mean) ** 2)
                
            ss_inter = 0.0
            ss_w = 0.0
            for norm in norms:
                for act in activations:
                    sub = df[(df['norm'] == norm) & (df['activation'] == act)]
                    if len(sub) > 0:
                        cell_mean = sub['val_loss'].mean()
                        norm_mean = df[df['norm'] == norm]['val_loss'].mean()
                        act_mean = df[df['activation'] == act]['val_loss'].mean()
                        ss_inter += len(sub) * ((cell_mean - norm_mean - act_mean + grand_mean) ** 2)
                        ss_w += ((sub['val_loss'] - cell_mean) ** 2).sum()
                        
            df_norm = I - 1
            df_act = J - 1
            df_inter = (I - 1) * (J - 1)
            df_error = N - (I * J)
            
            ms_norm = ss_norm / df_norm
            ms_act = ss_act / df_act
            ms_inter = ss_inter / df_inter
            ms_error = ss_w / df_error if df_error > 0 else 1.0
            
            f_norm = ms_norm / ms_error if df_error > 0 else 0.0
            f_act = ms_act / ms_error if df_error > 0 else 0.0
            f_inter = ms_inter / ms_error if df_error > 0 else 0.0
            
            p_norm = stats.f.sf(f_norm, df_norm, df_error) if df_error > 0 else 1.0
            p_act = stats.f.sf(f_act, df_act, df_error) if df_error > 0 else 1.0
            p_inter = stats.f.sf(f_inter, df_inter, df_error) if df_error > 0 else 1.0
            
            report.append("| Source of Variation | Sum of Squares (SS) | Degrees of Freedom (df) | Mean Square (MS) | F-statistic | p-value | Significant ($p < 0.05$) |")
            report.append("|---|---|---|---|---|---|---|")
            report.append(f"| Normalization (Factor A) | {ss_norm:.6f} | {df_norm} | {ms_norm:.6f} | {f_norm:.4f} | {p_norm:.6f} | {'Yes' if p_norm < 0.05 else 'No'} |")
            report.append(f"| Activation (Factor B) | {ss_act:.6f} | {df_act} | {ms_act:.6f} | {f_act:.4f} | {p_act:.6f} | {'Yes' if p_act < 0.05 else 'No'} |")
            report.append(f"| Interaction (A × B) | {ss_inter:.6f} | {df_inter} | {ms_inter:.6f} | {f_inter:.4f} | {p_inter:.6f} | {'Yes' if p_inter < 0.05 else 'No'} |")
            report.append(f"| Error (Within) | {ss_w:.6f} | {df_error} | {ms_error:.6f} | - | - | - |")
            report.append(f"| Total | {ss_total:.6f} | {N-1} | - | - | - | - |")
            report.append("\n")
            
            if df_error <= 0:
                report.append("> [!WARNING]\n")
                report.append("> Insufficient degrees of freedom for error calculations. Run more seeds to validate ANOVA.\n\n")
        except Exception as e:
            report.append(f"Error computing ANOVA: {e}\n\n")

        # Pairwise T-tests comparing Norm types for each Activation function
        report.append("### Pairwise Comparisons: RMSNorm vs LayerNorm\n")
        report.append("| Activation | RMSNorm Loss (Mean) | LayerNorm Loss (Mean) | t-statistic | p-value | Cohen's d |")
        report.append("|---|---|---|---|---|---|")
        
        activations = df["activation"].unique()
        for act in activations:
            sub_rms = df[(df["activation"] == act) & (df["norm"] == "RMSNORM")]["val_loss"].values
            sub_ln = df[(df["activation"] == act) & (df["norm"] == "LAYERNORM")]["val_loss"].values
            
            if len(sub_rms) > 1 and len(sub_ln) > 1:
                t_stat, p_val = stats.ttest_ind(sub_rms, sub_ln, equal_var=False)
                
                mean1, mean2 = np.mean(sub_rms), np.mean(sub_ln)
                var1, var2 = np.var(sub_rms, ddof=1), np.var(sub_ln, ddof=1)
                pooled_std = math.sqrt((var1 + var2) / 2)
                cohens_d = (mean1 - mean2) / (pooled_std + 1e-8)
                
                report.append(
                    f"| {act} | {mean1:.4f} | {mean2:.4f} | "
                    f"{t_stat:.4f} | {p_val:.6f} | {cohens_d:.4f} |"
                )
        report.append("\n")
        
    else:
        report.append("## Hypothesis Testing\n")
        report.append("> [!NOTE]\n")
        report.append("> SciPy is not installed on this system. Advanced statistical hypothesis testing (ANOVA, t-tests) was skipped.\n")

    # Write Markdown file
    with open(report_path, "w") as f:
        f.writelines("\n".join(report))
        
    print(f"Statistical report saved to '{report_path}'")

if __name__ == "__main__":
    import sys
    r_dir = "transformer_research/results" if len(sys.argv) < 2 else sys.argv[1]
    o_dir = "transformer_research/results" if len(sys.argv) < 3 else sys.argv[2]
    compute_statistics(r_dir, o_dir)
