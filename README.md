# Transformer Research Framework

A self-contained, modular, and configurable research platform designed to study the empirical interactions between normalizations and activation functions in autoregressive decoder-only Transformers. 

This framework is isolated and ready to run locally or on Google Colab.

---

## 📁 Directory Structure

```
transformer_research/
├── README.md               # Framework documentation
├── configs/                # Hyperparameter & architecture overrides
│   ├── base.yaml           # Base configurations (ablation flags, learning rates, etc.)
│   └── gelu_ln.yaml, etc.  # Specific norm & activation overrides
├── components/             # Transformer layer blocks
│   ├── norms.py            # Custom LayerNorm & RMSNorm implemented from scratch
│   ├── activations.py      # Custom activation functions (GELU, ReLU², SiLU)
│   ├── feedforward.py      # FFN variants (GELUFFN, ReLU2FFN, SwiGLUFFN)
│   ├── attention.py        # Attention layer with GQA, RoPE, and QK-Norm controls
│   ├── rope.py             # Dynamic Rotary Position Embeddings cache
│   └── block.py            # Transformer Block tracking hidden state magnitudes
├── models/                 # Model registry & building
│   ├── mini_qwen.py        # Configurable model stack with cosine similarity tracking
│   └── builder.py          # Unified model construction factory & weight init
├── trainer/                # Decoupled training engine
│   ├── trainer.py          # Autocast AMP precision, grad accum, & token tracking
│   └── callbacks.py        # Hook-driven Logging, Saturation tracking, & Optimizer Diagnostics
├── analysis/               # Evaluation & publication-ready plotting
│   ├── loaders/
│   │   └── experiment_loader.py # Unified runs loader using experiment.json manifests
│   ├── metrics/
│   │   └── profiler.py          # Trainable parameter counts & analytical FLOPs estimation
│   ├── plotting/
│   │   └── plots.py             # Generates the 9 publication-ready paper plots
│   └── statistics/
│       └── statistics_analysis.py # Two-way ANOVA, pairwise t-tests, Cohen's d, & summary table
├── experiments/            # Orchestration
│   ├── registry.py         # Maps combinations to configurations
│   └── run_all.py          # Seed sweep runner with manifest logging and environment capture
├── research/               # Academic records (Hypotheses, Outlines, Logs)
├── tests/                  # Integration and unit test suite
└── results/                # Directory where experiments, checkpoints, and charts are saved
```

---

## 🧱 Model Architecture & Pipeline Flow

The research framework employs a configurable, decoder-only Transformer model. Below is a generic visualization of the full network pipeline and the internal structure of each Transformer block:

```mermaid
graph TD
    %% Global styling
    classDef blockStyle fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef opStyle fill:#0f172a,stroke:#10b981,stroke-width:1px,color:#d1d5db;
    classDef flowStyle fill:#0f172a,stroke:#64748b,stroke-width:1px,color:#94a3b8;

    subgraph Input_Stage ["Input Stage"]
        Tokens["Token IDs (Batch, Seq)"] --> Embed["Embedding Layer"]
        Embed --> H0["Hidden States (h_0)"]
    end

    subgraph Transformer_Blocks ["Transformer Blocks (x N Layers)"]
        H0 --> Block1["Transformer Block 1"]
        Block1 --> Block2["Transformer Block 2"]
        Block2 --> Dots["..."]
        Dots --> BlockN["Transformer Block N (h_N)"]
    end

    subgraph Output_Stage ["Output Stage"]
        BlockN --> FinalNorm["Final Norm (LayerNorm / RMSNorm)"]
        FinalNorm --> LMHead["LM Head (Linear)"]
        LMHead --> Logits["Logits (Batch, Seq, Vocab)"]
    end

    subgraph Detail_Block ["Transformer Block Detail (Pre-LN)"]
        Input["Block Input (h_l)"] --> PreAttnNorm["Pre-Attention Normalization (LayerNorm / RMSNorm)"]
        Input --> Residual1[("+ Add Residual")]
        PreAttnNorm --> SelfAttn["Self-Attention (GQA + RoPE)"]
        SelfAttn --> Residual1
        
        Residual1 --> PreFFNNorm["Pre-FFN Normalization (LayerNorm / RMSNorm)"]
        Residual1 --> Residual2[("+ Add Residual")]
        PreFFNNorm --> FFN["Feed-Forward Network (FFN / GLU variants)"]
        FFN --> Residual2
        
        Residual2 --> Output["Block Output (h_l+1)"]
    end

    %% Apply Classes
    class Tokens,Logits,H0,BlockN,Input,Output flowStyle;
    class Embed,LMHead,SelfAttn,FFN blockStyle;
    class FinalNorm,PreAttnNorm,PreFFNNorm,Residual1,Residual2 opStyle;
```

---

## 🛠️ Key Architectural Implementations

### 1. Factorial Design (Two-Way ANOVA)
The statistical pipeline evaluates:
- **Factor A**: Normalization (LayerNorm, RMSNorm)
- **Factor B**: Activation (GELU, ReLU², SwiGLU)
- **Interaction (A × B)**: Calculates whether the choice of normalization modifies the effect of activation functions.
Outputs a publication-ready ANOVA table and pairwise t-tests with **Cohen's d** effect sizes inside `results/statistical_analysis.md`, alongside a formatted table `results/summary_table.md`.

### 2. Advanced Mechanistic Diagnostics
- **Gradient-Flow Metrics**: Logs parameter-wise gradient norm, Mean, Standard Deviation, RMS, Variance, and **Gradient Signal-to-Noise Ratio (GSNR)** ($GSNR = \frac{\mu^2}{\sigma^2}$).
- **Activation Saturation**: Computes activation Mean, Std, RMS, percentage of values **near zero** ($|x| < 0.01$), and **saturation percentage** ($x \le 0.0$ for ReLU², $x < -3.0$ for GELU/SiLU).
- **Hidden-State Cosine Similarity**: Tracks layer-wise representation cosine similarity $\cos(h_l, h_{l+1})$ at each block forward pass.
- **Optimizer Statistics**: Records mean update magnitude, update variance, and adaptive denominator statistics for custom AdamW updates.

### 3. Future-Proof Ablations
The model config supports dynamic ablation overrides:
- `rope: true/false` (Rotary Position Embeddings)
- `qk_norm: true/false` (Query-Key Normalization)
- `gqa: true/false` (Grouped-Query Attention; falls back to Multi-Head Attention when false)

### 4. Experiment Manifest & Env Capture
Every experiment run automatically collects environment details (GPU model, total VRAM, logical CPU cores, PyTorch commit, and FlashAttention capabilities) and saves an `experiment.json` manifest recording architectural specs, tokens seen, FLOPs, learning history, and runtime.

---

## ⚙️ Configuration Overrides (`configs/`)

Architectural parameters are defined inside individual config override files named as `[activation]_[normalization].yaml`. These override files are merged with the baseline settings in [base.yaml](file:///e:/transformer_research/configs/base.yaml):

* **`ln`** stands for **LayerNorm** combinations:
  - `gelu_ln.yaml`: GELU MLP + LayerNorm
  - `relu2_ln.yaml`: ReLU² MLP + LayerNorm
  - `swiglu_ln.yaml`: SwiGLU MLP + LayerNorm
* **`rms`** stands for **RMSNorm** combinations:
  - `gelu_rms.yaml`: GELU MLP + RMSNorm
  - `relu2_rms.yaml`: ReLU² MLP + RMSNorm
  - `swiglu_rms.yaml`: SwiGLU MLP + RMSNorm

---

## 🚀 Execution Guide

This framework is 100% self-contained. Copy this folder directly to Google Colab and execute:

### 1. Verification Tests
Always run verification tests first to verify mathematical correctness:
```bash
python -m pytest tests/
```

### 2. Running the Entire Grid (6 configs × 3 seeds)
Runs all registered combinations under seeds `42`, `43`, and `44`:
```bash
python transformer_research/experiments/run_all.py
```

### 3. Running a Specific Configuration
Filter by target normalization and activation functions using CLI flags:
```bash
# Run only SwiGLU + RMSNorm on seed 42
python transformer_research/experiments/run_all.py --activation swiglu --norm rmsnorm --seeds 42
```

### 4. Running a Quick Sweep (Local/Test)
Runs a fast experiment sweep using a tiny dataset size and lightweight model configuration:
```bash
python transformer_research/experiments/run_all.py --quick_sweep
```

---

## 📊 Automated Publication Figures (`results/plots/`)

The plotting suite automatically generates the following 9 charts required for the paper:
1. `validation_loss_curves.png` - Mean loss curves across seeds with standard deviation shading.
2. `perplexity_curves.png` - Validation perplexity curves.
3. `interaction_plot.png` - Normalization × Activation interaction lines.
4. `gradient_norm_curves.png` - Aggregate gradient norm over training steps.
5. `hidden_state_rms_by_layer.png` - Layer-wise hidden-state RMS scaling.
6. `attention_entropy_by_layer.png` - Layer-wise attention entropy profiles.
7. `training_throughput.png` - Bar chart comparing throughput (tokens/second).
8. `parameter_update_ratio.png` - Parameter update ratio over steps.
9. `effect_size_forest_plot.png` - Forest plot showing Cohen's d effect sizes with 95% Confidence Intervals.
