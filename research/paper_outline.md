# Paper Outline: Norms and Activations in Transformers

## Title Proposal
*An Empirical Study on Normalization and Activation Interactions in Autoregressive Decoder-Only Transformers*

## Abstract
- Briefly summarize the goals of the study.
- State the configurations evaluated (GELU vs. ReLU² vs. SwiGLU combined with LayerNorm vs. RMSNorm).
- Summarize main findings: which combination converges fastest, stability properties (activation RMS, gradient updates, attention entropy), and training throughput trade-offs.

## 1. Introduction
- Background on Transformers and the shift towards RMSNorm and SwiGLU in state-of-the-art LLMs (Llama, Mistral, Qwen).
- Objective: Isolate these architectural changes under a controlled setting to analyze their interaction.

## 2. Methodology
- **Architectural Baseline**: Mini-Qwen architecture (GQA, RoPE, custom AdamW).
- **Normalization Variants**:
  - Custom LayerNorm (mean centering + variance scaling).
  - Custom RMSNorm (variance scaling only).
- **Activation Variants**:
  - GELU MLP.
  - ReLU² MLP ($ReLU(x)^2$).
  - SwiGLU MLP.
- **Controlled Hyperparameters**: Optimizer settings, learning rate schedule, context length, dataset.

## 3. Experimental Setup
- Dataset: TinyStories, WikiText, or FineWeb.
- Training parameters (optimizer, scheduler, sequence length).
- Hardware and precision configuration.

## 4. Results
- **Convergence Curves**: Loss and validation perplexity comparisons.
- **Computational Cost**: Throughput (tokens/sec), parameter counts, peak GPU memory.
- **Statistical Significance**: ANOVA, pairwise t-tests, Cohen's d effect sizes.

## 5. Mechanistic Diagnostics (Discussion)
- **Activation Scale Stabilization**: Analyze hidden-state RMS before/after norms to show if RMSNorm stabilizes hidden states as effectively as LayerNorm.
- **Gradient Update Dynamics**: Show layer-wise gradient norms and parameter update ratios ($\|\Delta W\| / \|W\|$).
- **Attention Focus**: Attention entropy trends across layers.

## 6. Conclusion
- Synthesis of findings.
- Actionable recommendations for future model designs.
