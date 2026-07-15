# Research Hypotheses

This document outlines the core scientific hypotheses to be tested by this empirical study.

---

## Hypothesis 1: Normalization Efficiency vs. Stability
- **Statement**: RMSNorm offers a computational speedup (throughput) over standard LayerNorm without degrading convergence rate, validation perplexity, or activation stability.
- **Diagnostics**:
  - Compare tokens/sec during training across all 6 configurations.
  - Measure pre-norm and post-norm hidden state RMS magnitudes across layers to verify that RMSNorm maintains a stable scaling factor identical to LayerNorm.

## Hypothesis 2: Feed-Forward Activation Gating
- **Statement**: Gated linear units (GLUs) such as SwiGLU outperform non-gated MLPs (GELU, ReLU²) because the gating mechanism allows the network to modulate information flow dynamically, leading to lower final validation perplexity.
- **Diagnostics**:
  - Compare final validation perplexity curves across GELU, ReLU², and SwiGLU.
  - Observe parameter count vs. perplexity trade-offs (SwiGLU uses more projection parameters for the same FFN dimension).

## Hypothesis 3: Attention Entropy under Normalizations
- **Statement**: Normalization layers directly affect attention entropy. RMSNorm maintains similar attention entropy distributions as LayerNorm, meaning that the lack of mean centering does not cause attention weights to saturate or disperse excessively.
- **Diagnostics**:
  - Record attention entropy per block. High entropy indicates diffuse attention; low entropy indicates highly focused attention.
