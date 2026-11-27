# hidden-state-aux-calibrator

Standalone PyTorch calibration, probing, and 32-byte AVX2 binary export toolkit for training lightweight auxiliary heads and semantic activation probes against **locked causal LM checkpoints**.

Designed for reasoning-efficiency routing, speculative transition scoring, and internal state telemetry without modifying base model weights:

1. **HyperThink (`HYPT`)** — Cross-layer halting & compute-budget controller operating on `[h_mid, h_final]` (`L // 2` vs. final layer) to predict `(halt_prob, budget_scale, escalation_prob)`.
2. **DCOCO-Lite (`DCLC`)** — Speculative transition scorer operating on consecutive token hidden states `[h_{t-1}, h_t]` to predict `(accept_prob, shift_magnitude, stability_measure)`.
3. **Activation & Layer 14–22 Semantic Pole Probes (`L0PL`)** — Unit-norm directional probes extracted from post-Layer-0 and mid-to-upper semantic layers (Layers 14–22 on 28-layer backbones) via unsupervised PCA or contrastive difference-of-means.

---

## Architecture & Signal Flow

```text
Locked Causal LM Backbone (e.g., Qwen3-1.7B, hidden_dim=2048, num_layers=28)
  │
  ├── Forward Hook @ Layer 0         ──► L0 Activation Pole (unit-norm, dim=2048) ──► L0PL (8,224 B)
  ├── Forward Hook @ Layers 14..22   ──► Semantic Pole Probe (unit-norm, dim=2048) ──► L0PL (8,224 B)
  ├── Forward Hook @ Mid (L // 2)  ──┐
  │                                  ├──► concat [h_mid, h_final] (4096) ──► Linear(4096→64)→SiLU→Linear(64→3)→Sigmoid ──► HYPT (1,049,664 B)
  └── Forward Hook @ Final (L - 1) ──┤
                                     └──► concat [h_{t-1}, h_t]  (4096) ──► Linear(4096→64)→SiLU→Linear(64→3)→Sigmoid ──► DCLC (1,049,664 B)
