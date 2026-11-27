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
```

* **Zero backbone drift:** All forward hooks execute under `torch.inference_mode()` with base weights frozen.
* **Sub-millisecond CPU inference:** Each auxiliary head contains 262,403 FP32 parameters (~1.05 MB), serialized with a 32-byte little-endian header (`<4sIIIIIII`, version `2`) and 32-byte padded tensor boundaries for direct AVX2 / C++ / Rust `mmap` loading.

---

## Verified Calibration Benchmarks

Reference calibration run against a locked 28-layer, `hidden_dim=2048` Qwen3-1.7B checkpoint (50 epochs, `AdamW`, `lr=2e-3`, `weight_decay=1e-4`, MSE loss):

| Artifact | Magic | Dimensions `(D, R, O)` | Targets / Direction | Verified Metric | Exact Binary Size |
|---|---|---|---|---|---|
| `hyperthink.bin` | `HYPT` | `(2048, 64, 3)` | `halt_prob`, `budget_scale`, `escalation_prob` | **0.000234 MSE** | `1,049,664` bytes |
| `dcoco_lite.bin` | `DCLC` | `(2048, 64, 3)` | `accept_prob`, `shift_magnitude`, `stability_measure` | **0.000534 MSE** | `1,049,664` bytes |
| `l0_activation_pole.bin` | `L0PL` | `(2048, 1, 2048)` | Unit-norm principal / semantic pole direction | **0.999981 L2 norm** | `8,224` bytes |

---

## Why Behavioral Auditing Precedes Activation Probes When Stylistic SFT Degrades CoT Monitorability

A central empirical takeaway from deploying these heads alongside post-trained reasoning models is **how to layer hidden-state probes relative to behavioral auditing**:

1. **Stylistic SFT & adapter merging degrade CoT monitorability.** When a base reasoning checkpoint undergoes stylistic SFT, brevity tuning, or multi-trajectory weight merging, surface chain-of-thought (`<think>`) traces often collapse, become performative, or omit intermediate calculations. Simultaneously, even mild LoRA rank updates rotate mid-to-upper residual subspaces (Layers 14–22), introducing calibration drift in fixed linear probes trained on the base checkpoint.
2. **Activation probes excel at continuous routing, not hard safety guarantees.** `HYPT` and `DCLC` reliably detect representational convergence (early halting), token-transition instability, and entropy spikes at near-zero compute cost—directly analogous to Multi-Token Prediction (MTP) auxiliary heads used for reasoning efficiency and speculative acceptance.
3. **Behavioral auditing must remain the primary hard gate.** For verifiable failure modes—exact numerical preservation, temporal/date invariants, tool-execution honesty (preventing phantom tool claims), and safety constraints—post-generation **deterministic behavioral auditing** must run ahead of (and strictly override) activation-probe heuristics. In production, we treat `L0PL` / `HYPT` / `DCLC` as **fast advisory telemetry and compute-allocation controllers**, while enforcing hard correctness and faithfulness invariants at the external behavioral/action boundary.

---

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Calibrate Heads & Probes Against a Locked Checkpoint

```bash
python scripts/calibrate.py \
  --model-id Qwen/Qwen3-1.7B \
  --output-dir ./artifacts \
  --probe-layers 14 15 16 17 18 19 20 21 22 \
  --rank 64 \
  --epochs 50 \
  --lr 2e-3 \
  --weight-decay 1e-4
```

This writes four 32-byte-aligned binaries plus a JSON summary into `./artifacts/`:
* `hyperthink.bin`
* `dcoco_lite.bin`
* `l0_activation_pole.bin`
* `semantic_pole_l14_22.bin`
* `calibration_report.json`

### 3. Verify Binary Alignment, Round-Trip Parity & CPU Latency

```bash
python scripts/evaluate.py \
  --artifact-dir ./artifacts \
  --hidden-dim 2048 \
  --rank 64
```

### 4. Custom Calibration & Contrastive Datasets

* **General calibration prompts:** Pass `--calib-jsonl path/to/prompts.jsonl` with one JSON object per line:
  ```json
  {"text": "Explain how residual stream cosine similarity behaves across transformer depth."}
  ```
* **Supervised contrastive semantic poles (Layers 14–22):** Pass `--contrastive-jsonl path/to/pairs.jsonl` to compute a unit-norm difference-of-means probe instead of unsupervised PCA:
  ```json
  {"positive": "Verify each arithmetic step before stating the final result.", "negative": "Guess the answer quickly without checking intermediate math."}
  ```

If neither flag is passed, `scripts/calibrate.py` runs against a built-in diagnostic calibration set for immediate smoke testing.

---

## Binary Format Specification (Version 2)

Every `.bin` artifact consists of a fixed **32-byte little-endian header** (`struct.pack("<4sIIIIIII", ...)`) followed by contiguous little-endian `float32` (`<f4`) tensors, each zero-padded to a 32-byte boundary for aligned AVX2 SIMD loads.

### Header Layout (32 Bytes)

| Offset | Field | Type | Description |
|---|---|---|---|
| `0..4` | `magic` | `char[4]` | `HYPT`, `DCLC`, or `L0PL` |
| `4..8` | `version` | `uint32` | Format version (`2`) |
| `8..12` | `hidden_dim` | `uint32` | Backbone hidden dimension `D` (e.g., `2048`) |
| `12..16` | `rank` | `uint32` | Bottleneck rank `R` (`64` for MLPs, `1` for poles) |
| `16..20` | `out_dim` | `uint32` | Output channels `O` (`3` for MLPs, `D` for poles) |
| `20..24` | `num_tensors` | `uint32` | Tensor count (`4` for 2-layer MLP, `1` for pole) |
| `24..32` | `reserved` | `uint32[2]` | Zero-padded reserved fields for 32-byte alignment |

### Payload Tensor Order & Byte-Size Parity (`D=2048, R=64, O=3`)

* **MLP Heads (`HYPT`, `DCLC`, `num_tensors=4`):**
  1. `fc1.weight` — shape `(64, 4096)` → `1,048,576` bytes (exact 32-byte multiple)
  2. `fc1.bias` — shape `(64,)` → `256` bytes (exact 32-byte multiple)
  3. `fc2.weight` — shape `(3, 64)` → `768` bytes (exact 32-byte multiple)
  4. `fc2.bias` — shape `(3,)` → `12` bytes + `20` zero-pad bytes = `32` bytes
  * **Total file size:** `32 + 1,048,576 + 256 + 768 + 32 = 1,049,664` bytes (`% 32 == 0`).
* **Activation / Semantic Poles (`L0PL`, `num_tensors=1`):**
  1. `pole` — shape `(2048,)` → `8,192` bytes (exact 32-byte multiple)
  * **Total file size:** `32 + 8,192 = 8,224` bytes (`% 32 == 0`).

---

## Repository Layout

```text
hidden-state-aux-calibrator/
├── README.md
├── pyproject.toml
├── aux_calibrator/
│   ├── __init__.py
│   ├── avx2_io.py        # 32-byte aligned FP32 binary export, loader & parity checks
│   ├── hooks.py          # Locked-checkpoint forward-hook activation extractor (L0, L14–22, Final)
│   └── models.py         # HyperThink, DCOCO-Lite MLP heads & unit-norm semantic pole probes
└── scripts/
    ├── calibrate.py      # Standalone forward-hook extraction + head/probe training + export CLI
    └── evaluate.py       # Binary alignment, round-trip parity, latency & probe norm evaluator
```

## License

MIT
