#!/usr/bin/env python3
"""Standalone PyTorch hidden-state calibration & 32-byte AVX2 binary exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from aux_calibrator.avx2_io import (
    MAGIC_DCOCO,
    MAGIC_HYPERTHINK,
    MAGIC_L0_POLE,
    write_avx2_bin,
)
from aux_calibrator.hooks import (
    extract_calibration_dataset,
    extract_pooled_band_states,
    resolve_transformer_layers,
)
from aux_calibrator.models import (
    AuxMLPHead,
    DCOCOLiteHead,
    HyperThinkHead,
    fit_contrastive_pole,
    fit_principal_pole,
)

DEFAULT_CALIB_TEXTS = [
    "Derive the closed-form solution for a damped harmonic oscillator and state the critical damping condition.",
    "Explain why multi-token prediction heads improve speculative acceptance rates during autoregressive decoding.",
    "Given a 28-layer transformer, compare residual stream cosine similarity between layer 14 and the final layer.",
    "Summarize the trade-offs between deterministic output verification and linear activation probing under distribution shift.",
    "Compute the effective rank of a matrix using the Renyi entropy of order 2 over its squared singular values.",
    "Write a concise checklist for validating 32-byte memory alignment when loading flat FP32 binary weights in C++.",
]


def load_texts(jsonl_path: str | None) -> List[str]:
    if not jsonl_path:
        return list(DEFAULT_CALIB_TEXTS)
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("text"):
                texts.append(str(obj["text"]))
    return texts


def load_contrastive_pairs(jsonl_path: str) -> Tuple[List[str], List[str]]:
    pos, neg = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("positive") and obj.get("negative"):
                pos.append(str(obj["positive"]))
                neg.append(str(obj["negative"]))
    return pos, neg


def train_head(
    head: AuxMLPHead,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    val_split: float = 0.15,
) -> Dict[str, float]:
    """Train an auxiliary MLP head with AdamW + MSE and return train/val MSE."""
    n = x.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, int(n * val_split))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx].to(device), y[val_idx].to(device)

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    final_train_mse = 0.0
    for _ in range(epochs):
        head.train()
        running, count = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(head(bx), by)
            loss.backward()
            opt.step()
            running += float(loss.item()) * bx.shape[0]
            count += bx.shape[0]
        final_train_mse = running / max(1, count)

    head.eval()
    with torch.inference_mode():
        val_mse = float(criterion(head(x_val), y_val).item())

    return {"train_mse": final_train_mse, "val_mse": val_mse, "samples": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate hidden-state heads and probes.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B", help="HF model ID or local path.")
    parser.add_argument("--calib-jsonl", default=None, help="JSONL with {'text': ...} rows.")
    parser.add_argument("--contrastive-jsonl", default=None, help="Optional JSONL with {'positive','negative'} pairs.")
    parser.add_argument("--output-dir", default="./artifacts", help="Directory for .bin exports.")
    parser.add_argument("--probe-layers", type=int, nargs="+", default=list(range(14, 23)))
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading locked checkpoint {args.model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.eval()

    hidden_dim = int(model.config.hidden_size)
    num_layers = len(resolve_transformer_layers(model))
    probe_layers = [i for i in args.probe_layers if 0 <= i < num_layers]
    texts = load_texts(args.calib_jsonl)

    print(f"Extracting hidden states ({len(texts)} sequences, D={hidden_dim}, probe_layers={probe_layers})...")
    batch = extract_calibration_dataset(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        probe_layers=probe_layers,
        max_length=args.max_length,
        device=device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "model_id": args.model_id,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "probe_layers": probe_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    # 1. HyperThink Halting Controller
    hyper_head = HyperThinkHead(hidden_dim=hidden_dim, rank=args.rank, out_dim=3)
    hyper_metrics = train_head(
        hyper_head, batch.hyper_x, batch.hyper_y,
        args.epochs, args.batch_size, args.lr, args.weight_decay, device,
    )
    hyper_meta = write_avx2_bin(
        out_dir / "hyperthink.bin", MAGIC_HYPERTHINK,
        hyper_head.export_tensors(), hidden_dim, args.rank, 3,
    )
    report["hyperthink"] = {**hyper_metrics, **hyper_meta}

    # 2. DCOCO-Lite Speculative Transition Scorer
    dcoco_head = DCOCOLiteHead(hidden_dim=hidden_dim, rank=args.rank, out_dim=3)
    dcoco_metrics = train_head(
        dcoco_head, batch.dcoco_x, batch.dcoco_y,
        args.epochs, args.batch_size, args.lr, args.weight_decay, device,
    )
    dcoco_meta = write_avx2_bin(
        out_dir / "dcoco_lite.bin", MAGIC_DCOCO,
        dcoco_head.export_tensors(), hidden_dim, args.rank, 3,
    )
    report["dcoco_lite"] = {**dcoco_metrics, **dcoco_meta}

    # 3. L0 Activation Pole (unsupervised principal direction)
    l0_pole = fit_principal_pole(batch.l0_states)
    l0_meta = write_avx2_bin(
        out_dir / "l0_activation_pole.bin", MAGIC_L0_POLE,
        [l0_pole], hidden_dim, 1, hidden_dim,
    )
    report["l0_activation_pole"] = {
        "l2_norm": float(torch.linalg.vector_norm(l0_pole).item()),
        **l0_meta,
    }

    # 4. Layer 14-22 Semantic Pole Probe (contrastive if pairs provided, else principal direction)
    if args.contrastive_jsonl:
        pos_texts, neg_texts = load_contrastive_pairs(args.contrastive_jsonl)
        pos_h = extract_pooled_band_states(model, tokenizer, pos_texts, probe_layers, args.max_length, device)
        neg_h = extract_pooled_band_states(model, tokenizer, neg_texts, probe_layers, args.max_length, device)
        sem_pole = fit_contrastive_pole(pos_h, neg_h)
        pole_mode = "contrastive_diff_of_means"
    else:
        sem_pole = fit_principal_pole(batch.probe_band_states)
        pole_mode = "principal_direction"

    sem_meta = write_avx2_bin(
        out_dir / "semantic_pole_l14_22.bin", MAGIC_L0_POLE,
        [sem_pole], hidden_dim, 1, hidden_dim,
    )
    report["semantic_pole_l14_22"] = {
        "mode": pole_mode,
        "l2_norm": float(torch.linalg.vector_norm(sem_pole).item()),
        **sem_meta,
    }

    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
