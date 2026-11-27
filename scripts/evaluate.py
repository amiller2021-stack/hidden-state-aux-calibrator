#!/usr/bin/env python3
"""Verify binary alignment, round-trip parity, latency, and norm/range invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict

import torch

from aux_calibrator.avx2_io import (
    MAGIC_DCOCO,
    MAGIC_HYPERTHINK,
    MAGIC_L0_POLE,
    read_avx2_bin,
    verify_avx2_bin,
)
from aux_calibrator.models import DCOCOLiteHead, HyperThinkHead


def benchmark_head(head: torch.nn.Module, hidden_dim: int, iters: int = 200) -> Dict[str, float]:
    """Measure CPU inference latency and validate sigmoid output bounds [0, 1]."""
    x = torch.randn(1, 2 * hidden_dim, dtype=torch.float32)
    head.eval()
    with torch.inference_mode():
        # Warmup
        for _ in range(10):
            out = head(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = head(x)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    min_v, max_v = float(out.min().item()), float(out.max().item())
    if min_v < 0.0 or max_v > 1.0:
        raise ValueError(f"Output outside [0, 1] sigmoid range: [{min_v}, {max_v}]")
    return {"mean_latency_ms": round(elapsed_ms, 4), "sample_out": [round(v, 5) for v in out[0].tolist()]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported AVX2 calibration binaries.")
    parser.add_argument("--artifact-dir", default="./artifacts")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()

    art_dir = Path(args.artifact_dir)
    summary: Dict[str, object] = {}

    specs = [
        ("hyperthink.bin", MAGIC_HYPERTHINK, HyperThinkHead),
        ("dcoco_lite.bin", MAGIC_DCOCO, DCOCOLiteHead),
        ("l0_activation_pole.bin", MAGIC_L0_POLE, None),
        ("semantic_pole_l14_22.bin", MAGIC_L0_POLE, None),
    ]

    for fname, magic, cls in specs:
        fpath = art_dir / fname
        if not fpath.is_file():
            continue

        info = verify_avx2_bin(fpath, expected_magic=magic, expected_hidden_dim=args.hidden_dim)
        header, tensors = read_avx2_bin(fpath)

        if cls is not None:
            if header.rank != args.rank:
                raise ValueError(f"{fname}: rank {header.rank} != expected {args.rank}")
            head = cls(hidden_dim=header.hidden_dim, rank=header.rank, out_dim=header.out_dim)
            head.load_exported_tensors(tensors)
            info["benchmark"] = benchmark_head(head, header.hidden_dim)
        else:
            norm = float(torch.linalg.vector_norm(tensors[0]).item())
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(f"{fname}: pole L2 norm {norm:.6f} deviates from unit norm 1.0")
            info["unit_norm_ok"] = True

        summary[fname] = info

    if not summary:
        raise SystemExit(f"No recognized .bin artifacts found in {art_dir}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
