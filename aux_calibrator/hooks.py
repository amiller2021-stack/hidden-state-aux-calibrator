"""Non-invasive forward-hook hidden-state extraction for locked causal LMs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationBatch:
    dcoco_x: torch.Tensor        # [N_trans, 2 * D] = [h_{t-1}^{final}, h_t^{final}]
    dcoco_y: torch.Tensor        # [N_trans, 3]     = (accept_prob, shift_mag, stability)
    hyper_x: torch.Tensor        # [N_tok, 2 * D]   = [h_t^{mid}, h_t^{final}]
    hyper_y: torch.Tensor        # [N_tok, 3]       = (halt_prob, budget_scale, escalation_prob)
    l0_states: torch.Tensor      # [N_tok, D]       = Layer-0 activations
    probe_band_states: torch.Tensor  # [N_tok, D]   = Mean-pooled Layers 14..22 activations


def resolve_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate transformer decoder blocks across Qwen/Llama/Mistral/Gemma/GPT architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError(
        "Could not locate decoder layer ModuleList (checked model.layers, transformer.h, gpt_neox.layers)."
    )


class HiddenStateHookCollector:
    """Registers PyTorch forward hooks on selected transformer layers and cleans up safely."""

    def __init__(self, model: nn.Module, layer_indices: Sequence[int]) -> None:
        self.layers = resolve_transformer_layers(model)
        self.num_layers = len(self.layers)
        self.layer_indices = sorted(set(layer_indices))
        for idx in self.layer_indices:
            if idx < 0 or idx >= self.num_layers:
                raise IndexError(
                    f"Layer index {idx} out of bounds for backbone with {self.num_layers} layers."
                )
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def _hook(_module: nn.Module, _inputs: Tuple[object, ...], output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer_idx] = hidden.detach().to(device="cpu", dtype=torch.float32)

        return _hook

    def __enter__(self) -> "HiddenStateHookCollector":
        for idx in self.layer_indices:
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()

    def pop_captured(self) -> Dict[int, torch.Tensor]:
        out = dict(self._captured)
        self._captured.clear()
        return out


@torch.inference_mode()
def extract_calibration_dataset(
    model: nn.Module,
    tokenizer: object,
    texts: Iterable[str],
    probe_layers: Sequence[int] | None = None,
    max_length: int = 256,
    device: str | torch.device = "cpu",
) -> CalibrationBatch:
    """Run locked-checkpoint forward passes to build training tensors for all 3 artifacts."""
    layers = resolve_transformer_layers(model)
    num_layers = len(layers)
    mid_idx = num_layers // 2
    final_idx = num_layers - 1

    if probe_layers is None:
        # Default to Layers 14..22 on 28-layer backbones (scaled proportionally otherwise)
        lo = min(14, mid_idx)
        hi = min(22, final_idx)
        probe_band = list(range(lo, hi + 1))
    else:
        probe_band = [i for i in probe_layers if 0 <= i < num_layers]
        if not probe_band:
            probe_band = [mid_idx]

    hook_indices = sorted({0, mid_idx, final_idx, *probe_band})

    dcoco_x_list, dcoco_y_list = [], []
    hyper_x_list, hyper_y_list = [], []
    l0_list, band_list = [], []

    model.eval()
    with HiddenStateHookCollector(model, hook_indices) as collector:
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 4:
                continue

            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0].detach().to(device="cpu", dtype=torch.float32)  # [T, V]
            captured = collector.pop_captured()

            h0 = captured[0][0]           # [T, D]
            h_mid = captured[mid_idx][0]  # [T, D]
            h_fin = captured[final_idx][0]  # [T, D]
            h_band = torch.stack([captured[i][0] for i in probe_band], dim=0).mean(dim=0)

            # Next-token predictive statistics from frozen logits (aligned on positions 0..T-1)
            probs = F.softmax(logits, dim=-1)
            top1_prob = probs.max(dim=-1).values.clamp(0.0, 1.0)
            vocab_size = logits.shape[-1]
            norm_entropy = (
                -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1) / math.log(vocab_size)
            ).clamp(0.0, 1.0)

            # 1. DCOCO-Lite transition pairs: [h_{t-1}^{final}, h_t^{final}]
            h_prev = h_fin[:-1]
            h_curr = h_fin[1:]
            cos_step = F.cosine_similarity(h_prev, h_curr, dim=-1).clamp(-1.0, 1.0)
            stability = 0.5 * (cos_step + 1.0)
            shift_mag = (
                torch.linalg.vector_norm(h_curr - h_prev, dim=-1)
                / (torch.linalg.vector_norm(h_curr, dim=-1) + torch.linalg.vector_norm(h_prev, dim=-1) + 1e-6)
            ).clamp(0.0, 1.0)
            accept_prob = top1_prob[1:]

            dcoco_x_list.append(torch.cat([h_prev, h_curr], dim=-1))
            dcoco_y_list.append(torch.stack([accept_prob, shift_mag, stability], dim=-1))

            # 2. HyperThink cross-layer pairs: [h_t^{mid}, h_t^{final}]
            cos_depth = F.cosine_similarity(h_mid, h_fin, dim=-1).clamp(-1.0, 1.0)
            halt_prob = 0.5 * (cos_depth + 1.0)
            budget_scale = torch.tanh(
                torch.linalg.vector_norm(h_fin - h_mid, dim=-1)
                / (torch.linalg.vector_norm(h_mid, dim=-1) + 1e-6)
            ).clamp(0.0, 1.0)
            escalation_prob = norm_entropy

            hyper_x_list.append(torch.cat([h_mid, h_fin], dim=-1))
            hyper_y_list.append(torch.stack([halt_prob, budget_scale, escalation_prob], dim=-1))

            # 3. Pole states
            l0_list.append(h0)
            band_list.append(h_band)

    if not hyper_x_list:
        raise RuntimeError("No valid calibration sequences (>= 4 tokens) were processed.")

    return CalibrationBatch(
        dcoco_x=torch.cat(dcoco_x_list, dim=0),
        dcoco_y=torch.cat(dcoco_y_list, dim=0),
        hyper_x=torch.cat(hyper_x_list, dim=0),
        hyper_y=torch.cat(hyper_y_list, dim=0),
        l0_states=torch.cat(l0_list, dim=0),
        probe_band_states=torch.cat(band_list, dim=0),
    )


@torch.inference_mode()
def extract_pooled_band_states(
    model: nn.Module,
    tokenizer: object,
    texts: Iterable[str],
    probe_layers: Sequence[int],
    max_length: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Extract sequence-mean hidden activations across probe_layers (e.g., Layers 14..22)."""
    model.eval()
    pooled: List[torch.Tensor] = []
    with HiddenStateHookCollector(model, probe_layers) as collector:
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            _ = model(input_ids=input_ids, use_cache=False)
            captured = collector.pop_captured()
            layer_mean = torch.stack([captured[i][0] for i in probe_layers], dim=0).mean(dim=0)
            pooled.append(layer_mean.mean(dim=0))
    return torch.stack(pooled, dim=0)
