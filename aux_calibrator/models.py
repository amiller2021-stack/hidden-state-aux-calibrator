"""Auxiliary MLP heads (HyperThink, DCOCO-Lite) and unit-norm semantic pole probes."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxMLPHead(nn.Module):
    """Dual-linear bottleneck head: Linear(2*D -> R) -> SiLU -> Linear(R -> 3) -> Sigmoid."""

    def __init__(self, hidden_dim: int, rank: int = 64, out_dim: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.out_dim = out_dim
        self.fc1 = nn.Linear(2 * hidden_dim, rank, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(rank, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))

    def export_tensors(self) -> List[torch.Tensor]:
        """Return weights in canonical V2 serialization order."""
        return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def load_exported_tensors(self, tensors: List[torch.Tensor]) -> None:
        if len(tensors) != 4:
            raise ValueError(f"Expected 4 MLP tensors, received {len(tensors)}.")
        with torch.no_grad():
            self.fc1.weight.copy_(tensors[0])
            self.fc1.bias.copy_(tensors[1])
            self.fc2.weight.copy_(tensors[2])
            self.fc2.bias.copy_(tensors[3])


class DCOCOLiteHead(AuxMLPHead):
    """Speculative transition scorer predicting (accept_prob, shift_mag, stability)."""

    CHANNEL_NAMES: Tuple[str, str, str] = (
        "accept_prob",
        "shift_magnitude",
        "stability_measure",
    )


class HyperThinkHead(AuxMLPHead):
    """Halting & compute-budget controller predicting (halt_prob, budget_scale, escalation_prob)."""

    CHANNEL_NAMES: Tuple[str, str, str] = (
        "halt_prob",
        "budget_scale",
        "escalation_prob",
    )


def fit_principal_pole(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the unit-norm leading principal direction of centered activations [N, D]."""
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError(f"Expected [N>=2, D] activations, got {tuple(activations.shape)}.")
    x = activations.to(dtype=torch.float32)
    centered = x - x.mean(dim=0, keepdim=True)
    # Economy SVD on CPU/GPU; top right singular vector gives principal axis
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    pole = vh[0]
    return F.normalize(pole, p=2, dim=0, eps=eps)


def fit_contrastive_pole(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute unit-norm difference-of-means semantic probe direction: norm(mu+ - mu-)."""
    mu_pos = pos_activations.to(dtype=torch.float32).mean(dim=0)
    mu_neg = neg_activations.to(dtype=torch.float32).mean(dim=0)
    return F.normalize(mu_pos - mu_neg, p=2, dim=0, eps=eps)
