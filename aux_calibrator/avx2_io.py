"""32-byte AVX2-aligned little-endian FP32 binary export and verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ALIGNMENT_BYTES = 32
HEADER_FORMAT = "<4sIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 32 bytes
FORMAT_VERSION = 2

MAGIC_DCOCO = b"DCLC"
MAGIC_HYPERTHINK = b"HYPT"
MAGIC_L0_POLE = b"L0PL"
VALID_MAGICS = {MAGIC_DCOCO, MAGIC_HYPERTHINK, MAGIC_L0_POLE}


@dataclass(frozen=True)
class BinaryHeader:
    magic: bytes
    version: int
    hidden_dim: int
    rank: int
    out_dim: int
    num_tensors: int


def _aligned_size(n_bytes: int, alignment: int = ALIGNMENT_BYTES) -> int:
    return ((n_bytes + alignment - 1) // alignment) * alignment


def expected_tensor_shapes(header: BinaryHeader) -> List[Tuple[int, ...]]:
    """Return canonical tensor shapes implied by a V2 binary header."""
    if header.magic in (MAGIC_DCOCO, MAGIC_HYPERTHINK):
        in_dim = 2 * header.hidden_dim
        return [
            (header.rank, in_dim),   # fc1.weight
            (header.rank,),          # fc1.bias
            (header.out_dim, header.rank),  # fc2.weight
            (header.out_dim,),       # fc2.bias
        ]
    if header.magic == MAGIC_L0_POLE:
        return [(header.hidden_dim,)]
    raise ValueError(f"Unsupported magic header: {header.magic!r}")


def expected_file_bytes(
    magic: bytes, hidden_dim: int, rank: int, out_dim: int
) -> int:
    """Compute exact 32-byte-aligned serialized size in bytes."""
    hdr = BinaryHeader(magic, FORMAT_VERSION, hidden_dim, rank, out_dim, 0)
    total = HEADER_SIZE
    for shape in expected_tensor_shapes(hdr):
        numel = 1
        for dim in shape:
            numel *= dim
        total += _aligned_size(numel * 4)
    return total


def write_avx2_bin(
    path: str | Path,
    magic_4b: bytes,
    tensors: Sequence[torch.Tensor],
    hidden_d: int,
    rank_d: int,
    out_d: int,
) -> Dict[str, object]:
    """Serialize FP32 tensors with a 32-byte header and 32-byte AVX2 alignment.

    For standard dimensions (e.g., hidden_d=2048, rank_d=64, out_d=3), fc1.weight,
    fc1.bias, and fc2.weight are already exact multiples of 32 bytes, so per-tensor
    and end-of-payload alignment yield identical byte layouts (1,049,664 bytes for
    MLP heads and 8,224 bytes for 2048-dim poles).
    """
    if len(magic_4b) != 4 or magic_4b not in VALID_MAGICS:
        raise ValueError(f"Invalid 4-byte magic: {magic_4b!r}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_bytes = struct.pack(
        HEADER_FORMAT,
        magic_4b,
        FORMAT_VERSION,
        int(hidden_d),
        int(rank_d),
        int(out_d),
        len(tensors),
        0,
        0,
    )
    assert len(header_bytes) == HEADER_SIZE

    hasher = hashlib.sha256()
    total_written = 0

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        hasher.update(header_bytes)
        total_written += len(header_bytes)

        for tensor in tensors:
            arr = (
                tensor.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .astype("<f4", copy=False)
            )
            raw = arr.tobytes(order="C")
            pad_len = (_aligned_size(len(raw)) - len(raw))
            payload = raw + (b"\x00" * pad_len)
            f.write(payload)
            hasher.update(payload)
            total_written += len(payload)

    return {
        "path": str(out_path),
        "magic": magic_4b.decode("ascii"),
        "bytes": total_written,
        "aligned_32b": (total_written % ALIGNMENT_BYTES) == 0,
        "sha256": hasher.hexdigest(),
    }


def read_avx2_bin(path: str | Path) -> Tuple[BinaryHeader, List[torch.Tensor]]:
    """Load and unpack a V2 32-byte aligned binary artifact back into FP32 tensors."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"File too small ({len(raw)} bytes) for 32-byte header.")

    magic, version, hidden_d, rank_d, out_d, n_tensors, r0, r1 = struct.unpack(
        HEADER_FORMAT, raw[:HEADER_SIZE]
    )
    if magic not in VALID_MAGICS:
        raise ValueError(f"Unknown magic bytes: {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported binary version {version}; expected {FORMAT_VERSION}.")
    if r0 != 0 or r1 != 0:
        raise ValueError(f"Non-zero reserved header fields: ({r0}, {r1}).")

    header = BinaryHeader(magic, version, hidden_d, rank_d, out_d, n_tensors)
    shapes = expected_tensor_shapes(header)
    if len(shapes) != n_tensors:
        raise ValueError(
            f"Header declares {n_tensors} tensors, expected {len(shapes)} for {magic!r}."
        )

    offset = HEADER_SIZE
    tensors: List[torch.Tensor] = []
    for shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        n_bytes = numel * 4
        aligned = _aligned_size(n_bytes)
        if offset + aligned > len(raw):
            raise ValueError("Truncated binary payload while unpacking tensors.")
        arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=offset).copy()
        tensors.append(torch.from_numpy(arr).reshape(shape))
        offset += aligned

    if offset != len(raw):
        raise ValueError(f"Unexpected trailing bytes: read {offset}, file size {len(raw)}.")

    return header, tensors


def verify_avx2_bin(
    path: str | Path,
    expected_magic: bytes | None = None,
    expected_hidden_dim: int | None = None,
) -> Dict[str, object]:
    """Validate header, byte alignment, NaN/Inf safety, and SHA-256 digest."""
    p = Path(path)
    raw = p.read_bytes()
    header, tensors = read_avx2_bin(p)

    if expected_magic is not None and header.magic != expected_magic:
        raise ValueError(f"Magic mismatch: got {header.magic!r}, expected {expected_magic!r}")
    if expected_hidden_dim is not None and header.hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"hidden_dim mismatch: got {header.hidden_dim}, expected {expected_hidden_dim}"
        )

    exp_bytes = expected_file_bytes(
        header.magic, header.hidden_dim, header.rank, header.out_dim
    )
    if len(raw) != exp_bytes:
        raise ValueError(f"Byte-size parity failed: got {len(raw)}, expected {exp_bytes}")

    finite = all(bool(torch.isfinite(t).all().item()) for t in tensors)
    if not finite:
        raise ValueError("Non-finite (NaN/Inf) values detected in serialized tensors.")

    result: Dict[str, object] = {
        "path": str(p),
        "magic": header.magic.decode("ascii"),
        "version": header.version,
        "dims": (header.hidden_dim, header.rank, header.out_dim),
        "num_tensors": header.num_tensors,
        "bytes": len(raw),
        "aligned_32b": (len(raw) % ALIGNMENT_BYTES) == 0,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if header.magic == MAGIC_L0_POLE:
        result["l2_norm"] = float(torch.linalg.vector_norm(tensors[0]).item())
    return result
