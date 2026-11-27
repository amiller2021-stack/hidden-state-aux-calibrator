"""Hidden-state auxiliary head calibration and AVX2 binary export utilities."""

from aux_calibrator.avx2_io import (
    ALIGNMENT_BYTES,
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC_DCOCO,
    MAGIC_HYPERTHINK,
    MAGIC_L0_POLE,
    read_avx2_bin,
    verify_avx2_bin,
    write_avx2_bin,
)
from aux_calibrator.models import (
    AuxMLPHead,
    DCOCOLiteHead,
    HyperThinkHead,
    fit_contrastive_pole,
    fit_principal_pole,
)

__all__ = [
    "ALIGNMENT_BYTES",
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "MAGIC_DCOCO",
    "MAGIC_HYPERTHINK",
    "MAGIC_L0_POLE",
    "AuxMLPHead",
    "DCOCOLiteHead",
    "HyperThinkHead",
    "fit_contrastive_pole",
    "fit_principal_pole",
    "read_avx2_bin",
    "verify_avx2_bin",
    "write_avx2_bin",
]
