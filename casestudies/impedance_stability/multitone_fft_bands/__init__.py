"""Band-split multitone FFT impedance identification (0.1–10 Hz)."""

from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    BAND_ORDER,
    BandSpec,
    DT_S,
    F1_HZ,
    F_MAX_HZ,
    F_MIN_HZ,
    T_MARGIN_S,
    T_SETTLE_S,
    T_SETTLE_S_FMU,
    T_SETTLE_S_WT,
    TARGET_TOTAL_RMS,
    TOL,
    get_band_spec,
    t_end_s,
    t_settle_s,
)

__all__ = [
    "BAND_ORDER",
    "BandSpec",
    "DT_S",
    "F1_HZ",
    "F_MAX_HZ",
    "F_MIN_HZ",
    "T_MARGIN_S",
    "T_SETTLE_S",
    "T_SETTLE_S_FMU",
    "T_SETTLE_S_WT",
    "TARGET_TOTAL_RMS",
    "TOL",
    "get_band_spec",
    "t_end_s",
    "t_settle_s",
]
