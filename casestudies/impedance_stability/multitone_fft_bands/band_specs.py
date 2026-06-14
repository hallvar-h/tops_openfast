"""Band definitions for split multitone FFT impedance ID."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BandId = Literal["lf", "mf", "hf"]
PlantId = Literal["wt", "fmu", "uic"]

# Shared simulation / injection
DT_S = 0.01
T_SETTLE_S_WT = 20.0
T_SETTLE_S_FMU = 50.0
T_SETTLE_S = T_SETTLE_S_WT  # default / WT (FMU uses T_SETTLE_S_FMU)
T_MARGIN_S = 5.0
TARGET_TOTAL_RMS = 0.05
RANDOM_SEED = 1
F1_HZ = 50.0
TOL = 1e-9

F_MIN_HZ = 0.05
F_MAX_HZ = 10.0

# Merge overlap (Hz)
LF_MF_BLEND_LO_HZ = 0.9
LF_MF_BLEND_HI_HZ = 1.0
LF_MF_SPLIT_HZ = 0.95
LF_MF_HI_OWN_HZ = 1.0

MF_HF_BLEND_LO_HZ = 3.5
MF_HF_BLEND_HI_HZ = 4.2
MF_HF_SPLIT_HZ = 3.85
MF_HF_HI_OWN_HZ = 4.2

BAND_ORDER: tuple[BandId, ...] = ("lf", "mf", "hf")


@dataclass(frozen=True)
class ToneSegment:
    """Piece of a multitone grid with uniform spacing (Hz)."""

    f_min_hz: float
    f_max_hz: float
    df_tone_hz: float


def _segment_frequency_list(
    segments: tuple[ToneSegment, ...], *, df_fft_hz: float
) -> tuple[float, ...]:
    """Build sorted unique tones on the FFT bin grid."""
    df_fft = float(df_fft_hz)
    parts: list[np.ndarray] = []
    for seg in segments:
        f0 = float(seg.f_min_hz)
        f1 = float(seg.f_max_hz)
        df = float(seg.df_tone_hz)
        if not (f1 > f0 and df > 0):
            raise ValueError(f"Invalid tone segment [{f0}, {f1}] df={df}")
        arr = np.arange(f0, f1, df, dtype=float)
        if arr.size == 0 or abs(float(arr[-1]) - f1) > 1e-9:
            arr = np.append(arr, f1)
        parts.append(arr)
    freqs = np.concatenate(parts)
    freqs = np.unique(np.round(freqs / df_fft) * df_fft)
    freqs = freqs[(freqs > 0.0) & np.isfinite(freqs)]
    return tuple(float(f) for f in np.sort(freqs))


# LF: fine 0.01 Hz below 0.5 Hz (incl. 0.17 Hz bracket), coarser 0.03 Hz above.
LF_TONE_SEGMENTS: tuple[ToneSegment, ...] = (
    ToneSegment(0.05, 0.5, 0.01),
    ToneSegment(0.53, 1.0, 0.03),
)
_LF_T_WIN_S = 500.0
_LF_DF_FFT_HZ = 1.0 / _LF_T_WIN_S
_LF_FREQS = _segment_frequency_list(LF_TONE_SEGMENTS, df_fft_hz=_LF_DF_FFT_HZ)


@dataclass(frozen=True)
class BandSpec:
    band: BandId
    f_min_hz: float
    f_max_hz: float
    n_tones: int
    df_tone_hz: float
    t_win_s: float
    tone_segments: tuple[ToneSegment, ...] | None = None

    @property
    def df_fft_hz(self) -> float:
        return 1.0 / float(self.t_win_s)

    @property
    def tone_frequencies_hz(self) -> tuple[float, ...]:
        if self.tone_segments is None:
            return ()
        return _segment_frequency_list(self.tone_segments, df_fft_hz=self.df_fft_hz)

    @property
    def t_end_s(self) -> float:
        """WT schedule (``t_settle`` 20 s); use :func:`t_end_s` for plant-specific end time."""
        return float(T_SETTLE_S_WT + self.t_win_s + T_MARGIN_S)

    @property
    def cycles_at_f_min(self) -> float:
        return float(self.f_min_hz * self.t_win_s)

    @property
    def amp_per_tone_pu(self) -> float:
        import math

        return float(TARGET_TOTAL_RMS * math.sqrt(2.0 / float(self.n_tones)))


_BANDS: dict[BandId, BandSpec] = {
    "lf": BandSpec(
        band="lf",
        f_min_hz=0.05,
        f_max_hz=1.0,
        # Piecewise grid: 0.05–0.5 Hz @ 0.01 Hz, 0.53–1.0 Hz @ 0.03 Hz (one LF sim).
        n_tones=len(_LF_FREQS),
        df_tone_hz=0.01,
        t_win_s=_LF_T_WIN_S,
        tone_segments=LF_TONE_SEGMENTS,
    ),
    "mf": BandSpec(
        band="mf",
        f_min_hz=0.9,
        f_max_hz=4.0,
        n_tones=24,
        df_tone_hz=0.10,
        t_win_s=200.0,
    ),
    "hf": BandSpec(
        band="hf",
        f_min_hz=3.5,
        f_max_hz=10.0,
        n_tones=22,
        df_tone_hz=0.15,
        t_win_s=120.0,
    ),
}


def t_settle_s(plant: PlantId | str) -> float:
    """Post-init transient discard before the FFT window (FMU needs longer than WT)."""
    key = str(plant).strip().lower()
    if key in ("wt", "uic"):
        return float(T_SETTLE_S_WT)
    if key == "fmu":
        return float(T_SETTLE_S_FMU)
    raise ValueError(f"plant must be 'wt', 'fmu', or 'uic', got {plant!r}")


def t_end_s(spec: BandSpec, plant: PlantId | str) -> float:
    """Total simulation time: settle + ``t_win`` + margin (plant-specific settle)."""
    return float(t_settle_s(plant) + spec.t_win_s + T_MARGIN_S)


def get_band_spec(band: str) -> BandSpec:
    key = str(band).strip().lower()
    if key not in _BANDS:
        raise KeyError(f"Unknown band {band!r}; expected one of {sorted(_BANDS)}")
    return _BANDS[key]  # type: ignore[return-value]
