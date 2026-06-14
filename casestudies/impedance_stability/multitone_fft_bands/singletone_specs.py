"""Single-tone cross-check specs (same settle/margin as band-split multitone)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    TARGET_TOTAL_RMS,
    T_MARGIN_S,
    get_band_spec,
    t_settle_s,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import PlantId

_MIN_CYCLES = 30.0


def freq_band_tag(f_hz: float) -> str:
    """Filesystem-safe tag, e.g. 0.17 Hz -> ``f0p17``."""
    s = f"{float(f_hz):.6f}".rstrip("0").rstrip(".")
    return "f" + s.replace(".", "p")


def _parse_freq_tag(tag: str) -> float:
    m = re.fullmatch(r"f([0-9p]+)", str(tag).strip().lower())
    if not m:
        raise ValueError(f"Invalid single-tone band tag {tag!r}")
    return float(m.group(1).replace("p", "."))


def _parent_band_for_freq(f_hz: float) -> str:
    f = float(f_hz)
    if f <= 1.0:
        return "lf"
    if f <= 4.0:
        return "mf"
    return "hf"


def snap_frequency_to_fft_bin(f_hz: float, *, t_win_s: float) -> float:
    df = 1.0 / float(t_win_s)
    return float(round(float(f_hz) / df) * df)


@dataclass(frozen=True)
class SingleToneSpec:
    """One injected tone; ``band_tag`` names log/CSV files (e.g. ``f0p17``)."""

    f_hz: float
    t_win_s: float
    amp_per_tone_pu: float
    band_tag: str

    @property
    def df_fft_hz(self) -> float:
        return 1.0 / float(self.t_win_s)

    @property
    def f_min_hz(self) -> float:
        return float(self.f_hz)

    @property
    def f_max_hz(self) -> float:
        return float(self.f_hz)

    @property
    def n_tones(self) -> int:
        return 1

    @property
    def df_tone_hz(self) -> float:
        return float(self.df_fft_hz)

    @property
    def tone_segments(self) -> None:
        return None

    @property
    def band(self) -> str:
        return str(self.band_tag)

    @property
    def cycles_at_f_min(self) -> float:
        return float(self.f_hz * self.t_win_s)

    def t_end_s(self, plant: PlantId | str) -> float:
        return float(t_settle_s(plant) + self.t_win_s + T_MARGIN_S)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.f_hz <= 0.0:
            errors.append(f"{self.band_tag}: f_hz must be > 0")
        if self.cycles_at_f_min < _MIN_CYCLES:
            errors.append(
                f"{self.band_tag}: f*T_win={self.cycles_at_f_min:.2g} < {_MIN_CYCLES} cycles"
            )
        if self.amp_per_tone_pu <= 0.0:
            errors.append(f"{self.band_tag}: amp must be > 0")
        return errors


def singletone_spec_for_frequency(
    f_hz: float,
    *,
    amp_per_tone_pu: float | None = None,
    t_win_s: float | None = None,
) -> SingleToneSpec:
    """
    Build a single-tone plan using the parent band's ``T_win`` (LF/MF/HF).

    Default amplitude: ``target_total_rms * sqrt(2)`` (one tone carries full RMS budget).
    Frequency is snapped to the FFT bin grid for that window.
    """
    parent = _parent_band_for_freq(f_hz)
    tw = float(t_win_s) if t_win_s is not None else float(get_band_spec(parent).t_win_s)
    f_snap = snap_frequency_to_fft_bin(f_hz, t_win_s=tw)
    amp = (
        float(amp_per_tone_pu)
        if amp_per_tone_pu is not None
        else float(TARGET_TOTAL_RMS * math.sqrt(2.0))
    )
    return SingleToneSpec(
        f_hz=f_snap,
        t_win_s=tw,
        amp_per_tone_pu=amp,
        band_tag=freq_band_tag(f_snap),
    )
