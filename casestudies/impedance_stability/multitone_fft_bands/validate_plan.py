"""
Pre-flight validation for band-split multitone parameters.

Run from repo root:
  python casestudies/impedance_stability/multitone_fft_bands/validate_plan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from casestudies.impedance_stability.multitone_fft_bands.collision_check import (
    check_msd_multitone_list,
    propose_tone_grid,
)
from casestudies.impedance_stability.ps_data.multitone_fft_bands import (
    get_band_injected_frequencies,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    BAND_ORDER,
    F1_HZ,
    TARGET_TOTAL_RMS,
    TOL,
    BandSpec,
    get_band_spec,
    t_end_s,
)

MIN_CYCLES = 30.0
# LF extends to 0.05 Hz; 500 s window → 25 cycles at f_min (still ≫ cycles at 0.17 Hz).
MIN_CYCLES_LF = 25.0
MAX_T_END_S = 600.0
MAX_AMP_PU = 0.03
MAX_AMP_WARN_PU = 0.025


def _validate_band(spec: BandSpec) -> list[str]:
    errors: list[str] = []
    if spec.tone_segments is None:
        grid = propose_tone_grid(
            f_min_hz=spec.f_min_hz,
            f_max_hz=spec.f_max_hz,
            n_tones=spec.n_tones,
            df_fft_hz=spec.df_fft_hz,
            df_tone_hz=spec.df_tone_hz,
        )
        freqs = grid.freqs_hz
        if freqs.size != spec.n_tones:
            errors.append(f"{spec.band}: tone count {freqs.size} != n_tones {spec.n_tones}")
    else:
        freqs = get_band_injected_frequencies(spec.band)
        if freqs.size != spec.n_tones:
            errors.append(f"{spec.band}: tone count {freqs.size} != n_tones {spec.n_tones}")
        for seg in spec.tone_segments:
            in_seg = (freqs >= seg.f_min_hz - 1e-9) & (freqs <= seg.f_max_hz + 1e-9)
            seg_f = freqs[in_seg]
            d_seg = np.diff(seg_f)
            # Last interval may be < df_tone to hit f_max on the FFT grid (e.g. 0.98 -> 1.0 Hz).
            d_check = d_seg[:-1] if d_seg.size > 1 else d_seg
            if d_check.size and float(np.min(d_check)) < seg.df_tone_hz - 1e-9:
                errors.append(
                    f"{spec.band}: spacing in [{seg.f_min_hz},{seg.f_max_hz}] "
                    f"< {seg.df_tone_hz:g} Hz"
                )
    min_cycles = MIN_CYCLES_LF if spec.band == "lf" else MIN_CYCLES
    if spec.cycles_at_f_min < min_cycles:
        errors.append(
            f"{spec.band}: f_min*T_win={spec.cycles_at_f_min:.2g} < {min_cycles} cycles"
        )
    for plant in ("wt", "fmu"):
        t_end = t_end_s(spec, plant)
        if t_end > MAX_T_END_S:
            errors.append(f"{spec.band} {plant}: t_end={t_end:g}s > {MAX_T_END_S:g}s")
    amp = spec.amp_per_tone_pu
    if amp > MAX_AMP_PU:
        errors.append(f"{spec.band}: amp={amp:.4g} pu > {MAX_AMP_PU} pu")
    elif amp > MAX_AMP_WARN_PU:
        print(f"  [{spec.band}] warn: per-tone amp {amp:.4g} pu > {MAX_AMP_WARN_PU} pu", flush=True)

    # MSD check on the exact tones that will be injected (not propose_tone_grid only).
    inj = get_band_injected_frequencies(spec.band)
    coll = check_msd_multitone_list(inj, f1=F1_HZ, tol=TOL)
    if coll["same_mirror"] or coll["cross_coll"] or coll["special"]:
        errors.append(f"{spec.band}: MSD collisions {coll}")

    d = np.diff(inj)
    min_df = float(spec.df_tone_hz)
    if d.size and float(np.min(d)) < min_df - 1e-9:
        errors.append(f"{spec.band}: min tone spacing {float(np.min(d)):.4g} < {min_df:g} Hz")

    return errors


def print_plan_table() -> None:
    print(f"target_total_rms = {TARGET_TOTAL_RMS} pu\n", flush=True)
    print(
        f"{'band':<4} {'f range':<14} {'N':>3} {'T_win':>6} {'df_fft':>8} "
        f"{'df_tone':>8} {'t_end_WT':>8} {'t_end_FMU':>9} {'cycles':>7} {'amp':>8}",
        flush=True,
    )
    for band in BAND_ORDER:
        s = get_band_spec(band)
        df_label = (
            "seg"
            if s.tone_segments
            else f"{s.df_tone_hz:8.2f}"
        )
        print(
            f"{s.band:<4} [{s.f_min_hz:g},{s.f_max_hz:g}] {s.n_tones:3d} "
            f"{s.t_win_s:6.0f} {s.df_fft_hz:8.4f} {df_label:>8} "
            f"{t_end_s(s, 'wt'):8.0f} {t_end_s(s, 'fmu'):9.0f} "
            f"{s.cycles_at_f_min:7.1f} {s.amp_per_tone_pu:8.4f}",
            flush=True,
        )


def print_and_assert() -> None:
    print_plan_table()
    all_errors: list[str] = []
    for band in BAND_ORDER:
        all_errors.extend(_validate_band(get_band_spec(band)))
    if all_errors:
        for e in all_errors:
            print(f"ERROR: {e}", flush=True)
        raise SystemExit(1)
    print("\nAll band-split multitone checks passed.", flush=True)


if __name__ == "__main__":
    print_and_assert()
