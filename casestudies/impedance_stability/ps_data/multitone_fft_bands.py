"""
Band-limited multitone injection for split FFT impedance ID.

Used by CaseLoader entries ``multitone_bands_{lf,mf,hf}_{re,im}`` and FMU equivalents.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    RANDOM_SEED,
    TARGET_TOTAL_RMS,
    BandId,
    BandSpec,
    PlantId,
    get_band_spec,
)
from casestudies.impedance_stability.paths import REPO_ROOT
from casestudies.impedance_stability.ps_data.fmu_openfast import _fmu_path, load_fmu_multisine
from casestudies.impedance_stability.ps_data.uic_wt_three_bus import uic_wt_three_bus_model


def _build_uniform_pert_frequencies(
    *,
    spec_f_min: float,
    spec_f_max: float,
    n_tones: int,
    df_tone_hz: float,
    df_fft_hz: float,
) -> np.ndarray:
    """Bin-centered tone grid on the FFT lattice (``df_fft`` bins, ``df_tone`` stride)."""
    f_min = float(spec_f_min)
    f_max = float(spec_f_max)
    n_t = int(n_tones)
    df_fft = float(df_fft_hz)
    df_tone = float(df_tone_hz)

    k0 = int(np.ceil(f_min / df_fft))
    k1 = int(np.floor(f_max / df_fft))
    if k1 <= k0:
        raise ValueError("No FFT bins in [f_min, f_max] for band.")
    f_grid = (np.arange(k0, k1 + 1, dtype=float) * df_fft).astype(float)
    stride = max(1, int(np.ceil(df_tone / df_fft)))
    f_spaced = f_grid[::stride]
    if f_spaced.size < n_t:
        raise ValueError(
            f"Requested n_tones={n_t} but only {f_spaced.size} bins with "
            f"df_fft={df_fft:g} Hz, df_tone={df_tone:g} Hz."
        )
    idx = np.linspace(0, f_spaced.size - 1, n_t, dtype=int)
    if n_t >= 2:
        idx[0] = 0
        idx[-1] = int(f_spaced.size - 1)
    freqs = np.asarray(f_spaced[idx], dtype=float)
    k_lo = int(np.round(f_min / df_fft))
    k_hi = int(np.round(f_max / df_fft))
    freqs[0] = float(k_lo * df_fft)
    freqs[-1] = float(k_hi * df_fft)
    return freqs


def _build_pert_frequencies(spec: BandSpec) -> np.ndarray:
    if spec.tone_segments is not None:
        freqs = np.asarray(spec.tone_frequencies_hz, dtype=float)
        if freqs.size != int(spec.n_tones):
            raise ValueError(
                f"LF segmented tone count {freqs.size} != n_tones {spec.n_tones}"
            )
        return freqs
    return _build_uniform_pert_frequencies(
        spec_f_min=spec.f_min_hz,
        spec_f_max=spec.f_max_hz,
        n_tones=spec.n_tones,
        df_tone_hz=spec.df_tone_hz,
        df_fft_hz=spec.df_fft_hz,
    )


def _build_pert_rows(
    *,
    axis: str,
    spec: BandSpec,
    target_total_rms: float = TARGET_TOTAL_RMS,
    random_seed: int = RANDOM_SEED,
) -> list[list[Any]]:
    axis = str(axis).strip().lower()
    freqs = _build_pert_frequencies(spec)
    n_t = int(freqs.size)
    amp = float(target_total_rms * np.sqrt(2.0 / float(n_t)))
    phases = (2.0 * np.pi) * np.random.RandomState(int(random_seed)).rand(n_t)
    rows: list[list[Any]] = []
    for k, (f_hz, ph) in enumerate(zip(freqs, phases)):
        rows.append([f"Ipert_{k:04d}", "B2", 0.0, 0.0, amp, float(f_hz), float(ph), axis])
    return rows


def load_multitone_fft_band(
    *,
    plant: PlantId,
    band: BandId,
    axis: str,
    target_total_rms: float = TARGET_TOTAL_RMS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Load WT or FMU PS model with band-limited multitone perturbation."""
    spec = get_band_spec(band)
    axis = str(axis).strip().lower()
    pert_rows = _build_pert_rows(
        axis=axis,
        spec=spec,
        target_total_rms=float(target_total_rms),
        random_seed=int(random_seed),
    )
    header = [["name", "bus", "I_re", "I_im", "amp", "f", "phase", "axis"], *pert_rows]

    if plant == "wt":
        return uic_wt_three_bus_model(perturbation_rows=header)

    if plant != "fmu":
        raise ValueError(f"plant must be 'wt' or 'fmu', got {plant!r}")

    root = str(REPO_ROOT)
    fmu_path = _fmu_path()
    model = load_fmu_multisine(axis=axis)
    model["perturbations"] = {"BusCurrentPerturbation": header}
    if "FMUtoUICdrivetrain" in model and "FMUtoUICdrivetrain" in model["FMUtoUICdrivetrain"]:
        row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"][1]
        row[4] = fmu_path
        row[7] = os.path.join(root, "openfast_fmu", "resources", "wd.txt")
        row[8] = root
    return model


def load_wt_multitone_fft_band(*, band: BandId, axis: str) -> dict[str, Any]:
    return load_multitone_fft_band(plant="wt", band=band, axis=axis)


def load_fmu_multitone_fft_band(*, band: BandId, axis: str) -> dict[str, Any]:
    return load_multitone_fft_band(plant="fmu", band=band, axis=axis)


def get_band_injected_frequencies(band: BandId) -> np.ndarray:
    """Tone frequencies that will be injected (re/im share the same f list)."""
    spec = get_band_spec(band)
    freqs = _build_pert_frequencies(spec)
    return freqs[np.isfinite(freqs) & (freqs > 0.0)]
