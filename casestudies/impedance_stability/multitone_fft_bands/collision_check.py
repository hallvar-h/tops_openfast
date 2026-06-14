from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def check_msd_multitone_list(f_list, f1: float = 50.0, tol: float = 1e-9) -> dict:
    """
    Check collisions for modified sequence domain extraction.

    For each modified frequency fm:
        upper bin  = f1 + fm
        mirror bin = abs(f1 - fm)

    Returns a dict (no printing) so callers can decide how to present.
    """

    f = np.asarray(f_list, dtype=float)

    same_mirror: list[tuple[float, float]] = []
    cross_coll: list[tuple[float, float]] = []
    special: list[tuple[float, str]] = []

    # Single-frequency special cases
    for fm in f:
        if abs(fm - 0.0) < tol:
            special.append((float(fm), "hits 50 Hz fundamental"))
        if abs(fm - f1) < tol:
            special.append((float(fm), "hits DC mirror bin"))
        if abs(fm - 2 * f1) < tol:
            special.append((float(fm), "mirror bin lands on 50 Hz fundamental"))

    # Pairwise collisions
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            fi, fj = float(f[i]), float(f[j])

            # |f1-fi| = |f1-fj|
            if abs(abs(f1 - fi) - abs(f1 - fj)) < tol:
                same_mirror.append((fi, fj))

            # f1+fi = |f1-fj|  OR  f1+fj = |f1-fi|
            if abs((f1 + fi) - abs(f1 - fj)) < tol or abs((f1 + fj) - abs(f1 - fi)) < tol:
                cross_coll.append((fi, fj))

    return {"same_mirror": same_mirror, "cross_coll": cross_coll, "special": special}


@dataclass(frozen=True)
class ToneGrid:
    freqs_hz: np.ndarray
    df_fft_hz: float
    df_tone_hz: float
    f_min_hz: float
    f_max_hz: float


def propose_tone_grid(
    *,
    f_min_hz: float,
    f_max_hz: float,
    n_tones: int,
    df_fft_hz: float,
    df_tone_hz: float,
) -> ToneGrid:
    """
    Propose a deterministic tone list that is:
    - Bin-centered on df_fft_hz
    - At least df_tone_hz separated
    - Spans [f_min_hz, f_max_hz]
    """

    f_min = float(f_min_hz)
    f_max = float(f_max_hz)
    n = int(n_tones)
    df_fft = float(df_fft_hz)
    df_tone = float(df_tone_hz)

    if not (np.isfinite(f_min) and np.isfinite(f_max) and f_min > 0.0 and f_max > f_min):
        raise ValueError("Require 0 < f_min_hz < f_max_hz")
    if n < 2:
        raise ValueError("n_tones must be >= 2")
    if not (np.isfinite(df_fft) and df_fft > 0.0):
        raise ValueError("df_fft_hz must be > 0")
    if not (np.isfinite(df_tone) and df_tone > 0.0):
        raise ValueError("df_tone_hz must be > 0")

    k0 = int(np.ceil(f_min / df_fft))
    k1 = int(np.floor(f_max / df_fft))
    if k1 <= k0:
        raise ValueError("No FFT bins available in requested [f_min, f_max] range.")
    f_grid = (np.arange(k0, k1 + 1, dtype=float) * df_fft).astype(float)

    stride = max(1, int(np.ceil(df_tone / df_fft)))
    f_spaced = f_grid[::stride]
    if f_spaced.size < n:
        raise ValueError(
            f"Requested n_tones={n} but only {f_spaced.size} bins available with "
            f"df_fft={df_fft:g} Hz and min tone spacing {df_tone:g} Hz."
        )

    idx = np.linspace(0, f_spaced.size - 1, n, dtype=int)
    freqs = np.asarray(f_spaced[idx], dtype=float)

    return ToneGrid(
        freqs_hz=freqs,
        df_fft_hz=df_fft,
        df_tone_hz=df_tone,
        f_min_hz=f_min,
        f_max_hz=f_max,
    )
