"""
Publication-style Z_bus Bode figures from ``impedance_matrix_fft.csv``.

Use these for the thesis — not ``Zdev_sequence_bode*.png`` (device admittance is often ill-conditioned).

Figures are written automatically by ``multitone_fft_bands/pipeline.py``. To regenerate
from existing merged CSVs, run the band pipeline with ``--no-sims`` or call
``plot_zbus_pp`` from a short script. Full workflow (from repo root)::

    python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_wt.py
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from casestudies.impedance_stability.identification.impedance_matrix_uic_wt import (
    _mag_db_bode_plot,
    _phase_deg_bode_plot,
    _zm_bode_plot_arrays,
    plot_sequence_four_zm_bode,
)
from casestudies.impedance_stability.plots.bode_axes import (
    BODE_COLORS,
    apply_bode_mag_phase_axes,
    draw_freq_marker,
    format_log_freq_axis,
    matrix_bode_suptitle,
    phase_ylabel,
    semilogx_bode,
)

DEFAULT_F_MIN_HZ = 1.0
# HF band upper edge (see multitone_fft_bands/band_specs.F_MAX_HZ).
DEFAULT_F_MAX_HZ = 10.0
# Off-diagonal Z_bus are ~40 dB below ++; mask only when clearly below diagonal (avoids LF "cracks").
MASK_WEAK_CROSS_TERMS_DB = 25.0
# Fixed magnitude window on Z++ so ±0.2 dB ripple does not look like spikes (autoscale artifact).
PP_MAG_YLIM_DB = 1.0


def _load_zbus_pp(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "f_Hz" not in df.columns:
        raise KeyError(f"{csv_path}: missing f_Hz")
    for col in ("Zbus_m00_re", "Zbus_m00_im"):
        if col not in df.columns:
            raise KeyError(f"{csv_path}: missing {col} (run chirp ID post-process first)")
    f = df["f_Hz"].to_numpy(dtype=float)
    z = df["Zbus_m00_re"].to_numpy(dtype=float) + 1j * df["Zbus_m00_im"].to_numpy(dtype=float)
    return f, z


def _load_zbus_sequence(
    csv_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(csv_path)
    f = df["f_Hz"].to_numpy(dtype=float)
    z00 = df["Zbus_m00_re"].to_numpy(dtype=float) + 1j * df["Zbus_m00_im"].to_numpy(dtype=float)
    z01 = df["Zbus_m01_re"].to_numpy(dtype=float) + 1j * df["Zbus_m01_im"].to_numpy(dtype=float)
    z10 = df["Zbus_m10_re"].to_numpy(dtype=float) + 1j * df["Zbus_m10_im"].to_numpy(dtype=float)
    z11 = df["Zbus_m11_re"].to_numpy(dtype=float) + 1j * df["Zbus_m11_im"].to_numpy(dtype=float)
    det = df["detIpert_abs"].to_numpy(dtype=float) if "detIpert_abs" in df.columns else None
    return f, z00, z01, z10, z11, det


def _band_mask(
    f: np.ndarray,
    *,
    f_min_hz: float,
    f_max_hz: float,
) -> np.ndarray:
    return np.isfinite(f) & (f >= float(f_min_hz)) & (f <= float(f_max_hz))


def plot_zbus_pp(
    csv_path: str,
    out_png: str,
    *,
    label: str = "identified",
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = DEFAULT_F_MAX_HZ,
    mark_hz: float | None = None,
    title: str | None = None,
) -> None:
    """Single-panel Z_bus++ magnitude and phase (terminal impedance, sequence component)."""
    f, z = _load_zbus_pp(csv_path)
    m = _band_mask(f, f_min_hz=f_min_hz, f_max_hz=f_max_hz)
    f = f[m]
    z = z[m]
    if f.size < 2:
        raise ValueError(f"Need ≥2 points in [{f_min_hz}, {f_max_hz}] Hz in {csv_path}")

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    mag = _mag_db_bode_plot(z)
    ph = _phase_deg_bode_plot(z)
    semilogx_bode(ax_mag, f, mag, label=label)
    semilogx_bode(ax_ph, f, ph, label=label)
    med = float(np.nanmedian(mag))
    if np.isfinite(med):
        half = float(PP_MAG_YLIM_DB) / 2.0
        ax_mag.set_ylim(med - half, med + half)
    for ax in (ax_mag, ax_ph):
        draw_freq_marker(ax, mark_hz)
    ax_mag.set_ylabel("|Z|")
    ax_ph.set_ylabel(phase_ylabel())
    for ax in (ax_mag, ax_ph):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    apply_bode_mag_phase_axes(ax_mag, ax_ph, float(f.min()), float(f.max()), show_freq_axis=True)
    fig.suptitle(
        title
        or matrix_bode_suptitle(
            f"Z_bus,++ [{f_min_hz:g}, {f_max_hz:g}] Hz",
            extra=label,
        ),
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_zbus_pp_overlay(
    series: Iterable[tuple[str, str]],
    out_png: str,
    *,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = DEFAULT_F_MAX_HZ,
    mark_hz: float | None = None,
    title: str | None = None,
) -> None:
    """Overlay Z_bus++ for several cases (e.g. baseline WT vs coupled FMU)."""
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
    for i, (label, csv_path) in enumerate(series):
        f, z = _load_zbus_pp(csv_path)
        m = _band_mask(f, f_min_hz=f_min_hz, f_max_hz=f_max_hz)
        f = f[m]
        z = z[m]
        c = BODE_COLORS[i % len(BODE_COLORS)]
        semilogx_bode(ax_mag, f, _mag_db_bode_plot(z), label=label, color=c)
        semilogx_bode(ax_ph, f, _phase_deg_bode_plot(z), label=label, color=c)
    for ax in (ax_mag, ax_ph):
        draw_freq_marker(ax, mark_hz)
    ax_mag.set_ylabel("|Z|")
    ax_ph.set_ylabel(phase_ylabel())
    lo, hi = float(f_min_hz), float(f_max_hz)
    for ax in (ax_mag, ax_ph):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    apply_bode_mag_phase_axes(ax_mag, ax_ph, lo, hi, show_freq_axis=True)
    fig.suptitle(
        title
        or matrix_bode_suptitle(f"Z_bus,++ comparison [{f_min_hz:g}, {f_max_hz:g}] Hz"),
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_zbus_diagonal_sequence(
    csv_path: str,
    out_png: str,
    *,
    label: str = "",
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = DEFAULT_F_MAX_HZ,
    mark_hz: float | None = None,
    mask_cross_db: float = MASK_WEAK_CROSS_TERMS_DB,
) -> None:
    """Four-panel Z_bus with weak cross-terms masked (+- / -+ may show no data)."""
    f, z00, z01, z10, z11, det = _load_zbus_sequence(csv_path)
    zb00, zb01, zb10, zb11 = _zm_bode_plot_arrays(
        z00, z01, z10, z11, det_ipert_abs=det, cross_snr_db=float(mask_cross_db)
    )
    extra = label or "identified"
    plot_sequence_four_zm_bode(
        f,
        zb00,
        zb01,
        zb10,
        zb11,
        out_png,
        title=matrix_bode_suptitle("Z_bus: Zm (++ / +- / -+ / --)", extra=extra),
        mag_db_label="|Z|",
        phase_ylabel="",
        legend_prefix="Z",
        mark_hz=mark_hz,
        plot_f_min_hz=float(f_min_hz),
        decimate_log_step=0.03,
    )


def plot_zbus_dq_diagonal_midband(
    csv_path: str,
    out_png: str,
    *,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = DEFAULT_F_MAX_HZ,
    mark_hz: float | None = None,
) -> None:
    """Z_bus in dq frame: Z_dd and Z_qq only."""
    df = pd.read_csv(csv_path)
    f_all = df["f_Hz"].to_numpy(dtype=float)
    band = _band_mask(f_all, f_min_hz=f_min_hz, f_max_hz=f_max_hz)
    f = f_all[band]

    def _z(cr: str, ci: str) -> np.ndarray:
        return (df[cr].to_numpy(dtype=float) + 1j * df[ci].to_numpy(dtype=float))[band]

    elems = [("Z_dd", _z("Zbus_dd_re", "Zbus_dd_im")), ("Z_qq", _z("Zbus_qq_re", "Zbus_qq_im"))]
    fig = plt.figure(figsize=(10.0, 5.5))
    outer = fig.add_gridspec(1, 2, wspace=0.25)
    for k, (name, z) in enumerate(elems):
        sub = outer[0, k].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.12)
        ax_mag = fig.add_subplot(sub[0, 0])
        ax_ph = fig.add_subplot(sub[1, 0], sharex=ax_mag)
        mag = _mag_db_bode_plot(z)
        ph = _phase_deg_bode_plot(z)
        semilogx_bode(ax_mag, f, mag)
        semilogx_bode(ax_ph, f, ph)
        med = float(np.nanmedian(mag))
        if np.isfinite(med):
            half = float(PP_MAG_YLIM_DB) / 2.0
            ax_mag.set_ylim(med - half, med + half)
        ax_mag.set_title(name)
        ax_mag.set_ylabel("|Z_bus| (dB re 1 pu)")
        ax_ph.set_ylabel("∠Z_bus (deg, unwrap)")
        for ax in (ax_mag, ax_ph):
            ax.grid(True, which="both", alpha=0.3)
            draw_freq_marker(ax, mark_hz)
        apply_bode_mag_phase_axes(ax_mag, ax_ph, float(f.min()), float(f.max()), show_freq_axis=True)
    fig.suptitle(
        f"Z_bus diagonal (dq) [{f_min_hz:g}, {f_max_hz:g}] Hz — Z_dq, Z_qd omitted",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
