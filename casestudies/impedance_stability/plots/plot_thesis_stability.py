"""
Thesis figures for impedance-based stability: Z_bus, Y_dev, and Nyquist loop 1 + Z_bus Y_dev.

Use Y_dev (device admittance) — not Z_dev = inv(Y_dev) — together with Z_bus in the loop.

Typical loop (scalar ++ / single-input channel)::

    L(jω) = Z_bus,++(jω) · Y_dev,++(jω),   closed-loop poles: 1 + L(jω) = 0 → critical point L = -1.

``plot_nyquist_loop_pp`` combines **Bode magnitude vs frequency** (|L|, |1+L|) with a **Nyquist**
plane (**Re L** vs **Im L**; frequency is implicit along the curve — not an axis).
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from casestudies.impedance_stability.identification.impedance_matrix_uic_wt import (
    _mag_db_bode_plot,
    _phase_deg_bode_plot,
)
from casestudies.impedance_stability.plots.bode_axes import (
    BODE_COLOR,
    BODE_COLORS,
    OVERLAY_GRID_HSPACE,
    OVERLAY_GRID_WSPACE,
    OVERLAY_PANEL_MAGPH_HSPACE,
    OVERLAY_PHASE_REF_HZ,
    OVERLAY_SERIES_COLORS,
    MAG_DB_UNIT,
    OVERLAY_DPI,
    OVERLAY_FIGSIZE_DQ,
    OVERLAY_FIGSIZE_PP,
    apply_bode_mag_phase_axes,
    apply_thesis_plot_style,
    draw_freq_marker,
    dq_panel_title,
    finalize_overlay_figure,
    format_log_freq_axis,
    format_small_locus_inset_axes,
    matrix_bode_suptitle,
    phase_ylabel,
    semilogx_bode,
    style_bode_panel_axes,
)
from casestudies.impedance_stability.identification.impedance_matrix_uic_wt import KAPPA_YDEV_MAX
from casestudies.impedance_stability.plots.plot_thesis_zbus import (
    DEFAULT_F_MAX_HZ,
    DEFAULT_F_MIN_HZ,
)

# Matches HF band upper edge (multitone_fft_bands/band_specs.F_MAX_HZ).
PLOT_F_MAX_HZ = DEFAULT_F_MAX_HZ

# Legacy fallback when CSV has no kappa_Ydev column (|det(Y)| grows with |Y|, not κ).
DET_YDEV_ABS_MAX = 25.0

# Main Nyquist frame when |L| ≪ 1: show (-1,0) without wide side margins.
NYQUIST_RE_LO = -1.25
NYQUIST_RE_HI = 0.25

# Scalar loop channels for standalone Nyquist figures (sequence ++ vs dq qq).
_NYQUIST_SCALAR_CHANNELS: dict[str, dict[str, str]] = {
    "pp": {
        "z_re": "Zbus_m00_re",
        "z_im": "Zbus_m00_im",
        "y_re": "Ydev_m00_re",
        "y_im": "Ydev_m00_im",
        "tex": r"Z_{bus,++}Y_{dev,++}",
    },
    "qq": {
        "z_re": "Zbus_qq_re",
        "z_im": "Zbus_qq_im",
        "y_re": "Ydev_qq_re",
        "y_im": "Ydev_qq_im",
        "tex": r"Z_{bus,qq}Y_{dev,qq}",
    },
}


def _ydev_trust_masks(
    df: pd.DataFrame,
    band: np.ndarray,
    *,
    kappa_max: float = KAPPA_YDEV_MAX,
    det_y_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split bins into trusted vs de-emphasized using κ(Y) when available."""
    if "kappa_Ydev" in df.columns:
        k = df["kappa_Ydev"].to_numpy(dtype=float)
        good = band & np.isfinite(k) & (k <= float(kappa_max))
        weak = band & ~good
        return good, weak, rf"$\kappa(Y)\leq${kappa_max:g}"
    cap = _det_ydev_cap(df) if det_y_cap is None else float(det_y_cap)
    if "detYdev_abs" in df.columns:
        dy = df["detYdev_abs"].to_numpy(dtype=float)
        good = band & np.isfinite(dy) & (dy <= cap)
        weak = band & ~good
        return good, weak, rf"detYdev>{cap:g}"
    return band, np.zeros_like(band, dtype=bool), ""


def collect_ydev_low_trust_points(
    df: pd.DataFrame,
    *,
    f_min_hz: float,
    f_max_hz: float,
    kappa_max: float = KAPPA_YDEV_MAX,
) -> tuple[list[dict[str, float]], str]:
    """Frequency bins where Y_dev identification is poorly conditioned."""
    f = df["f_Hz"].to_numpy(dtype=float)
    ydd = _complex_col(df, "Ydev_dd_re", "Ydev_dd_im")
    band = _qa_mask(df, f, f_min_hz=float(f_min_hz), f_max_hz=float(f_max_hz), det_y_max=1e30)
    band &= np.isfinite(f) & np.isfinite(ydd.real) & np.isfinite(ydd.imag)
    _, weak, criterion = _ydev_trust_masks(df, band, kappa_max=float(kappa_max))
    rows: list[dict[str, float]] = []
    for i in np.flatnonzero(weak):
        entry: dict[str, float] = {
            "f_Hz": float(f[i]),
            "Ydd_mag_db": float(20.0 * np.log10(np.abs(ydd[i]) + 1e-30)),
        }
        for col in ("kappa_Ydev", "detVt_abs", "detYdev_abs", "detIpert_abs"):
            if col in df.columns:
                v = float(df[col].iloc[i])
                if np.isfinite(v):
                    entry[col] = v
        rows.append(entry)
    rows.sort(key=lambda r: float(r.get("kappa_Ydev", r.get("detYdev_abs", 0.0))), reverse=True)
    return rows, criterion


def format_ydev_trust_report(
    entries: Iterable[tuple[str, list[dict[str, float]]]],
    *,
    f_min_hz: float,
    f_max_hz: float,
    kappa_max: float = KAPPA_YDEV_MAX,
    criterion: str = "",
) -> str:
    """Plain-text side-panel report for overlay figures."""
    crit = f"kappa(Y) > {kappa_max:g}"
    lines = [
        "Low-trust Y_dev bins",
        f"Band: [{f_min_hz:g}, {f_max_hz:g}] Hz",
        f"Criterion: {crit}",
        "",
    ]
    any_weak = False
    for label, rows in entries:
        lines.append(f"{label}:")
        if not rows:
            lines.append("  (none)")
            lines.append("")
            continue
        any_weak = True
        for r in rows[:24]:
            parts = [f"  {r['f_Hz']:.4g} Hz", f"|Ydd|={r['Ydd_mag_db']:.1f} dB"]
            if "kappa_Ydev" in r:
                parts.append(f"kappa={r['kappa_Ydev']:.1f}")
            if "detVt_abs" in r:
                parts.append(f"detVt={r['detVt_abs']:.2e}")
            if "detYdev_abs" in r:
                parts.append(f"detY={r['detYdev_abs']:.2e}")
            lines.append("  ".join(parts))
        if len(rows) > 24:
            lines.append(f"  ... +{len(rows) - 24} more")
        lines.append("")
    if not any_weak:
        lines.append("All bins pass the trust criterion in this band.")
    return "\n".join(lines).rstrip() + "\n"


def _draw_trust_report_panel(fig: plt.Figure, gs_report, report_text: str) -> None:
    ax_rep = fig.add_subplot(gs_report)
    ax_rep.axis("off")
    ax_rep.text(
        0.0,
        1.0,
        report_text,
        transform=ax_rep.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        family="monospace",
        linespacing=1.25,
        wrap=True,
    )


def _det_ydev_cap(df: pd.DataFrame, default: float = DET_YDEV_ABS_MAX) -> float:
    """Allow more bins at HF while dropping outliers (WT often exceeds 25 by ~4 Hz)."""
    if "detYdev_abs" not in df.columns:
        return float(default)
    dy = df["detYdev_abs"].to_numpy(dtype=float)
    ok = np.isfinite(dy) & (dy > 0.0)
    if not np.any(ok):
        return float(default)
    med = float(np.median(dy[ok]))
    return float(min(120.0, max(default, 8.0 * med)))


def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "f_Hz" not in df.columns:
        raise KeyError(f"{csv_path}: missing f_Hz")
    return df


def _complex_col(df: pd.DataFrame, re: str, im: str) -> np.ndarray:
    return df[re].to_numpy(dtype=float) + 1j * df[im].to_numpy(dtype=float)


def _overlay_series_legend_handles(series_list: list[tuple[str, str]]) -> list[Line2D]:
    """Legend entries matching overlay Bode line style (color + marker)."""
    return [
        Line2D(
            [0],
            [0],
            color=OVERLAY_SERIES_COLORS[i % len(OVERLAY_SERIES_COLORS)],
            marker="o",
            ms=4,
            mew=0.3,
            mec="white",
            lw=1.1,
            label=label,
        )
        for i, (label, _) in enumerate(series_list)
    ]


def _matrix_mag_ylabel(matrix_prefix: str) -> str:
    """Short axis label; units are in ``matrix_bode_suptitle``."""
    sym = "Z" if str(matrix_prefix).strip() == "Zbus" else "Y"
    return f"|{sym}|"


def _plot_overlay_trace(
    ax_mag: plt.Axes,
    ax_ph: plt.Axes,
    f: np.ndarray,
    z: np.ndarray,
    *,
    color: str,
    phase_anchor_hz: float | None,
) -> None:
    zz = np.asarray(z, dtype=complex)
    semilogx_bode(
        ax_mag,
        f,
        _mag_db_bode_plot(zz),
        color=color,
        lw=1.05,
        ms=2.8,
    )
    if phase_anchor_hz is not None:
        ph = _phase_deg_bode_plot(zz, f, f_anchor_hz=float(phase_anchor_hz))
    else:
        ph = _phase_deg_bode_plot(zz)
    semilogx_bode(
        ax_ph,
        f,
        ph,
        color=color,
        lw=1.05,
        ms=2.8,
    )


def _overlay_phasors_common_ref(
    z_series: list[np.ndarray],
    f: np.ndarray,
    f_ref_hz: float,
    *,
    ref_series: int = 0,
) -> list[np.ndarray]:
    """
    Rotate all overlay phasors by the same angle (baseline at ``f_ref_hz``).

    Preserves phase *difference* between cases; avoids per-trace rotation plus
    independent unwrap from the band edge (which can add a false ~360° offset at low f).
    """
    if not z_series:
        return z_series
    f = np.asarray(f, dtype=float)
    z_ref = np.asarray(z_series[ref_series], dtype=complex)
    m = np.isfinite(f) & np.isfinite(z_ref.real) & np.isfinite(z_ref.imag)
    if not np.any(m):
        return [np.asarray(z, dtype=complex).copy() for z in z_series]
    ii = np.flatnonzero(m)
    j = int(np.argmin(np.abs(f[m] - float(f_ref_hz))))
    idx = ii[j]
    ref = z_ref[idx]
    if not np.isfinite(ref.real) or not np.isfinite(ref.imag) or abs(ref) < 1e-30:
        return [np.asarray(z, dtype=complex).copy() for z in z_series]
    rot = np.exp(-1j * np.angle(ref))
    return [np.asarray(z, dtype=complex) * rot for z in z_series]


def _overlay_phase_ref_hz(phase_ref_hz: float | None) -> float:
    """Default ``OVERLAY_PHASE_REF_HZ`` (0.1 Hz); explicit value overrides."""
    default = float(OVERLAY_PHASE_REF_HZ)
    if phase_ref_hz is None:
        return default
    ref = float(phase_ref_hz)
    if not np.isfinite(ref) or ref <= 0.0:
        return default
    return ref


def _mimo_2x2_from_csv(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Build (n, 2, 2) complex matrices from ``{prefix}_dd/dq/qd/qq`` CSV columns."""
    dd = _complex_col(df, f"{prefix}_dd_re", f"{prefix}_dd_im")
    dq = _complex_col(df, f"{prefix}_dq_re", f"{prefix}_dq_im")
    qd = _complex_col(df, f"{prefix}_qd_re", f"{prefix}_qd_im")
    qq = _complex_col(df, f"{prefix}_qq_re", f"{prefix}_qq_im")
    n = int(dd.size)
    m = np.empty((n, 2, 2), dtype=complex)
    m[:, 0, 0] = dd
    m[:, 0, 1] = dq
    m[:, 1, 0] = qd
    m[:, 1, 1] = qq
    return m


def _loop_eigenvalues_tracked(
    z_bus: np.ndarray,
    y_dev: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalues of L = Z_bus @ Y_dev per frequency bin.

    Tracks λ by |λ| order (λ_a smaller, λ_b larger) — same in dq and mirror frames.
    """
    n = int(z_bus.shape[0])
    lam_a = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    lam_b = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    for i in np.flatnonzero(mask):
        if not np.all(np.isfinite(z_bus[i])) or not np.all(np.isfinite(y_dev[i])):
            continue
        ev = np.linalg.eigvals(z_bus[i] @ y_dev[i])
        ev = ev[np.isfinite(ev)]
        if ev.size < 2:
            continue
        order = np.argsort(np.abs(ev))
        lam_a[i] = ev[order[0]]
        lam_b[i] = ev[order[1]]
    return lam_a, lam_b


def _plot_nyquist_scalar_ref(
    ax: plt.Axes,
    mask: np.ndarray,
    l: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str = ".",
    zorder: int = 2,
) -> None:
    """Scalar loop locus with ω<0 mirror (for ++ or Z_dd Y_dd references)."""
    if not np.any(mask):
        return
    re_p = np.real(l[mask])
    im_p = np.imag(l[mask])
    ax.plot(
        re_p,
        im_p,
        lw=0.9,
        color=color,
        ls=":",
        marker=marker,
        ms=3,
        mew=0.5,
        alpha=0.88,
        label=label,
        zorder=zorder,
    )
    ax.plot(re_p, -im_p, lw=0.75, color=color, ls="--", alpha=0.42, zorder=zorder - 1)


def _origin_zoom_inset_axes(
    ax_n: plt.Axes,
    re_all: np.ndarray,
    im_all: np.ndarray,
    *,
    l_max: float,
) -> plt.Axes | None:
    """Create inset axes + connector lines when the locus sits near the origin (|L| ≪ 1)."""
    if l_max >= 0.15 or re_all.size < 2:
        return None
    re_lo, re_hi = float(np.min(re_all)), float(np.max(re_all))
    im_lo = float(min(np.min(im_all), -np.max(im_all)))
    im_hi = float(max(np.max(im_all), -np.min(im_all)))
    pr = max(re_hi - re_lo, 1e-12)
    pi = max(im_hi - im_lo, 1e-12)
    icx = 0.5 * (re_lo + re_hi)
    icy = 0.5 * (im_lo + im_hi)
    pad = 1.18
    half = max(0.5 * pr * pad, 0.5 * pi * pad) + max(2e-5, 0.05 * l_max)
    axins = inset_axes(
        ax_n,
        width="50%",
        height="40%",
        loc="lower center",
        bbox_to_anchor=(0.0, 0.05, 1.0, 0.92),
        bbox_transform=ax_n.transAxes,
        borderpad=1.0,
    )
    axins.axhline(0.0, color="0.5", lw=0.5)
    axins.axvline(0.0, color="0.5", lw=0.5)
    axins.set_xlim(icx - half, icx + half)
    axins.set_ylim(icy - half, icy + half)
    axins.set_aspect("equal", adjustable="box")
    axins.grid(True, alpha=0.3)
    axins.set_title(r"zoom", fontsize=8, pad=3)
    format_small_locus_inset_axes(axins)
    mark_inset(
        ax_n,
        axins,
        loc1=2,
        loc2=1,
        fc="none",
        ec="0.45",
        lw=0.9,
        ls="--",
        alpha=0.85,
        zorder=1,
    )
    return axins


def _qa_mask(
    df: pd.DataFrame,
    f: np.ndarray,
    *,
    f_min_hz: float,
    f_max_hz: float,
    det_y_max: float = DET_YDEV_ABS_MAX,
) -> np.ndarray:
    m = np.isfinite(f) & (f >= float(f_min_hz)) & (f <= float(f_max_hz))
    if "detYdev_abs" in df.columns:
        dy = df["detYdev_abs"].to_numpy(dtype=float)
        m &= np.isfinite(dy) & (dy <= float(det_y_max))
    if "detVt_abs" in df.columns:
        dv = df["detVt_abs"].to_numpy(dtype=float)
        m &= np.isfinite(dv) & (dv > 0.0)
    if "detIpert_abs" in df.columns:
        di = df["detIpert_abs"].to_numpy(dtype=float)
        m &= np.isfinite(di) & (di > 0.0)
    return m


def plot_ydev_pp(
    csv_path: str,
    out_png: str,
    *,
    label: str = "",
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    det_y_warn: float | None = None,
) -> None:
    """
    Device admittance Y_dev,++ (I_a U^{-1}) on the full ID grid.

    Bins with large ``kappa_Ydev`` are drawn faded; falls back to ``detYdev_abs`` on old CSVs.
    """
    df = _load_csv(csv_path)
    f = df["f_Hz"].to_numpy(dtype=float)
    y = _complex_col(df, "Ydev_m00_re", "Ydev_m00_im")
    band = _qa_mask(
        df, f, f_min_hz=f_min_hz, f_max_hz=f_max_hz, det_y_max=1e30
    )  # band only; do not drop HF on detY
    band &= np.isfinite(y.real) & np.isfinite(y.imag)
    good, weak, qa_tag = _ydev_trust_masks(
        df, band, det_y_cap=det_y_warn if det_y_warn is not None else None
    )

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    def _plot_seg(mask: np.ndarray, *, color: str, alpha: float, lw: float) -> None:
        if not np.any(mask):
            return
        yy = np.where(mask, y, np.nan + 1j * np.nan)
        semilogx_bode(ax_mag, f, _mag_db_bode_plot(yy), color=color or BODE_COLOR, lw=lw, alpha=alpha)
        semilogx_bode(ax_ph, f, _phase_deg_bode_plot(yy), color=color or BODE_COLOR, lw=lw, alpha=alpha)

    _plot_seg(good, color="C0", alpha=1.0, lw=1.2)
    _plot_seg(weak, color="C0", alpha=0.35, lw=0.9)

    if np.any(weak) and np.any(good):
        f_hi = float(np.nanmax(f[good]))
        for ax in (ax_mag, ax_ph):
            ax.axvspan(f_hi, float(f_max_hz), color="0.9", alpha=0.35, zorder=0)

    for ax in (ax_mag, ax_ph):
        draw_freq_marker(ax, mark_hz)
    ax_mag.set_ylabel("|Y|")
    ax_ph.set_ylabel(phase_ylabel())
    f_ok = f[good]
    hi = float(f_ok.max()) if f_ok.size else float(f_max_hz)
    lo = float(f_ok.min()) if f_ok.size else float(f_min_hz)
    for ax in (ax_mag, ax_ph):
        ax.grid(True, which="both", alpha=0.3)
    apply_bode_mag_phase_axes(ax_mag, ax_ph, lo, hi, show_freq_axis=True)
    title = matrix_bode_suptitle(
        f"Y_dev,++ [{lo:g}, {hi:g}] Hz",
        extra=(f"{label}; solid: {qa_tag}" if label else f"solid: {qa_tag}"),
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ydev_pp_overlay(
    series: Iterable[tuple[str, str]],
    out_png: str,
    *,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    kappa_max: float = KAPPA_YDEV_MAX,
    trust_side_report: bool = False,
    trust_report_path: str | None = None,
    phase_ref_hz: float | None = None,
    phase_ref: bool = True,
) -> str:
    """Overlay Y_dev,++; all bins drawn alike; optional ``trust_report_path`` for low-trust list."""
    apply_thesis_plot_style()
    f_lo, f_hi = float(f_min_hz), float(f_max_hz)
    series_list = list(series)
    f_phase_ref = _overlay_phase_ref_hz(phase_ref_hz) if phase_ref and len(series_list) > 1 else None
    report_entries: list[tuple[str, list[dict[str, float]]]] = []
    criterion = rf"$\kappa(Y)\leq${kappa_max:g}"

    if trust_side_report:
        fig = plt.figure(figsize=(10.8, 5.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.46], wspace=0.28, hspace=0.12)
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_ph = fig.add_subplot(gs[1, 0], sharex=ax_mag)
        gs_report = gs[:, 1]
    else:
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=OVERLAY_FIGSIZE_PP, sharex=True)

    f = None
    traces: list[tuple[np.ndarray, str, np.ndarray]] = []
    for i, (label, csv_path) in enumerate(series_list):
        df = _load_csv(csv_path)
        f = df["f_Hz"].to_numpy(dtype=float)
        y = _complex_col(df, "Ydev_m00_re", "Ydev_m00_im")
        band = _qa_mask(df, f, f_min_hz=f_lo, f_max_hz=f_hi, det_y_max=1e30)
        band &= np.isfinite(y.real) & np.isfinite(y.imag)
        rows, crit = collect_ydev_low_trust_points(
            df, f_min_hz=f_lo, f_max_hz=f_hi, kappa_max=float(kappa_max)
        )
        report_entries.append((label, rows))
        if i == 0:
            criterion = crit
        color = OVERLAY_SERIES_COLORS[i % len(OVERLAY_SERIES_COLORS)]
        traces.append((y, color, band))

    if f is not None and f_phase_ref is not None:
        ys = _overlay_phasors_common_ref([t[0] for t in traces], f, f_phase_ref)
        traces = [(ys[i], traces[i][1], traces[i][2]) for i in range(len(traces))]

    for y, color, band in traces:
        yy = np.where(band, y, np.nan + 1j * np.nan)
        _plot_overlay_trace(
            ax_mag,
            ax_ph,
            f,
            yy,
            color=color,
            phase_anchor_hz=f_phase_ref,
        )

    draw_freq_marker(ax_mag, mark_hz)
    draw_freq_marker(ax_ph, mark_hz)
    style_bode_panel_axes(ax_mag, ax_ph)
    ax_mag.set_ylabel(_matrix_mag_ylabel("Ydev"))
    ax_ph.set_ylabel("Phase (°)")
    apply_bode_mag_phase_axes(ax_mag, ax_ph, f_lo, f_hi, show_freq_axis=True)

    report_text = format_ydev_trust_report(
        report_entries,
        f_min_hz=f_lo,
        f_max_hz=f_hi,
        kappa_max=float(kappa_max),
        criterion=criterion,
    )
    if trust_side_report:
        _draw_trust_report_panel(fig, gs_report, report_text)
    finalize_overlay_figure(
        fig,
        title=matrix_bode_suptitle("Y_dev,++", phase_ref_hz=f_phase_ref),
        legend_handles=_overlay_series_legend_handles(series_list),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=OVERLAY_DPI)
    plt.close(fig)
    if trust_report_path:
        with open(trust_report_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)
    return report_text


def plot_nyquist_loop(
    csv_path: str,
    out_png: str,
    *,
    channel: str = "pp",
    label: str = "",
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    det_y_warn: float | None = None,
    layout: str = "nyquist_only",
) -> None:
    """
    Scalar loop Nyquist: L = Z_bus Y_dev for one matrix entry (``pp`` or ``qq``).

    ``layout="nyquist_only"`` (default): single large Nyquist plane, framed on the measured
    L(jω) locus (|L| is often ≪ 1 here, so a unit-circle plot hides the curve).

    ``layout="bode_nyquist"``: legacy 3-panel figure (|L|, |1+L| Bode + small Nyquist).
    """
    ch = (channel or "pp").strip().lower()
    if ch not in _NYQUIST_SCALAR_CHANNELS:
        raise ValueError(f"channel must be one of {sorted(_NYQUIST_SCALAR_CHANNELS)}")
    cfg = _NYQUIST_SCALAR_CHANNELS[ch]
    loop_tex = cfg["tex"]

    df = _load_csv(csv_path)
    f = df["f_Hz"].to_numpy(dtype=float)
    zb = _complex_col(df, cfg["z_re"], cfg["z_im"])
    yd = _complex_col(df, cfg["y_re"], cfg["y_im"])
    band = _qa_mask(df, f, f_min_hz=f_min_hz, f_max_hz=f_max_hz, det_y_max=1e30)
    loop_raw = zb * yd
    finite = band & np.isfinite(loop_raw.real) & np.isfinite(loop_raw.imag)
    good, weak, qa_tag = _ydev_trust_masks(
        df, finite, det_y_cap=det_y_warn if det_y_warn is not None else None
    )
    loop = np.where(finite, loop_raw, np.nan + 1j * np.nan)
    ret = 1.0 + loop
    f_good = f[good]
    f_weak = f[weak]
    hi_all = float(np.max(f[finite])) if np.any(finite) else float(f_max_hz)
    lo_all = float(np.min(f[finite])) if np.any(finite) else float(f_min_hz)

    def _plot_locus(
        ax_n: plt.Axes,
        mask: np.ndarray,
        *,
        lw: float,
        alpha: float,
        label: str | None,
        zorder: int,
    ) -> None:
        if not np.any(mask):
            return
        z = loop[mask]
        re_p = np.real(z)
        im_p = np.imag(z)
        ax_n.plot(
            re_p,
            im_p,
            lw=lw,
            color="C2",
            ls="-",
            marker="x",
            ms=2.5,
            mew=0.6,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        ax_n.plot(
            re_p,
            -im_p,
            lw=max(0.6, lw - 0.2),
            color="C2",
            ls="--",
            alpha=0.45 * alpha,
            zorder=zorder,
            label=r"$\omega<0$ mirror" if label else None,
        )

    def _draw_nyquist(ax_n: plt.Axes, *, compact_legend: bool) -> tuple[np.ndarray, np.ndarray]:
        ok = finite
        th = np.linspace(0.0, 2.0 * np.pi, 240)
        re_p = np.real(loop[good]) if np.any(good) else np.real(loop[ok])
        im_p = np.imag(loop[good]) if np.any(good) else np.imag(loop[ok])
        re_zoom = np.real(loop[ok])
        im_zoom = np.imag(loop[ok])
        l_max = float(np.nanmax(np.abs(loop[ok]))) if np.any(ok) else 0.0

        if np.any(weak):
            _plot_locus(
                ax_n,
                weak,
                lw=0.9,
                alpha=0.35,
                label=rf"ill-conditioned ($\kappa(Y)>${KAPPA_YDEV_MAX:g})",
                zorder=2,
            )
        if np.any(good):
            _plot_locus(
                ax_n,
                good,
                lw=1.4,
                alpha=1.0,
                label=r"$L(j\omega_k)$, $\kappa(Y)\leq50$",
                zorder=3,
            )
            ax_n.plot(re_p[0], im_p[0], "o", ms=3.5, color="C2", zorder=4)
            ax_n.plot(re_p[-1], im_p[-1], "s", ms=3.5, color="C2", zorder=4)

        ax_n.plot(-1.0, 0.0, "x", ms=11, color="k", mew=2, zorder=5, label=r"critical $(-1,0)$")

        if l_max >= 0.15:
            ax_n.plot(
                np.cos(th),
                np.sin(th),
                color="0.55",
                ls="--",
                lw=0.8,
                alpha=0.55,
                label=r"$|L|=1$",
            )

        ax_n.axhline(0.0, color="0.5", lw=0.6)
        ax_n.axvline(0.0, color="0.5", lw=0.6)
        ax_n.set_xlabel(r"Re$\{L\}$")
        ax_n.set_ylabel(r"Im$\{L\}$")
        ax_n.grid(True, alpha=0.3)

        # Frame must include critical point (-1, 0) and the measured locus.
        if np.any(ok):
            pts_re = np.concatenate((re_p, np.array([-1.0, 0.0])))
            pts_im = np.concatenate((im_p, np.array([0.0, 0.0])))
            r_lo = float(np.min(pts_re))
            r_hi = float(np.max(pts_re))
            i_lo = float(np.min(pts_im))
            i_hi = float(np.max(pts_im))
            pr = max(r_hi - r_lo, 1e-9)
            pi = max(i_hi - i_lo, 1e-9)
            half = 0.5 * max(pr, pi, 2.05) + 0.06
            cx = 0.5 * (r_lo + r_hi)
            cy = 0.5 * (i_lo + i_hi)
            # When |L| ≪ 1, inset zoom uses the locus bbox (not max|L|), else points look like one dot.
            if l_max < 0.15:
                half_x = 0.5 * (NYQUIST_RE_HI - NYQUIST_RE_LO)
                ax_n.set_xlim(NYQUIST_RE_LO, NYQUIST_RE_HI)
                ax_n.set_ylim(cy - half_x, cy + half_x)
            else:
                ax_n.set_xlim(cx - half, cx + half)
                ax_n.set_ylim(cy - half, cy + half)

            if l_max < 0.15:
                re_lo, re_hi = float(np.min(re_zoom)), float(np.max(re_zoom))
                # Include ω<0 mirror in y limits so dashed branch is not clipped.
                im_lo = float(min(np.min(im_zoom), -np.max(im_zoom)))
                im_hi = float(max(np.max(im_zoom), -np.min(im_zoom)))
                pr = max(re_hi - re_lo, 1e-12)
                pi = max(im_hi - im_lo, 1e-12)
                icx = 0.5 * (re_lo + re_hi)
                icy = 0.5 * (im_lo + im_hi)
                pad = 1.18
                half = max(0.5 * pr * pad, 0.5 * pi * pad) + max(2e-5, 0.05 * l_max)
                # Large inset below the locus; connector lines to the zoomed region at the origin.
                axins = inset_axes(
                    ax_n,
                    width="50%",
                    height="40%",
                    loc="lower center",
                    bbox_to_anchor=(0.0, 0.05, 1.0, 0.92),
                    bbox_transform=ax_n.transAxes,
                    borderpad=1.0,
                )
                if np.any(weak):
                    axins.plot(
                        np.real(loop[weak]),
                        np.imag(loop[weak]),
                        lw=0.8,
                        color="C2",
                        marker="x",
                        ms=1.2,
                        mew=0.4,
                        alpha=0.35,
                    )
                axins.plot(re_zoom, im_zoom, lw=1.0, color="C2", marker="x", ms=1.2, mew=0.4)
                axins.plot(
                    re_zoom,
                    -im_zoom,
                    lw=1.0,
                    color="C2",
                    ls="--",
                    alpha=0.65,
                )
                axins.axhline(0.0, color="0.5", lw=0.5)
                axins.axvline(0.0, color="0.5", lw=0.5)
                axins.set_xlim(icx - half, icx + half)
                axins.set_ylim(icy - half, icy + half)
                axins.set_aspect("equal", adjustable="box")
                axins.grid(True, alpha=0.3)
                axins.set_title(r"$L$ zoom", fontsize=8, pad=3)
                format_small_locus_inset_axes(axins)
                mark_inset(
                    ax_n,
                    axins,
                    loc1=2,
                    loc2=1,
                    fc="none",
                    ec="0.45",
                    lw=0.9,
                    ls="--",
                    alpha=0.85,
                    zorder=1,
                )
        else:
            ax_n.set_xlim(-1.2, 0.2)
            ax_n.set_ylim(-0.6, 0.6)

        ax_n.set_aspect("equal", adjustable="box")
        leg_loc = "upper left" if (compact_legend and np.any(ok) and l_max < 0.15) else "upper right"
        ax_n.legend(loc=leg_loc, fontsize=8 if compact_legend else 7, framealpha=0.92)
        return re_p, im_p

    mode = (layout or "nyquist_only").strip().lower()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)

    if mode in ("nyquist", "nyquist_only", "nyquist-only"):
        fig, ax_n = plt.subplots(1, 1, figsize=(7.2, 6.8))
        _draw_nyquist(ax_n, compact_legend=True)
        n_weak = int(np.sum(weak))
        title = (
            f"Nyquist: $L={loop_tex}$ [{lo_all:g}, {hi_all:g}] Hz — "
            f"{int(np.sum(good))} trusted bins ({qa_tag})"
        )
        if n_weak:
            title += f", {n_weak} ill-conditioned ($\\kappa>{KAPPA_YDEV_MAX:g}$)"
        if label:
            title += f" — {label}"
        fig.suptitle(title, fontsize=10, y=0.98)
        fig.subplots_adjust(bottom=0.16, top=0.90, left=0.10, right=0.98)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.38)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[0, 2])

    for ax, yy, ylab in (
        (ax_l, loop, "|L|"),
        (ax_r, ret, "|1+L|"),
    ):
        yy_g = np.where(good, yy, np.nan + 1j * np.nan)
        yy_w = np.where(weak, yy, np.nan + 1j * np.nan)
        semilogx_bode(ax, f, _mag_db_bode_plot(yy_w), alpha=0.4, color=BODE_COLOR)
        semilogx_bode(ax, f, _mag_db_bode_plot(yy_g), alpha=0.95, color=BODE_COLOR)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", alpha=0.3)
        format_log_freq_axis(ax, lo_all, hi_all)

    ax_l.set_title(f"|L| ({MAG_DB_UNIT})")
    ax_r.set_title(f"|1+L| ({MAG_DB_UNIT})")
    _draw_nyquist(ax_n, compact_legend=False)

    for ax in (ax_l, ax_r):
        draw_freq_marker(ax, mark_hz)

    title = f"Loop $L={loop_tex}$ [{lo_all:g}, {hi_all:g}] Hz — magnitude ({MAG_DB_UNIT})"
    if label:
        title += f" — {label}"
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_nyquist_loop_pp(
    csv_path: str,
    out_png: str,
    **kwargs,
) -> None:
    """Nyquist of sequence scalar loop L = Z_bus,++ Y_dev,++."""
    plot_nyquist_loop(csv_path, out_png, channel="pp", **kwargs)


def plot_nyquist_loop_qq(
    csv_path: str,
    out_png: str,
    **kwargs,
) -> None:
    """Nyquist of dq diagonal q-channel loop L = Z_bus,qq Y_dev,qq."""
    plot_nyquist_loop(csv_path, out_png, channel="qq", **kwargs)


def plot_loop_eigenloci_dq(
    csv_path: str,
    out_png: str,
    *,
    label: str = "",
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    det_y_warn: float | None = None,
) -> dict[str, float]:
    """
    MIMO loop eigenloci: λᵢ(jω) of L(jω) = Z_bus,dq(jω) Y_dev,dq(jω).

    dq-only MIMO view: eigenloci of L = Z_bus,dq Y_dev,dq. Use ``plot_nyquist_loop_pp`` /
    ``plot_nyquist_loop_qq`` for scalar ++ and qq Nyquist figures separately.

    For 2×2 loops, det(I + L) = 0 ⟺ some λᵢ = −1.

    Returns summary stats (max |λ|) for quick comparison.
    """
    df = _load_csv(csv_path)
    f = df["f_Hz"].to_numpy(dtype=float)
    z_bus = _mimo_2x2_from_csv(df, "Zbus")
    y_dev = _mimo_2x2_from_csv(df, "Ydev")

    band = _qa_mask(df, f, f_min_hz=f_min_hz, f_max_hz=f_max_hz, det_y_max=1e30)
    lam_a, lam_b = _loop_eigenvalues_tracked(z_bus, y_dev, band)
    finite = band & np.isfinite(lam_a.real) & np.isfinite(lam_b.real)
    good, weak, qa_tag = _ydev_trust_masks(
        df, finite, det_y_cap=det_y_warn if det_y_warn is not None else None
    )

    def _plot_eig_track(
        ax: plt.Axes,
        mask: np.ndarray,
        lam: np.ndarray,
        *,
        color: str,
        label: str,
        lw: float,
        alpha: float,
    ) -> None:
        if not np.any(mask):
            return
        re_p = np.real(lam[mask])
        im_p = np.imag(lam[mask])
        ax.plot(
            re_p,
            im_p,
            lw=lw,
            color=color,
            marker="x",
            ms=2.0,
            mew=0.5,
            alpha=alpha,
            label=label,
            zorder=3,
        )
        ax.plot(re_p, -im_p, lw=max(0.6, lw - 0.2), color=color, ls="--", alpha=0.45 * alpha, zorder=2)

    fig, ax_n = plt.subplots(1, 1, figsize=(7.2, 6.8))

    if np.any(weak):
        _plot_eig_track(
            ax_n,
            weak,
            lam_a,
            color="C0",
            label=rf"$\lambda_a$, ill-conditioned ($\kappa>{KAPPA_YDEV_MAX:g}$)",
            lw=0.9,
            alpha=0.35,
        )
        _plot_eig_track(
            ax_n,
            weak,
            lam_b,
            color="C1",
            label=None,
            lw=0.9,
            alpha=0.35,
        )
    if np.any(good):
        _plot_eig_track(
            ax_n,
            good,
            lam_a,
            color="C0",
            label=r"$\lambda_a$ ($|\lambda_a|\leq|\lambda_b|$)",
            lw=1.3,
            alpha=1.0,
        )
        _plot_eig_track(
            ax_n,
            good,
            lam_b,
            color="C1",
            label=r"$\lambda_b$",
            lw=1.3,
            alpha=1.0,
        )

    ax_n.plot(-1.0, 0.0, "x", ms=11, color="k", mew=2, zorder=5, label=r"critical $(-1,0)$")
    ax_n.axhline(0.0, color="0.5", lw=0.6)
    ax_n.axvline(0.0, color="0.5", lw=0.6)
    ax_n.set_xlabel(r"Re$\{\lambda\}$")
    ax_n.set_ylabel(r"Im$\{\lambda\}$")
    ax_n.grid(True, alpha=0.3)

    ok = good
    lam_max = float(np.nanmax(np.maximum(np.abs(lam_a[ok]), np.abs(lam_b[ok])))) if np.any(ok) else 0.0
    if np.any(ok):
        half_x = 0.5 * (NYQUIST_RE_HI - NYQUIST_RE_LO)
        cy = float(np.nanmean(np.concatenate((np.imag(lam_a[ok]), np.imag(lam_b[ok])))))
        ax_n.set_xlim(NYQUIST_RE_LO, NYQUIST_RE_HI)
        ax_n.set_ylim(cy - half_x, cy + half_x)
        zoom_re = np.concatenate((np.real(lam_a[ok]), np.real(lam_b[ok])))
        zoom_im = np.concatenate((np.imag(lam_a[ok]), np.imag(lam_b[ok])))
        axins = _origin_zoom_inset_axes(ax_n, zoom_re, zoom_im, l_max=lam_max)
        if axins is not None:
            if np.any(weak):
                _plot_eig_track(axins, weak, lam_a, color="C0", label="", lw=0.8, alpha=0.35)
                _plot_eig_track(axins, weak, lam_b, color="C1", label="", lw=0.8, alpha=0.35)
            _plot_eig_track(
                axins,
                good,
                lam_a,
                color="C0",
                label="",
                lw=1.0,
                alpha=1.0,
            )
            _plot_eig_track(
                axins,
                good,
                lam_b,
                color="C1",
                label="",
                lw=1.0,
                alpha=1.0,
            )
    else:
        ax_n.set_xlim(-1.2, 0.2)
        ax_n.set_ylim(-0.6, 0.6)
    ax_n.set_aspect("equal", adjustable="box")
    ax_n.legend(loc="upper left", fontsize=7, framealpha=0.92)

    f_hi = float(np.nanmax(f[ok])) if np.any(ok) else float(f_max_hz)
    f_lo = float(np.nanmin(f[ok])) if np.any(ok) else float(f_min_hz)
    title = (
        f"dq eigenloci: $\\lambda_i(Z_{{bus,dq}}Y_{{dev,dq}})$ [{f_lo:g}, {f_hi:g}] Hz — "
        f"{int(np.sum(good))} trusted ({qa_tag}); max $|\\lambda|={lam_max:.4g}$"
    )
    if label:
        title += f" — {label}"
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.10, right=0.98)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "max_abs_lambda": lam_max,
        "n_good": int(np.sum(good)),
    }


def plot_loop_eigenloci_mimo(*args, **kwargs) -> dict[str, float]:
    """Backward-compatible alias for :func:`plot_loop_eigenloci_dq`."""
    return plot_loop_eigenloci_dq(*args, **kwargs)


_DQ_MATRIX_ELEMS: tuple[tuple[str, str], ...] = (
    ("dd", "dd"),
    ("dq", "dq"),
    ("qd", "qd"),
    ("qq", "qq"),
)


def plot_dq_matrix_overlay(
    series: Iterable[tuple[str, str]],
    out_png: str,
    *,
    matrix_prefix: str,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    trust_ydev_kappa: bool = False,
    kappa_max: float = KAPPA_YDEV_MAX,
    high_kappa_style: str = "fade",
    trust_side_report: bool = False,
    trust_report_path: str | None = None,
    phase_ref_hz: float | None = None,
    phase_ref: bool = True,
    title: str | None = None,
) -> str:
    """
    Overlay several cases on a 2×2 dq Bode grid (dd / dq / qd / qq).

    ``matrix_prefix`` is ``Zbus`` or ``Ydev`` (CSV columns ``{prefix}_{elem}_re/im``).

    For ``Ydev``, every bin is drawn with the same style when ``trust_ydev_kappa`` is off.
    Use ``trust_report_path`` to write low-trust bins to a text file (optional ``trust_side_report``).

    Legacy ``trust_ydev_kappa=True`` fades or dashes high-κ bins instead of using the panel.
    """
    apply_thesis_plot_style()
    prefix = str(matrix_prefix).strip()
    if prefix not in ("Zbus", "Ydev"):
        raise ValueError("matrix_prefix must be 'Zbus' or 'Ydev'")

    series_list = list(series)
    if len(series_list) < 1:
        raise ValueError("Need at least one (label, csv_path) series")

    use_side = bool(trust_side_report) and prefix == "Ydev"
    fig = plt.figure(figsize=(15.2, 7.8) if use_side else OVERLAY_FIGSIZE_DQ)
    if use_side:
        root = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40], wspace=0.22)
        outer = root[0].subgridspec(
            2, 2, wspace=OVERLAY_GRID_WSPACE, hspace=OVERLAY_GRID_HSPACE
        )
        gs_report = root[1]
    else:
        outer = fig.add_gridspec(2, 2, wspace=OVERLAY_GRID_WSPACE, hspace=OVERLAY_GRID_HSPACE)
        gs_report = None
    layout = [(0, 0), (0, 1), (1, 0), (1, 1)]

    f_lo = float(f_min_hz)
    f_hi = float(f_max_hz)
    f_phase_ref = _overlay_phase_ref_hz(phase_ref_hz) if phase_ref else None
    use_common_phase = f_phase_ref is not None and len(series_list) > 1

    report_entries: list[tuple[str, list[dict[str, float]]]] = []
    criterion = rf"$\kappa(Y)\leq${kappa_max:g}"
    need_report = prefix == "Ydev" and (use_side or trust_report_path)
    for i_s, (label, csv_path) in enumerate(series_list):
        if need_report:
            df_rep = _load_csv(csv_path)
            rows, crit = collect_ydev_low_trust_points(
                df_rep, f_min_hz=f_lo, f_max_hz=f_hi, kappa_max=float(kappa_max)
            )
            report_entries.append((label, rows))
            if i_s == 0:
                criterion = crit

    for (elem, elem_tag), (r, c) in zip(_DQ_MATRIX_ELEMS, layout):
        sub = outer[r, c].subgridspec(
            2, 1, height_ratios=[1.05, 1.0], hspace=OVERLAY_PANEL_MAGPH_HSPACE
        )
        ax_mag = fig.add_subplot(sub[0, 0])
        ax_ph = fig.add_subplot(sub[1, 0], sharex=ax_mag)
        panel_label = dq_panel_title(prefix, elem_tag)

        f = None
        panel_traces: list[tuple[np.ndarray, str, np.ndarray, np.ndarray, str, np.ndarray]] = []
        for i, (label, csv_path) in enumerate(series_list):
            df = _load_csv(csv_path)
            f = df["f_Hz"].to_numpy(dtype=float)
            z = _complex_col(df, f"{prefix}_{elem}_re", f"{prefix}_{elem}_im")
            band = np.isfinite(f) & (f >= f_lo) & (f <= f_hi) & np.isfinite(z.real) & np.isfinite(z.imag)
            plot_mask = band
            if prefix == "Ydev":
                band = _qa_mask(df, f, f_min_hz=f_lo, f_max_hz=f_hi, det_y_max=1e30) & band
            if trust_ydev_kappa and prefix == "Ydev" and not use_side:
                good, weak, _ = _ydev_trust_masks(df, band, kappa_max=float(kappa_max))
            else:
                good, weak = plot_mask if prefix == "Zbus" else band, np.zeros_like(band, dtype=bool)

            color = OVERLAY_SERIES_COLORS[i % len(OVERLAY_SERIES_COLORS)]
            hstyle = str(high_kappa_style).strip().lower()
            plot_mask_use = band if prefix == "Ydev" else plot_mask
            panel_traces.append((z, color, good, weak, hstyle, plot_mask_use))

        if f is None:
            continue
        z_rot = [t[0] for t in panel_traces]
        if use_common_phase:
            z_rot = _overlay_phasors_common_ref(z_rot, f, float(f_phase_ref))

        for i, (z, color, good, weak, hstyle, plot_mask_use) in enumerate(panel_traces):
            if use_common_phase:
                z = z_rot[i]
            phase_anchor = f_phase_ref if use_common_phase else None
            if trust_ydev_kappa and prefix == "Ydev" and not use_side:
                if np.any(good):
                    _plot_overlay_trace(
                        ax_mag,
                        ax_ph,
                        f,
                        np.where(good, z, np.nan + 1j * np.nan),
                        color=color,
                        phase_anchor_hz=phase_anchor,
                    )
                if np.any(weak) and hstyle != "omit":
                    _plot_overlay_trace(
                        ax_mag,
                        ax_ph,
                        f,
                        np.where(weak, z, np.nan + 1j * np.nan),
                        color=color,
                        phase_anchor_hz=phase_anchor,
                    )
            else:
                _plot_overlay_trace(
                    ax_mag,
                    ax_ph,
                    f,
                    np.where(plot_mask_use, z, np.nan + 1j * np.nan),
                    color=color,
                    phase_anchor_hz=phase_anchor,
                )

        ax_mag.set_title(panel_label, pad=3)
        if c == 0:
            ax_mag.set_ylabel(_matrix_mag_ylabel(prefix))
            ax_ph.set_ylabel("Phase (°)")
        else:
            ax_mag.set_ylabel("")
        draw_freq_marker(ax_mag, mark_hz)
        draw_freq_marker(ax_ph, mark_hz)
        style_bode_panel_axes(ax_mag, ax_ph)
        apply_bode_mag_phase_axes(ax_mag, ax_ph, f_lo, f_hi, show_freq_axis=(r == 1))

    matrix_name = "Z_bus (dq)" if prefix == "Zbus" else "Y_dev (dq)"
    report_text = ""
    if need_report and report_entries:
        report_text = format_ydev_trust_report(
            report_entries,
            f_min_hz=f_lo,
            f_max_hz=f_hi,
            kappa_max=float(kappa_max),
            criterion=criterion,
        )
        if use_side:
            _draw_trust_report_panel(fig, gs_report, report_text)
    finalize_overlay_figure(
        fig,
        title=title
        or matrix_bode_suptitle(
            matrix_name, phase_ref_hz=f_phase_ref if use_common_phase else None
        ),
        legend_handles=_overlay_series_legend_handles(series_list),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=OVERLAY_DPI)
    plt.close(fig)
    if trust_report_path and report_text:
        with open(trust_report_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)
    return report_text


def plot_ydev_dq_overlay(
    series: Iterable[tuple[str, str]],
    out_png: str,
    *,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    trust_side_report: bool = False,
    trust_report_path: str | None = None,
    phase_ref_hz: float | None = None,
    phase_ref: bool = True,
    title: str | None = None,
) -> str:
    """Overlay Y_dev dq; uniform trace + optional trust report text file."""
    return plot_dq_matrix_overlay(
        series,
        out_png,
        matrix_prefix="Ydev",
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        mark_hz=mark_hz,
        trust_ydev_kappa=False,
        trust_side_report=trust_side_report,
        trust_report_path=trust_report_path,
        phase_ref_hz=phase_ref_hz,
        phase_ref=phase_ref,
        title=title,
    )


def plot_zbus_dq_overlay(
    series: Iterable[tuple[str, str]],
    out_png: str,
    *,
    f_min_hz: float = DEFAULT_F_MIN_HZ,
    f_max_hz: float = PLOT_F_MAX_HZ,
    mark_hz: float | None = None,
    phase_ref_hz: float | None = None,
    phase_ref: bool = True,
    title: str | None = None,
) -> None:
    """Overlay Z_bus dq elements for several cases."""
    plot_dq_matrix_overlay(
        series,
        out_png,
        matrix_prefix="Zbus",
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        mark_hz=mark_hz,
        trust_ydev_kappa=False,
        phase_ref_hz=phase_ref_hz,
        phase_ref=phase_ref,
        title=title,
    )
