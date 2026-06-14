"""Shared Bode axis formatting for thesis impedance figures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, ScalarFormatter

MAG_DB_UNIT = "dB re 1 pu"
PHASE_UNIT = "deg, unwrap"

# Overlay / comparison figures
OVERLAY_DPI = 200
OVERLAY_FIGSIZE_DQ = (11.4, 8.6)
OVERLAY_FIGSIZE_PP = (8.2, 5.8)
OVERLAY_GRID_WSPACE = 0.14
OVERLAY_GRID_HSPACE = 0.20
OVERLAY_PANEL_MAGPH_HSPACE = 0.08
OVERLAY_PHASE_REF_HZ = 0.1

# Muted thesis palette (line + dot markers).
BODE_COLOR = "#4a6a85"
BODE_COLORS = ("#4a6a85", "#8f6b52", "#5a8f6e", "#7a6b8f", "#6b8faf", "#a67c52")
# Baseline vs coupled-model overlays (muted terracotta + sage green)
OVERLAY_SERIES_COLORS = ("#a67c52", "#5f7d6a")
BODE_MARKER = "o"
BODE_MS = 2.5
BODE_LW = 0.95
BODE_ALPHA = 0.9
BODE_MEC = "white"
BODE_MEW = 0.3


def mag_db_ylabel(short: str) -> str:
    """Short magnitude axis label (units go in suptitle)."""
    return short


def phase_ylabel(short: str = "Phase") -> str:
    return short


def apply_thesis_plot_style() -> None:
    """Consistent sans-serif styling for publication figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.05,
            "savefig.dpi": OVERLAY_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def dq_panel_title(matrix_prefix: str, elem_tag: str) -> str:
    sym = "Z" if str(matrix_prefix).strip() == "Zbus" else "Y"
    return rf"${sym}_{{{elem_tag}}}$"


def matrix_bode_suptitle(
    matrix_name: str,
    *,
    extra: str = "",
    phase_ref_hz: float | None = None,
) -> str:
    phase_desc = PHASE_UNIT
    if phase_ref_hz is not None:
        phase_desc = f"{PHASE_UNIT}, ref. {float(phase_ref_hz):g} Hz"
    base = f"{matrix_name} — magnitude in {MAG_DB_UNIT}, phase in {phase_desc}"
    return f"{base} — {extra}" if extra else base


def format_log_freq_axis(ax, f_min_hz: float, f_max_hz: float) -> None:
    """Log-spaced frequency axis with decade labels ($10^{-1}$, $10^0$, …)."""
    lo = float(f_min_hz)
    hi = float(f_max_hz)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo):
        return
    ax.set_xlim(lo, hi)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.xaxis.get_offset_text().set_visible(False)


def format_phase_y_axis(ax: Axes, *, n_ticks: int = 5) -> None:
    """Phase in degrees without matplotlib '+1.79e2' offset box (common near ±180°)."""
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_visible(False)


def format_mag_y_axis(ax: Axes, *, n_ticks: int = 5) -> None:
    """Magnitude (dB) ticks without offset multiplier."""
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_visible(False)


def apply_bode_mag_phase_axes(
    ax_mag: Axes,
    ax_ph: Axes,
    f_min_hz: float,
    f_max_hz: float,
    *,
    show_freq_axis: bool = True,
) -> None:
    """
    Log $f$ axis on both panels; $10^n$ Hz labels only on the phase panel when
    ``show_freq_axis`` (matrix Bode: bottom row only).
    """
    for ax in (ax_mag, ax_ph):
        format_log_freq_axis(ax, f_min_hz, f_max_hz)
    ax_mag.tick_params(axis="x", which="both", labelbottom=False)
    plt.setp(ax_mag.get_xticklabels(), visible=False)
    format_mag_y_axis(ax_mag)
    format_phase_y_axis(ax_ph)
    if show_freq_axis:
        ax_ph.set_xlabel("Frequency (Hz)")
    else:
        ax_ph.tick_params(axis="x", which="both", labelbottom=False)
        plt.setp(ax_ph.get_xticklabels(), visible=False)
        ax_ph.set_xlabel("")


def format_small_locus_inset_axes(ax: Axes, *, labelsize: float = 7, n_ticks: int = 4) -> None:
    """Compact tick labels for Nyquist/eigenloci zoom insets (sparse sci notation)."""
    from matplotlib import ticker as mticker

    ax.tick_params(axis="both", which="major", labelsize=labelsize, length=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(n_ticks))
    sci = mticker.FuncFormatter(lambda v, _: f"{v:.1e}")
    sci_x = mticker.FuncFormatter(
        lambda v, pos: "" if pos is not None and int(pos) % 2 else f"{v:.1e}"
    )
    ax.xaxis.set_major_formatter(sci_x)
    ax.yaxis.set_major_formatter(sci)
    ax.xaxis.offsetText.set_visible(False)
    ax.yaxis.offsetText.set_visible(False)


def semilogx_bode(
    ax: Axes,
    f: np.ndarray,
    y: np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    lw: float | None = None,
    alpha: float | None = None,
    ls: str = "-",
    marker: str | None = None,
    ms: float | None = None,
    zorder: int = 2,
    **kwargs: Any,
) -> None:
    """Log-$f$ Bode trace with dot markers at every plotted frequency bin."""
    ax.plot(
        f,
        y,
        color=color or BODE_COLOR,
        lw=lw if lw is not None else BODE_LW,
        ls=ls,
        marker=marker or BODE_MARKER,
        ms=ms if ms is not None else BODE_MS,
        mew=BODE_MEW,
        mec=BODE_MEC,
        alpha=alpha if alpha is not None else BODE_ALPHA,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.set_xscale("log")


def apply_matrix_bode_layout(fig, suptitle: str) -> None:
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.98, hspace=0.32, wspace=0.28)
    fig.suptitle(suptitle, fontsize=11, y=0.97)


def draw_freq_marker(
    ax,
    mark_hz: float | None,
    *,
    alpha: float = 0.45,
    lw: float = 0.75,
    color: str = "0.45",
    linestyle: str | tuple = (0, (4, 3)),
    ls: str | tuple | None = None,
) -> None:
    """Optional vertical guide at a reference frequency (omit when ``mark_hz`` is None)."""
    if mark_hz is None:
        return
    fm = float(mark_hz)
    if not np.isfinite(fm) or fm <= 0.0:
        return
    ls_plot = ls if ls is not None else linestyle
    ax.axvline(fm, color=color, linestyle=ls_plot, alpha=alpha, lw=lw, zorder=1)


def style_bode_panel_axes(ax_mag: Axes, ax_ph: Axes) -> None:
    """Grid, ticks, and spines for one magnitude + phase panel pair."""
    for ax in (ax_mag, ax_ph):
        ax.grid(True, which="major", color="0.88", linewidth=0.55, alpha=1.0)
        ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=1.0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", direction="out", length=3.5, width=0.6, pad=3)
        ax.tick_params(axis="both", which="minor", length=2, width=0.45)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.35")


def finalize_overlay_figure(
    fig,
    *,
    title: str,
    legend_handles: list | None = None,
    top: float | None = None,
) -> None:
    """Title, legend, and margins for baseline vs coupled comparison figures."""
    panel_top = 0.905 if top is None else float(top)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.972),
            ncol=len(legend_handles),
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            facecolor="white",
            framealpha=1.0,
            borderpad=0.45,
            handlelength=2.4,
            handletextpad=0.6,
            columnspacing=1.8,
        )
    fig.suptitle(
        title,
        fontsize=11,
        y=1.02 if legend_handles else 0.98,
        fontweight="normal",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=panel_top,
        bottom=0.08,
        hspace=OVERLAY_GRID_HSPACE,
        wspace=OVERLAY_GRID_WSPACE,
    )
