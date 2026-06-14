"""Thesis time-domain figure styling (aligned with eigenvalue + impedance plots)."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

# Match casestudies/impedance_stability/plots/bode_axes.py overlay palette
COLOR_BASELINE = "#a67c52"  # muted terracotta — baseline (simplified) model
COLOR_COUPLED = "#5f7d6a"   # muted sage — coupled (OpenFAST) model
COLOR_BASELINE_ALT = "#c9a88a"
COLOR_COUPLED_ALT = "#8aa399"
COLOR_WIND = "#4a6a85"
COLOR_REF = "#6b8faf"

THESIS_DPI = 200
THESIS_FIGSIZE = (6.4, 3.6)

# Line styles: solid for measured/actual; dashed only for references/setpoints
LS_ACTUAL = "-"
LS_REF = "--"


def apply_thesis_td_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.05,
            "savefig.dpi": THESIS_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def style_time_axis(ax: Axes, *, xlabel: str = "Time (s)") -> None:
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color="0.88", linewidth=0.55, alpha=1.0)
    ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", direction="out", length=3.5, width=0.6, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.35")


def plain_y_axis(ax: Axes) -> None:
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _round_axis_limits(lo: float, hi: float, *, n_ticks: int = 5) -> tuple[float, float]:
    """Expand to readable major-tick boundaries."""
    span = hi - lo
    if span <= 0:
        mag = max(abs(lo), abs(hi), 1.0)
        delta = 0.05 * mag if mag > 0 else 0.1
        return lo - delta, hi + delta
    raw_step = span / max(n_ticks - 1, 1)
    exp = np.floor(np.log10(raw_step))
    frac = raw_step / (10.0**exp) if exp > -20 else raw_step
    if frac <= 1.0:
        step = 10.0**exp
    elif frac <= 2.0:
        step = 2.0 * 10.0**exp
    elif frac <= 5.0:
        step = 5.0 * 10.0**exp
    else:
        step = 10.0 * 10.0**exp
    lo_n = np.floor(lo / step) * step
    hi_n = np.ceil(hi / step) * step
    if hi_n <= lo_n:
        hi_n = lo_n + step
    return float(lo_n), float(hi_n)


def ylim_style_from_ylabel(ylabel: str) -> dict:
    """Heuristic axis padding keyed on axis label text."""
    yl = ylabel.lower()
    if "wind" in yl and "m/s" in yl:
        return {"min_span": 1.0}
    if "pitch" in yl:
        return {"min_span": 2.0}
    if "speed" in yl and "p.u." in yl:
        return {"min_span": 0.1}
    if "voltage" in yl:
        return {"tight_voltage": True}
    if "power" in yl or "active" in yl or "reactive" in yl:
        return {"floor_nonneg": True}
    if "current" in yl and "p.u." in yl:
        return {"floor_nonneg": True}
    if "torque" in yl and ("kn" in yl or "n·m" in yl):
        return {"pad_frac": 0.02}
    return {}


def ylim_nice(
    ax: Axes,
    *series: Iterable[float],
    pad_frac: float = 0.08,
    pad_abs: float | None = None,
    floor: float | None = None,
    ceil: float | None = None,
    floor_nonneg: bool = False,
    min_span: float | None = None,
    tight_voltage: bool = False,
) -> None:
    y = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series if s is not None])
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = ymax - ymin
    center = 0.5 * (ymin + ymax)

    if span <= max(1e-15, 1e-9 * max(abs(ymin), abs(ymax), 1.0)):
        half = 0.5 * (min_span or max(0.02 * max(abs(center), 1.0), 0.05))
        lo, hi = center - half, center + half
    else:
        pad = pad_frac * span if span > 0 else (pad_abs or 1e-3)
        lo, hi = ymin - pad, ymax + pad
        if min_span is not None and (hi - lo) < min_span:
            half = 0.5 * min_span
            lo, hi = center - half, center + half
        lo, hi = _round_axis_limits(lo, hi)

    if tight_voltage or (ymin > 0.85 and ymax < 1.15 and span < 0.05):
        half = max(0.5 * span + 2e-4, 0.006)
        lo, hi = center - half, center + half
        lo, hi = _round_axis_limits(lo, hi)

    if floor_nonneg and ymin >= -1e-9:
        lo = max(0.0, lo)
    elif floor is not None:
        lo = max(floor, lo)
    if ceil is not None:
        hi = min(ceil, hi) if hi > lo else ceil
    if hi > lo:
        ax.set_ylim(lo, hi)


def xlim_time(ax: Axes, t, *, pad_frac: float = 0.01) -> None:
    t = np.asarray(t, dtype=float).ravel()
    t = t[np.isfinite(t)]
    if t.size == 0:
        return
    tmin, tmax = float(np.min(t)), float(np.max(t))
    pad = pad_frac * (tmax - tmin) if tmax > tmin else 0.5
    ax.set_xlim(max(0.0, tmin - pad), tmax + pad)


# Short legend labels for thesis figures (model noted in caption, not legend).
_LEGEND_SHORT: dict[str, str] = {
    "P_inf_sys_pu": r"$P_{\mathrm{inf}}$",
    "Q_inf_sys_pu": r"$Q_{\mathrm{inf}}$",
    "P_e_sys_pu": r"$P_e$",
    "P_ref_sys_pu": r"$P_{\mathrm{ref}}$",
    "P_aero_sys_pu": r"$P_{\mathrm{aero}}$",
    "P_uic_bus_actual_sys_pu": r"$P_t$",
    "P_uic_bus_ref_sys_pu": r"$P_{\mathrm{ref},t}$",
    "Q_uic_bus_actual_sys_pu": r"$Q_t$",
    "Q_uic_bus_ref_sys_pu": r"$Q_{\mathrm{ref},t}$",
    "omega_m_pu": r"$\omega_m$",
    "omega_e_pu": r"$\omega_e$",
    "omega_m": r"$\omega_m$",
    "omega_e": r"$\omega_e$",
    "v_bus_pu": r"$|V_t|$",
    "vi_mag": r"$|v_i|$",
    "vi_x": r"$v_{i,x}$",
    "vi_y": r"$v_{i,y}$",
    "pitch_deg": r"$\beta$",
    "wind_speed_mps": r"$v_w$",
    "theta_s": r"$\theta_s$",
    "i_a_mag": r"$|i_a|$",
    "i_a_angle": r"$\angle i_a$",
}


def legend_label(key: str | None = None, *, text: str | None = None) -> str:
    """Short legend for single-model plots (model named in caption)."""
    if text is not None:
        return text
    if key is None:
        return ""
    return _LEGEND_SHORT.get(key, key)


# Thesis symbol for each OpenFAST output / input (legend: symbol (OpenFASTName)).
_FMU_OPENFAST_SYMBOL: dict[str, str] = {
    "GenSpeed": r"$\omega_e$",
    "RotSpeed": r"$\omega_m$",
    "BldPitch1": r"$\beta$",
    "NacYaw": r"$\psi_{\mathrm{nac}}$",
    "Azimuth": r"$\psi_{\mathrm{az}}$",
    "LSSGagPxa": r"$\theta_{\mathrm{LSS}}$",
    "GenAccel": r"$\alpha_{\mathrm{gen}}$",
    "YawBrTAxp": r"$\ddot{y}_{\mathrm{tow},x}$",
    "YawBrTAyp": r"$\ddot{y}_{\mathrm{tow},y}$",
    "RtAeroMxh": r"$M_{\mathrm{aero}}$",
    "HSShftTq": r"$T_{\mathrm{HSS}}$",
    "GenTq": r"$T_{\mathrm{gen}}$",
    "Wind1VelX": r"$v_w$",
    "RtVAvgxh": r"$\bar{v}_w$",
    "GenSpdOrTrq": r"$T_{\mathrm{e,cmd}}$",
    "RefGenSpd": r"$\omega_{e,\mathrm{ref}}$",
}


def fmu_legend_label(openfast_name: str) -> str:
    """Legend for OpenFAST signals: e.g. $\\omega_m$ (RotSpeed)."""
    sym = _FMU_OPENFAST_SYMBOL.get(openfast_name)
    if sym:
        return f"{sym} ({openfast_name})"
    return openfast_name


# Descriptive figure-title overrides for signals where the bare symbol is too
# terse to be self-explanatory (legend keeps the compact symbol via _FMU_OPENFAST_SYMBOL).
_FMU_OPENFAST_TITLE: dict[str, str] = {
    "YawBrTAxp": "Tower-top fore-aft acceleration",
    "YawBrTAyp": "Tower-top side-to-side acceleration",
}


def fmu_plot_title(openfast_name: str) -> str:
    """Figure title for a single OpenFAST signal (descriptive when available, else symbol)."""
    if openfast_name in _FMU_OPENFAST_TITLE:
        return _FMU_OPENFAST_TITLE[openfast_name]
    return _FMU_OPENFAST_SYMBOL.get(openfast_name, openfast_name)


def legend_compare(model: str, *, detail: str | None = None, ref: bool = False) -> str:
    """Legend for baseline vs coupled overlays — always identifies the model.

    *detail* adds a signal qualifier when the header alone is not enough
    (e.g. ``detail='wind speed'``, ``detail='hub-averaged'``).
    *ref=True* marks a reference/setpoint trace on the same axes.
    """
    tag = "Baseline" if model == "baseline" else "Coupled"
    if ref:
        return f"{tag}, ref"
    if detail:
        return f"{tag}, {detail}"
    return tag


def new_figure(title: str | None = None) -> tuple[Figure, Axes]:
    apply_thesis_td_style()
    fig, ax = plt.subplots(1, 1, figsize=THESIS_FIGSIZE)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    style_time_axis(ax)
    return fig, ax


def save_figure(fig: Figure, path: str, *, show: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=THESIS_DPI)
    if not show:
        plt.close(fig)
    return path


def clear_thesis_plot_dir(out_dir: str) -> int:
    """Remove existing PNGs in a thesis plot folder so stale figures are not left behind."""
    if not os.path.isdir(out_dir):
        return 0
    removed = 0
    for name in os.listdir(out_dir):
        if name.lower().endswith(".png"):
            os.remove(os.path.join(out_dir, name))
            removed += 1
    return removed


def plot_baseline(
    ax: Axes,
    t,
    y,
    *,
    label: str,
    ls: str = "-",
    lw: float = 1.1,
    alpha: float = 0.95,
    color: str | None = None,
    zorder: int = 2,
) -> None:
    ax.plot(
        t,
        y,
        ls=ls,
        label=label,
        color=color or COLOR_BASELINE,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )


def plot_coupled(
    ax: Axes,
    t,
    y,
    *,
    label: str,
    ls: str = LS_ACTUAL,
    lw: float = 1.1,
    alpha: float = 0.95,
    color: str | None = None,
    zorder: int = 2,
) -> None:
    ax.plot(
        t,
        y,
        ls=ls,
        label=label,
        color=color or COLOR_COUPLED,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )


def finalize_compare_legend(ax: Axes) -> None:
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.75", framealpha=1.0)


def save_compare_overlay(
    out_dir: str,
    stem: str,
    title: str,
    t,
    *,
    ylabel: str,
    baseline: tuple[np.ndarray, str] | None = None,
    coupled: tuple[np.ndarray, str] | None = None,
    extra: list[tuple[np.ndarray, str, str, str]] | None = None,
    ylim: tuple[float, float] | None = None,
    baseline_ls: str = LS_ACTUAL,
    coupled_ls: str = LS_ACTUAL,
    show: bool = False,
) -> str | None:
    """Save one baseline vs coupled overlay figure.

    *baseline* / *coupled*: ``(y, legend_text)`` — use ``legend_compare()`` for overlays.
    *extra*: ``(y, full_label, color, ls)`` — use ``LS_REF`` for reference traces.
    """
    if baseline is None and coupled is None and not extra:
        return None
    fig, ax = new_figure(title)

    if baseline is not None:
        plot_baseline(ax, t, baseline[0], label=baseline[1], ls=baseline_ls)
    if coupled is not None:
        plot_coupled(ax, t, coupled[0], label=coupled[1], ls=coupled_ls)
    for item in extra or []:
        y, lab, color, ls = item
        ax.plot(t, y, ls=ls, label=lab, color=color, linewidth=1.1, alpha=0.95)
    ax.set_ylabel(ylabel)
    plain_y_axis(ax)
    xlim_time(ax, t)
    series = []
    if baseline:
        series.append(baseline[0])
    if coupled:
        series.append(coupled[0])
    for item in extra or []:
        series.append(item[0])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ylim_nice(ax, *series, **ylim_style_from_ylabel(ylabel))
    finalize_compare_legend(ax)
    path = os.path.join(out_dir, f"{stem}.png")
    save_figure(fig, path, show=show)
    return path


def save_single_model_trace(
    out_dir: str,
    stem: str,
    title: str,
    t,
    y,
    *,
    ylabel: str,
    model: str = "baseline",
    label: str | None = None,
    var_name: str | None = None,
    ylim: tuple[float, float] | None = None,
    ls: str = "-",
    show: bool = False,
) -> str:
    fig, ax = new_figure(title)
    color = COLOR_BASELINE if model == "baseline" else COLOR_COUPLED
    leg = label or (legend_label(var_name) if var_name else "")
    ax.plot(t, y, color=color, ls=ls, linewidth=1.1, label=leg)
    ax.set_ylabel(ylabel)
    plain_y_axis(ax)
    xlim_time(ax, t)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ylim_nice(ax, y, **ylim_style_from_ylabel(ylabel))
    finalize_compare_legend(ax)
    path = os.path.join(out_dir, f"{stem}.png")
    save_figure(fig, path, show=show)
    return path


def save_single_model_multi(
    out_dir: str,
    stem: str,
    title: str,
    t,
    traces: list[tuple[np.ndarray, str, str | None]],
    *,
    ylabel: str,
    model: str = "baseline",
    ylim: tuple[float, float] | None = None,
    ylim_nice_kwargs: dict | None = None,
    show: bool = False,
) -> str:
    """*traces*: (y, legend_label, linestyle or None)."""
    fig, ax = new_figure(title)
    base = COLOR_BASELINE if model == "baseline" else COLOR_COUPLED
    alt = COLOR_BASELINE_ALT if model == "baseline" else COLOR_COUPLED_ALT
    colors = [base, alt, COLOR_REF, COLOR_WIND]
    for i, (y, lab, ls) in enumerate(traces):
        ax.plot(
            t,
            y,
            label=lab,
            color=colors[i % len(colors)],
            ls=ls or LS_ACTUAL,
            linewidth=1.1 if i == 0 else 1.05,
        )
    ax.set_ylabel(ylabel)
    plain_y_axis(ax)
    xlim_time(ax, t)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        kw = dict(ylim_style_from_ylabel(ylabel))
        if ylim_nice_kwargs:
            kw.update(ylim_nice_kwargs)
        ylim_nice(ax, *[tr[0] for tr in traces], **kw)
    finalize_compare_legend(ax)
    path = os.path.join(out_dir, f"{stem}.png")
    save_figure(fig, path, show=show)
    return path


# OpenFAST FMU outputs (modelDescription.xml).
_FMU_SKIP_COLS = frozenset({"Time", "Time_fmu", "RefGenSpd", "RotSpeed"})
_FMU_TORQUE_COLS = ("HSShftTq", "GenTq")
# All other FMU outputs: one PNG each; legend via fmu_legend_label().
_FMU_INDIVIDUAL: list[tuple[str, str]] = [
    ("GenSpeed", "Speed (rpm)"),
    # RotSpeed: also on coupled_omega_e_RotSpeed_pu with TOPS omega_e
    ("BldPitch1", "Pitch angle (deg)"),
    ("NacYaw", "Angle (deg)"),
    ("Azimuth", "Angle (deg)"),
    ("LSSGagPxa", "Angle (deg)"),
    ("GenAccel", "Acceleration (deg/s²)"),
    ("YawBrTAxp", "Acceleration (m/s²)"),
    ("YawBrTAyp", "Acceleration (m/s²)"),
    ("RtAeroMxh", "Moment (kN·m)"),
]
# Wrapper → OpenFAST input name for legend (column in drivetrain CSV).
_FMU_WRAPPER_OPENFAST: list[tuple[str, str, str]] = [
    ("GenSpdOrTrq_set_kNm", "GenSpdOrTrq", "Torque (kN·m)"),
]


def _fmu_series(df, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    y = np.asarray(df[col], dtype=float)
    if not np.any(np.isfinite(y)):
        return None
    return y


def save_fmu_outputs_thesis(
    out_dir: str,
    t,
    df_fmu,
    *,
    model: str = "coupled",
    show: bool = False,
) -> list[str]:
    """Save thesis-style PNGs for OpenFAST FMU outputs (one signal per figure, except wind and torques)."""
    if df_fmu is None or df_fmu.empty:
        return []
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, dtype=float)
    paths: list[str] = []
    plotted: set[str] = set()

    # Combined wind (only multi-signal wind plot).
    wind_tr: list[tuple[np.ndarray, str, str | None]] = []
    for col in ("Wind1VelX", "RtVAvgxh"):
        y = _fmu_series(df_fmu, col)
        if y is None:
            continue
        wind_tr.append((y, fmu_legend_label(col), None))
        plotted.add(col)
    if wind_tr:
        paths.append(
            save_single_model_multi(
                out_dir,
                "fmu_wind_mps",
                "Wind speed",
                t,
                wind_tr,
                ylabel="Wind speed (m/s)",
                model=model,
                show=show,
            )
        )

    # Shaft torques on one axes when both are present (HSS vs generator).
    tq_items: list[tuple[str, np.ndarray]] = []
    for col in _FMU_TORQUE_COLS:
        y = _fmu_series(df_fmu, col)
        if y is None:
            continue
        tq_items.append((col, y))
        plotted.add(col)
    if len(tq_items) == 1:
        col, y = tq_items[0]
        paths.append(
            save_single_model_trace(
                out_dir,
                f"fmu_{col}",
                fmu_plot_title(col),
                t,
                y,
                ylabel="Torque (kN·m)",
                model=model,
                label=fmu_legend_label(col),
                show=show,
            )
        )
    elif len(tq_items) >= 2:
        tq_tr = [(y, fmu_legend_label(col), None) for col, y in tq_items]
        paths.append(
            save_single_model_multi(
                out_dir,
                "fmu_torque_kNm",
                "Drivetrain torque",
                t,
                tq_tr,
                ylabel="Torque (kN·m)",
                model=model,
                ylim_nice_kwargs=ylim_style_from_ylabel("Torque (kN·m)"),
                show=show,
            )
        )

    for col, ylabel in _FMU_INDIVIDUAL:
        if col in plotted:
            continue
        y = _fmu_series(df_fmu, col)
        if y is None:
            continue
        paths.append(
            save_single_model_trace(
                out_dir,
                f"fmu_{col}",
                fmu_plot_title(col),
                t,
                y,
                ylabel=ylabel,
                model=model,
                label=fmu_legend_label(col),
                show=show,
            )
        )
        plotted.add(col)

    for col in df_fmu.columns:
        if col in _FMU_SKIP_COLS or col in plotted:
            continue
        y = _fmu_series(df_fmu, col)
        if y is None:
            continue
        paths.append(
            save_single_model_trace(
                out_dir,
                f"fmu_{col}",
                fmu_plot_title(col),
                t,
                y,
                ylabel=col,
                model=model,
                label=fmu_legend_label(col),
                show=show,
            )
        )
        plotted.add(col)

    return paths


def _tops_omega_e_pu(
    df,
    result_df=None,
    *,
    key: tuple[str, str] = ("FMUtoUICdrivetrain1", "omega_e"),
) -> np.ndarray | None:
    if result_df is not None and key in result_df.columns:
        y = np.asarray(result_df[key], dtype=float)
        return y if np.any(np.isfinite(y)) else None
    return _series(df, "omega_e_tops_pu")


def save_coupled_omega_e_rotspeed_pu(
    out_dir: str,
    t,
    df,
    *,
    result_df=None,
    df_fmu=None,
    omega_base_rpm: float | None = None,
    model: str = "coupled",
    show: bool = False,
) -> str | None:
    """TOPS electrical speed omega_e with OpenFAST RotSpeed on one axes (p.u.)."""
    omega_e = _tops_omega_e_pu(df, result_df)
    rot_pu = None
    if (
        df_fmu is not None
        and "RotSpeed" in df_fmu.columns
        and omega_base_rpm is not None
        and np.isfinite(omega_base_rpm)
        and omega_base_rpm > 0
    ):
        rot = _fmu_series(df_fmu, "RotSpeed")
        if rot is not None:
            rot_pu = rot / float(omega_base_rpm)
    if omega_e is None and rot_pu is None:
        return None
    traces: list[tuple[np.ndarray, str, str | None]] = []
    if omega_e is not None:
        traces.append((omega_e, r"$\omega_e$ (TOPS)", None))
    if rot_pu is not None:
        traces.append((rot_pu, fmu_legend_label("RotSpeed"), None))
    if len(traces) == 1:
        return save_single_model_trace(
            out_dir,
            _stem(model, "omega_e_RotSpeed_pu"),
            "Electrical and rotor speed",
            t,
            traces[0][0],
            ylabel=r"Speed (p.u., $\omega_{m,\mathrm{rated}}$ base)",
            model=model,
            label=traces[0][1],
            show=show,
        )
    return save_single_model_multi(
        out_dir,
        _stem(model, "omega_e_RotSpeed_pu"),
        "Electrical and rotor speed",
        t,
        traces,
        ylabel=r"Speed (p.u., $\omega_{m,\mathrm{rated}}$ base)",
        model=model,
        show=show,
    )


def save_fmu_wrapper_inputs_thesis(
    out_dir: str,
    t,
    df,
    *,
    model: str = "coupled",
    show: bool = False,
) -> list[str]:
    """Plot values written to the OpenFAST FMU (symbol + OpenFAST name in legend)."""
    paths: list[str] = []
    for csv_col, of_name, ylabel in _FMU_WRAPPER_OPENFAST:
        y = _series(df, csv_col)
        if y is None:
            continue
        paths.append(
            save_single_model_trace(
                out_dir,
                f"fmu_{of_name}",
                fmu_plot_title(of_name),
                t,
                y,
                ylabel=ylabel,
                model=model,
                label=fmu_legend_label(of_name),
                show=show,
            )
        )
    return paths


def _series(df, col: str) -> np.ndarray | None:
    """Return column as float array if present and has any finite values."""
    if col not in df.columns:
        return None
    y = np.asarray(df[col], dtype=float)
    if not np.any(np.isfinite(y)):
        return None
    return y


def _stem(model: str, name: str) -> str:
    return f"{'baseline' if model == 'baseline' else 'coupled'}_{name}"


def save_inf_bus_PQ_pu(
    out_dir: str,
    t,
    df,
    *,
    model: str = "baseline",
    show: bool = False,
) -> str | None:
    """Infinite-bus P and Q on one axes (same layout for baseline and coupled)."""
    p_inf, q_inf = _series(df, "P_inf_sys_pu"), _series(df, "Q_inf_sys_pu")
    if p_inf is None or q_inf is None:
        return None
    return save_single_model_multi(
        out_dir,
        _stem(model, "inf_bus_PQ_pu"),
        "Infinite-bus power",
        t,
        [
            (p_inf, legend_label("P_inf_sys_pu"), None),
            (q_inf, legend_label("Q_inf_sys_pu"), None),
        ],
        ylabel="Power (p.u., system base)",
        model=model,
        show=show,
    )


def save_uic_voltages_pu(
    out_dir: str,
    t,
    df,
    *,
    model: str = "baseline",
    show: bool = False,
) -> str | None:
    """Plot |v_i| and |V_t| (UIC internal and terminal magnitudes) on one axes."""
    traces: list[tuple[np.ndarray, str, str | None]] = []
    vi = _series(df, "vi_mag_pu")
    if vi is None:
        vi = _series(df, "vi_mag")
    if vi is not None:
        traces.append((vi, legend_label("vi_mag"), None))
    vt = _series(df, "v_bus_pu")
    if vt is not None:
        traces.append((vt, legend_label("v_bus_pu"), None))
    if not traces:
        return None
    return save_single_model_multi(
        out_dir,
        _stem(model, "voltages_pu"),
        "UIC terminal and internal voltages",
        t,
        traces,
        ylabel="Voltage magnitude (p.u.)",
        model=model,
        show=show,
    )


def save_uic_bus_PQ_pu(
    out_dir: str,
    t,
    df,
    *,
    model: str = "baseline",
    show: bool = False,
) -> str | None:
    """Plot UIC bus active and reactive power (actual and reference) together."""
    traces: list[tuple[np.ndarray, str, str | None]] = []
    for col, ls in (
        ("P_uic_bus_actual_sys_pu", None),
        ("P_uic_bus_ref_sys_pu", "--"),
        ("Q_uic_bus_actual_sys_pu", None),
        ("Q_uic_bus_ref_sys_pu", "--"),
    ):
        y = _series(df, col)
        if y is not None:
            traces.append((y, legend_label(col), ls))
    if not traces:
        return None
    return save_single_model_multi(
        out_dir,
        _stem(model, "uic_bus_PQ_pu"),
        "UIC terminal power",
        t,
        traces,
        ylabel="Power (p.u., system base)",
        model=model,
        show=show,
    )


def save_wt_electrical_power_pu(
    out_dir: str,
    t,
    df,
    *,
    model: str = "baseline",
    include_aero: bool = False,
    show: bool = False,
) -> str | None:
    """Plot P_aero (optional), P_e and P_ref on one axes."""
    traces: list[tuple[np.ndarray, str, str | None]] = []
    if include_aero:
        y = _series(df, "P_aero_sys_pu")
        if y is not None:
            traces.append((y, legend_label("P_aero_sys_pu"), None))
    for col, ls in (("P_e_sys_pu", None), ("P_ref_sys_pu", "--")):
        y = _series(df, col)
        if y is not None:
            traces.append((y, legend_label(col), ls))
    if not traces:
        return None
    title = "Wind-turbine power" if include_aero else "Electrical power"
    stem = "wt_power_pu" if include_aero else "Pe_pref_pu"
    return save_single_model_multi(
        out_dir,
        _stem(model, stem),
        title,
        t,
        traces,
        ylabel="Power (p.u., system base)",
        model=model,
        show=show,
    )


def save_baseline_thesis_plots(
    out_dir: str,
    t,
    df,
    *,
    clean_first: bool = True,
    show: bool = False,
) -> list[str]:
    """Thesis PNGs for the simplified wind-turbine (windturbine.py) simulation."""
    if clean_first:
        clear_thesis_plot_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, dtype=float)
    model = "baseline"
    paths: list[str] = []

    def _add(path: str | None) -> None:
        if path:
            paths.append(path)

    om_m, om_e = _series(df, "omega_m_pu"), _series(df, "omega_e_pu")
    if om_m is not None and om_e is not None:
        _add(
            save_single_model_multi(
                out_dir,
                _stem(model, "speeds_pu"),
                "Drivetrain speeds",
                t,
                [
                    (om_m, legend_label("omega_m_pu"), None),
                    (om_e, legend_label("omega_e_pu"), None),
                ],
                ylabel=r"Speed (p.u., $\omega_{m,\mathrm{rated}}$ base)",
                model=model,
                show=show,
            )
        )

    for col, stem, title, ylab, var in (
        ("pitch_deg", "pitch_deg", "Blade pitch angle", "Pitch angle (deg)", "pitch_deg"),
        ("wind_speed_mps", "wind_mps", "Wind speed", "Wind speed (m/s)", "wind_speed_mps"),
    ):
        y = _series(df, col)
        if y is not None:
            _add(
                save_single_model_trace(
                    out_dir,
                    _stem(model, stem),
                    title,
                    t,
                    y,
                    ylabel=ylab,
                    model=model,
                    var_name=var,
                    show=show,
                )
            )

    _add(save_uic_voltages_pu(out_dir, t, df, model=model, show=show))

    for col, stem, title, ylab, var in (
        (
            "i_a_mag_pu_uic",
            "current_mag_pu",
            "Armature current magnitude",
            "Current (p.u., UIC base)",
            "i_a_mag",
        ),
        (
            "i_a_angle_deg",
            "current_angle_deg",
            "Armature current angle",
            "Current angle (deg)",
            "i_a_angle",
        ),
    ):
        y = _series(df, col)
        if y is not None:
            _add(
                save_single_model_trace(
                    out_dir,
                    _stem(model, stem),
                    title,
                    t,
                    y,
                    ylabel=ylab,
                    model=model,
                    var_name=var,
                    show=show,
                )
            )

    y = _series(df, "T_mpt_wt_pu")
    if y is not None:
        _add(
            save_single_model_trace(
                out_dir,
                _stem(model, "T_mpt_pu"),
                "MPT mechanical torque",
                t,
                y,
                ylabel="Torque (p.u., mechanical)",
                model=model,
                label=r"$T_{\mathrm{MPT}}$",
                show=show,
            )
        )

    _add(save_wt_electrical_power_pu(out_dir, t, df, model=model, include_aero=True, show=show))
    _add(save_uic_bus_PQ_pu(out_dir, t, df, model=model, show=show))

    _add(save_inf_bus_PQ_pu(out_dir, t, df, model=model, show=show))

    return paths


def save_coupled_thesis_plots(
    out_dir: str,
    t,
    df,
    *,
    result_df=None,
    df_fmu=None,
    omega_base_rpm: float | None = None,
    clean_first: bool = True,
    show: bool = False,
) -> list[str]:
    """Thesis PNGs for the OpenFAST-coupled (FMU drivetrain) simulation."""
    if clean_first:
        clear_thesis_plot_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, dtype=float)
    model = "coupled"
    paths: list[str] = []

    def _add(path: str | None) -> None:
        if path:
            paths.append(path)

    _add(save_uic_voltages_pu(out_dir, t, df, model=model, show=show))

    _add(
        save_coupled_omega_e_rotspeed_pu(
            out_dir,
            t,
            df,
            result_df=result_df,
            df_fmu=df_fmu,
            omega_base_rpm=omega_base_rpm,
            model=model,
            show=show,
        )
    )

    if result_df is not None:
        th_key = ("FMUtoUICdrivetrain1", "theta_s")
        if th_key in result_df.columns:
            y = np.asarray(result_df[th_key], dtype=float)
            if np.any(np.isfinite(y)):
                _add(
                    save_single_model_trace(
                        out_dir,
                        _stem(model, "theta_s_pu"),
                        "Shaft twist",
                        t,
                        y,
                        ylabel=r"Shaft twist $\theta_s$ (p.u.)",
                        model=model,
                        var_name="theta_s",
                        show=show,
                    )
                )

    if df_fmu is not None and not df_fmu.empty:
        paths.extend(save_fmu_outputs_thesis(out_dir, t, df_fmu, model=model, show=show))

    paths.extend(save_fmu_wrapper_inputs_thesis(out_dir, t, df, model=model, show=show))

    _add(save_wt_electrical_power_pu(out_dir, t, df, model=model, include_aero=False, show=show))
    _add(save_uic_bus_PQ_pu(out_dir, t, df, model=model, show=show))

    for col, stem, title, ylab, var in (
        (
            "i_a_mag_pu_uic",
            "current_mag_pu",
            "Armature current magnitude",
            "Current (p.u., UIC base)",
            "i_a_mag",
        ),
        (
            "i_a_angle_deg",
            "current_angle_deg",
            "Armature current angle",
            "Current angle (deg)",
            "i_a_angle",
        ),
    ):
        y = _series(df, col)
        if y is not None:
            _add(
                save_single_model_trace(
                    out_dir,
                    _stem(model, stem),
                    title,
                    t,
                    y,
                    ylabel=ylab,
                    model=model,
                    var_name=var,
                    show=show,
                )
            )

    _add(save_inf_bus_PQ_pu(out_dir, t, df, model=model, show=show))

    return paths
