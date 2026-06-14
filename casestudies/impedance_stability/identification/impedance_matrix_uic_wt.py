import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _unwrap_phase_deg(zc: np.ndarray) -> np.ndarray:
    """Unwrap phase (deg) along contiguous finite samples only (do not bridge NaN gaps)."""
    ang = np.angle(np.asarray(zc, dtype=complex))
    out = np.full(ang.shape, np.nan, dtype=float)
    m = np.isfinite(ang)
    if not np.any(m):
        return out
    idx = np.flatnonzero(m)
    # Split into runs separated by index gaps (e.g. QA-masked frequency bins).
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [idx.size]))
    for s, e in zip(starts, ends):
        ii = idx[s:e]
        out[ii] = np.unwrap(ang[ii]) * 180.0 / np.pi
    return out


def _unwrap_phase_deg_at_freq(
    zc: np.ndarray,
    f_hz: np.ndarray,
    f_anchor_hz: float,
) -> np.ndarray:
    """
    Unwrap phase (deg) along frequency; each contiguous run is shifted so phase = 0
  at the bin nearest ``f_anchor_hz`` (display anchor, not a physical rotation).
    """
    zc = np.asarray(zc, dtype=complex)
    ang = np.angle(zc)
    f_hz = np.asarray(f_hz, dtype=float)
    out = np.full(ang.shape, np.nan, dtype=float)
    m = np.isfinite(ang) & np.isfinite(f_hz)
    if not np.any(m):
        return out
    idx = np.flatnonzero(m)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [idx.size]))
    f_anchor = float(f_anchor_hz)
    for s, e in zip(starts, ends):
        ii = idx[s:e]
        unwrapped = np.unwrap(ang[ii])
        j = int(np.argmin(np.abs(f_hz[ii] - f_anchor)))
        unwrapped -= unwrapped[j]
        out[ii] = unwrapped * 180.0 / np.pi
    return out


# Bode plots: omit failed / numerical garbage so autoscale stays on the physical band.
_BODE_ABS_FLOOR_PU = 1e-9
_BODE_MAG_DB_PLOT = (-120.0, 80.0)


def _decimate_bode_plot_freqs(
    freqs: np.ndarray,
    *arrays: np.ndarray,
    min_log_step: float = 0.04,
) -> tuple[np.ndarray, ...]:
    """Drop plot points closer than ``min_log_step`` decades (display only; CSV unchanged)."""
    f = np.asarray(freqs, dtype=float)
    if f.size < 3 or min_log_step <= 0.0:
        return (f, *arrays)
    keep = [0]
    for i in range(1, f.size):
        if f[i] <= 0.0 or f[keep[-1]] <= 0.0:
            continue
        if np.log10(f[i] / f[keep[-1]]) >= float(min_log_step):
            keep.append(i)
    if keep[-1] != f.size - 1:
        keep.append(int(f.size - 1))
    idx = np.asarray(keep, dtype=int)
    out: list[np.ndarray] = [f[idx]]
    for a in arrays:
        out.append(np.asarray(a, dtype=complex)[idx])
    return tuple(out)


def _plot_freq_mask(freqs: np.ndarray, plot_f_min_hz: float | None) -> np.ndarray:
    """Boolean mask for Bode/conditioning plots (CSV retains full ID band)."""
    f = np.asarray(freqs, dtype=float)
    if plot_f_min_hz is None:
        return np.ones(f.shape, dtype=bool)
    return f >= float(plot_f_min_hz)


def _draw_freq_marker(ax: plt.Axes, f_hz: float) -> None:
    """Thin vertical guide at split / reference frequency (semilog-safe)."""
    y0, y1 = ax.get_ylim()
    ax.plot(
        [f_hz, f_hz],
        [y0, y1],
        color="0.55",
        ls=(0, (2, 3)),
        lw=0.65,
        alpha=0.5,
        zorder=4,
        scalex=False,
        scaley=False,
        clip_on=True,
    )


def _bode_plot_valid(zc: np.ndarray) -> np.ndarray:
    """True where a complex impedance/admittance sample should be drawn."""
    zc = np.asarray(zc, dtype=complex)
    mag = np.abs(zc)
    ok = np.isfinite(zc.real) & np.isfinite(zc.imag) & np.isfinite(mag) & (mag > _BODE_ABS_FLOOR_PU)
    if not np.any(ok):
        return ok
    mag_db = 20.0 * np.log10(mag[ok])
    lo, hi = _BODE_MAG_DB_PLOT
    ok_out = np.zeros(zc.shape, dtype=bool)
    ok_out[ok] = (mag_db >= lo) & (mag_db <= hi)
    return ok_out


def _mag_db_bode_plot(zc: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(zc).shape, np.nan, dtype=float)
    ok = _bode_plot_valid(zc)
    out[ok] = 20.0 * np.log10(np.abs(np.asarray(zc, dtype=complex)[ok]))
    return out


def _phase_deg_bode_plot(
    zc: np.ndarray,
    f_hz: np.ndarray | None = None,
    *,
    f_anchor_hz: float | None = None,
) -> np.ndarray:
    zc = np.asarray(zc, dtype=complex).copy()
    zc[~_bode_plot_valid(zc)] = np.nan + 1j * np.nan
    if f_hz is not None and f_anchor_hz is not None:
        return _unwrap_phase_deg_at_freq(zc, f_hz, float(f_anchor_hz))
    return _unwrap_phase_deg(zc)


def _zm_bode_plot_arrays(
    z00: np.ndarray,
    z01: np.ndarray,
    z10: np.ndarray,
    z11: np.ndarray,
    *,
    det_ipert_abs: np.ndarray | None = None,
    cross_snr_db: float = 35.0,
    det_rel_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask weak off-diagonal mirror entries so Bode plots do not show noise-floor jitter."""
    z00p = np.asarray(z00, dtype=complex).copy()
    z11p = np.asarray(z11, dtype=complex).copy()
    z01p = np.asarray(z01, dtype=complex).copy()
    z10p = np.asarray(z10, dtype=complex).copy()
    ref = np.maximum(np.abs(z00p), np.abs(z11p))
    floor = ref * (10.0 ** (-float(cross_snr_db) / 20.0))
    nan_c = np.nan + 1j * np.nan
    weak = (~np.isfinite(ref)) | (ref <= _BODE_ABS_FLOOR_PU) | (np.abs(z01p) < floor) | (np.abs(z10p) < floor)
    z01p[weak] = nan_c
    z10p[weak] = nan_c
    if det_ipert_abs is not None:
        det = np.asarray(det_ipert_abs, dtype=float)
        ok = np.isfinite(det) & (det > 0.0)
        if np.any(ok):
            det_med = float(np.median(det[ok]))
            if det_med > 0.0:
                bad = ~np.isfinite(det) | (det < det_rel_floor * det_med)
                z01p[bad] = nan_c
                z10p[bad] = nan_c
    return z00p, z01p, z10p, z11p


def plot_sequence_four_zm_bode(
    freqs: np.ndarray,
    zm_00: np.ndarray,
    zm_01: np.ndarray,
    zm_10: np.ndarray,
    zm_11: np.ndarray,
    out_png: str,
    *,
    title: str,
    mag_db_label: str,
    phase_ylabel: str,
    legend_prefix: str = "Z",
    mark_hz: float | None = None,
    plot_f_min_hz: float | None = None,
    decimate_log_step: float | None = None,
) -> None:
    """
    Four-panel Bode plot (same layout as dq matrix Bode): one subplot per Zm (or Ym) entry
    labelled ++, +-, -+, -- from the mirror-frequency / stacked harmonic 2×2 used in this repo.

    With the measurement stack used here (Pedra/Sainz/Monjo-style formulation),

      [U(ω); U*(ω)] = Zm(ω) [I(ω); I*(ω)],

    index (row, col) gives coupling from the column current content to the row voltage content.
    We label the four entries Z++, Z+-, Z-+, Z-- ≡ Zm[0,0], Zm[0,1], Zm[1,0], Zm[1,1].
    The same indexing applies to admittance Ym (set legend_prefix='Y' for Y plots).

    If mark_hz is set and lies within the plotted frequency range, a vertical dashed line is drawn
    on each subplot when ``mark_hz`` is set. Pass None to omit (default).
    Uses an explicit vertical segment (not axvline) so the marker stays visible on semilog-x axes.
    """
    f_all = np.asarray(freqs, dtype=float)
    pmask = _plot_freq_mask(f_all, plot_f_min_hz)
    lp = (legend_prefix or "Z").strip()
    elems = [
        (f"{lp}++", np.asarray(zm_00, dtype=complex)[pmask]),
        (f"{lp}+-", np.asarray(zm_01, dtype=complex)[pmask]),
        (f"{lp}-+", np.asarray(zm_10, dtype=complex)[pmask]),
        (f"{lp}--", np.asarray(zm_11, dtype=complex)[pmask]),
    ]
    f = f_all[pmask]
    z_elems = [z for _, z in elems]
    if decimate_log_step is not None and decimate_log_step > 0.0:
        f, *z_elems = _decimate_bode_plot_freqs(f, *z_elems, min_log_step=float(decimate_log_step))
    elems = [(elems[i][0], z_elems[i]) for i in range(len(elems))]
    f_pos = f[np.isfinite(f) & (f > 0.0)]
    f_xlim = (float(np.min(f_pos)), float(np.max(f_pos))) if f_pos.size else None
    fm_draw: float | None = None
    if mark_hz is not None and f.size > 0:
        fm = float(mark_hz)
        if np.isfinite(fm) and fm > 0.0:
            f_lo = float(np.nanmin(f))
            f_hi = float(np.nanmax(f))
            tol = max(1e-9 * f_hi, 1e-12)
            if (f_lo - tol) <= fm <= (f_hi + tol):
                fm_draw = fm
            else:
                print(
                    f"plot_sequence_four_zm_bode: mark_hz={fm:g} Hz is outside plotted f range "
                    f"[{f_lo:g}, {f_hi:g}] Hz — vertical marker omitted.",
                    flush=True,
                )
    def _plot_panel(ax_mag, ax_ph, zc: np.ndarray, name: str) -> None:
        zc = np.asarray(zc, dtype=complex)
        mag = _mag_db_bode_plot(zc)
        ph = _phase_deg_bode_plot(zc)
        has_data = np.any(np.isfinite(mag)) or np.any(np.isfinite(ph))
        if has_data:
            from casestudies.impedance_stability.plots.bode_axes import semilogx_bode

            semilogx_bode(ax_mag, f, mag)
            semilogx_bode(ax_ph, f, ph)
        else:
            ax_mag.text(
                0.5,
                0.5,
                "no data above plot SNR",
                transform=ax_mag.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.45",
            )
        from casestudies.impedance_stability.plots.bode_axes import phase_ylabel

        ax_mag.set_title(name)
        ax_mag.set_ylabel(mag_db_label)
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_ph.set_ylabel(phase_ylabel())
        ax_ph.grid(True, which="both", alpha=0.3)
        if fm_draw is not None:
            for ax in (ax_mag, ax_ph):
                _draw_freq_marker(ax, fm_draw)

    fig = plt.figure(figsize=(12.0, 7.6))
    outer = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)
    layout = [(0, 0), (0, 1), (1, 0), (1, 1)]
    from casestudies.impedance_stability.plots.bode_axes import apply_bode_mag_phase_axes

    for (name, zc), (r, c) in zip(elems, layout):
        sub = outer[r, c].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.10)
        ax_mag = fig.add_subplot(sub[0, 0])
        ax_ph = fig.add_subplot(sub[1, 0], sharex=ax_mag)
        _plot_panel(ax_mag, ax_ph, zc, name)
        if f_xlim is not None:
            apply_bode_mag_phase_axes(
                ax_mag, ax_ph, f_xlim[0], f_xlim[1], show_freq_axis=(r == 1)
            )

    if title:
        from casestudies.impedance_stability.plots.bode_axes import apply_matrix_bode_layout

        apply_matrix_bode_layout(fig, title)
    elif f_xlim is not None:
        fig.subplots_adjust(top=0.92, hspace=0.32, wspace=0.28)

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


_MIMO_T = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1j], [1.0, -1j]], dtype=complex)
_MIMO_TINV = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [-1j, 1j]], dtype=complex)
_RANK_TOL_MIMO = 1e-18
# Default cap for κ(Y_dev): bins above this are flagged as poorly conditioned (not merely large |Y|).
KAPPA_YDEV_MAX = 50.0


def _matrix_kappa_2x2(m: np.ndarray) -> float:
    """Condition number κ = σ_max/σ_min for a 2×2 complex matrix."""
    m = np.asarray(m, dtype=complex)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)):
        return float("nan")
    try:
        s = np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if s.size < 2 or float(s[-1]) <= _RANK_TOL_MIMO:
        return float("inf")
    return float(s[0] / s[-1])


def _mimo_a_inv_b(
    a_m: np.ndarray,
    b_m: np.ndarray,
    *,
    reg_det_rel: float = 0.0,
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """``A @ inv(B)`` with optional Tikhonov damping on ``B`` (used for Z_bus and Y_dev)."""
    b_m = np.asarray(b_m, dtype=complex)
    d_b = b_m[0, 0] * b_m[1, 1] - b_m[0, 1] * b_m[1, 0]
    det_abs = float(abs(d_b))
    if not np.isfinite(det_abs) or det_abs <= _RANK_TOL_MIMO:
        return None, det_abs
    b_reg = b_m.copy()
    if reg_det_rel > 0.0:
        scale = reg_det_rel * (float(np.sum(np.abs(b_reg) ** 2)) / 2.0 + 1e-18)
        b_reg = b_reg + scale * np.eye(2, dtype=complex)
    try:
        out = np.asarray(a_m, dtype=complex) @ np.linalg.inv(b_reg)
    except np.linalg.LinAlgError:
        return None, det_abs
    return out, det_abs


def _mimo_zm_bus(
    u_m: np.ndarray,
    ipert_m: np.ndarray,
    *,
    reg_det_rel: float = 0.0,
) -> tuple[np.ndarray, float] | tuple[None, float]:
    """Z_m = U @ inv(I_pert) with optional Tikhonov damping on ill-conditioned inversions."""
    return _mimo_a_inv_b(u_m, ipert_m, reg_det_rel=reg_det_rel)


def _mimo_impedance_at_freqs(
    freqs: np.ndarray,
    Vp_d: np.ndarray,
    Vm_d: np.ndarray,
    Vp_q: np.ndarray,
    Vm_q: np.ndarray,
    Ip_p_d: np.ndarray,
    Ip_m_d: np.ndarray,
    Ip_p_q: np.ndarray,
    Ip_m_q: np.ndarray,
    Ia_p_d: np.ndarray,
    Ia_m_d: np.ndarray,
    Ia_p_q: np.ndarray,
    Ia_m_q: np.ndarray,
    *,
    mimo_reg_det_rel: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build Z_bus, Y_dev, Z_dev and mirror 2×2 blocks from per-frequency phasor estimates."""
    n = int(freqs.size)
    nan_c = np.nan + 1j * np.nan

    def _zeros_c() -> np.ndarray:
        return np.zeros(n, dtype=complex)

    Zbus_dd = _zeros_c()
    Zbus_dq = _zeros_c()
    Zbus_qd = _zeros_c()
    Zbus_qq = _zeros_c()
    detIpert_abs = np.zeros(n, dtype=float)
    Ydev_dd = _zeros_c()
    Ydev_dq = _zeros_c()
    Ydev_qd = _zeros_c()
    Ydev_qq = _zeros_c()
    detVt_abs = np.zeros(n, dtype=float)
    Zdev_dd = _zeros_c()
    Zdev_dq = _zeros_c()
    Zdev_qd = _zeros_c()
    Zdev_qq = _zeros_c()
    detYdev_abs = np.zeros(n, dtype=float)
    kappa_Ydev = np.zeros(n, dtype=float)
    detYdev_norm = np.zeros(n, dtype=float)
    Zplus_bus = _zeros_c()
    Zminus_bus = _zeros_c()
    Zm_bus_00 = _zeros_c()
    Zm_bus_01 = _zeros_c()
    Zm_bus_10 = _zeros_c()
    Zm_bus_11 = _zeros_c()
    Yplus_dev = _zeros_c()
    Yminus_dev = _zeros_c()
    Ym_dev_00 = _zeros_c()
    Ym_dev_01 = _zeros_c()
    Ym_dev_10 = _zeros_c()
    Ym_dev_11 = _zeros_c()
    Zplus_dev = _zeros_c()
    Zminus_dev = _zeros_c()
    Zm_dev_00 = _zeros_c()
    Zm_dev_01 = _zeros_c()
    Zm_dev_10 = _zeros_c()
    Zm_dev_11 = _zeros_c()

    for i in range(n):
        U_m = np.array(
            [[Vp_d[i], Vp_q[i]], [np.conj(Vm_d[i]), np.conj(Vm_q[i])]],
            dtype=complex,
        )
        Ipert_m = np.array(
            [[Ip_p_d[i], Ip_p_q[i]], [np.conj(Ip_m_d[i]), np.conj(Ip_m_q[i])]],
            dtype=complex,
        )
        Ia_m = np.array(
            [[Ia_p_d[i], Ia_p_q[i]], [np.conj(Ia_m_d[i]), np.conj(Ia_m_q[i])]],
            dtype=complex,
        )

        zm_bus, detIpert_abs[i] = _mimo_zm_bus(U_m, Ipert_m, reg_det_rel=float(mimo_reg_det_rel))
        if zm_bus is not None:
            Zplus_bus[i] = zm_bus[0, 0]
            Zminus_bus[i] = zm_bus[0, 1]
            Zm_bus_00[i] = zm_bus[0, 0]
            Zm_bus_01[i] = zm_bus[0, 1]
            Zm_bus_10[i] = zm_bus[1, 0]
            Zm_bus_11[i] = zm_bus[1, 1]
            Zb_real = _MIMO_TINV @ zm_bus @ _MIMO_T
            Zbus_dd[i], Zbus_dq[i] = Zb_real[0, 0], Zb_real[0, 1]
            Zbus_qd[i], Zbus_qq[i] = Zb_real[1, 0], Zb_real[1, 1]
        else:
            Zbus_dd[i] = Zbus_dq[i] = Zbus_qd[i] = Zbus_qq[i] = nan_c
            Zplus_bus[i] = Zminus_bus[i] = nan_c
            Zm_bus_00[i] = Zm_bus_01[i] = Zm_bus_10[i] = Zm_bus_11[i] = nan_c

        dV = U_m[0, 0] * U_m[1, 1] - U_m[0, 1] * U_m[1, 0]
        detVt_abs[i] = float(abs(dV))
        if not np.isfinite(detVt_abs[i]) or detVt_abs[i] < _RANK_TOL_MIMO:
            Ydev_dd[i] = Ydev_dq[i] = Ydev_qd[i] = Ydev_qq[i] = nan_c
            Zdev_dd[i] = Zdev_dq[i] = Zdev_qd[i] = Zdev_qq[i] = nan_c
            detYdev_abs[i] = np.nan
            kappa_Ydev[i] = np.nan
            detYdev_norm[i] = np.nan
            Yplus_dev[i] = Yminus_dev[i] = nan_c
            Ym_dev_00[i] = Ym_dev_01[i] = Ym_dev_10[i] = Ym_dev_11[i] = nan_c
            Zplus_dev[i] = Zminus_dev[i] = nan_c
            Zm_dev_00[i] = Zm_dev_01[i] = Zm_dev_10[i] = Zm_dev_11[i] = nan_c
            continue

        reg_u = max(float(mimo_reg_det_rel), 0.05)
        ym_out = _mimo_a_inv_b(Ia_m, U_m, reg_det_rel=reg_u)
        if ym_out[0] is None:
            Ydev_dd[i] = Ydev_dq[i] = Ydev_qd[i] = Ydev_qq[i] = nan_c
            detYdev_abs[i] = ym_out[1]
            kappa_Ydev[i] = np.nan
            detYdev_norm[i] = np.nan
            Yplus_dev[i] = Yminus_dev[i] = nan_c
            Ym_dev_00[i] = Ym_dev_01[i] = Ym_dev_10[i] = Ym_dev_11[i] = nan_c
            Zplus_dev[i] = Zminus_dev[i] = nan_c
            Zm_dev_00[i] = Zm_dev_01[i] = Zm_dev_10[i] = Zm_dev_11[i] = nan_c
            continue
        Ym_dev, _ = ym_out
        Ym_dev_00[i] = Ym_dev[0, 0]
        Ym_dev_01[i] = Ym_dev[0, 1]
        Ym_dev_10[i] = Ym_dev[1, 0]
        Ym_dev_11[i] = Ym_dev[1, 1]
        Yplus_dev[i] = Ym_dev[0, 0]
        Yminus_dev[i] = Ym_dev[0, 1]
        dYm = Ym_dev[0, 0] * Ym_dev[1, 1] - Ym_dev[0, 1] * Ym_dev[1, 0]
        if np.isfinite(dYm.real) and np.isfinite(dYm.imag) and abs(dYm) > _RANK_TOL_MIMO:
            Zm_dev = np.linalg.inv(Ym_dev)
            Zplus_dev[i] = Zm_dev[0, 0]
            Zminus_dev[i] = Zm_dev[0, 1]
            Zm_dev_00[i] = Zm_dev[0, 0]
            Zm_dev_01[i] = Zm_dev[0, 1]
            Zm_dev_10[i] = Zm_dev[1, 0]
            Zm_dev_11[i] = Zm_dev[1, 1]
        else:
            Zplus_dev[i] = Zminus_dev[i] = nan_c
            Zm_dev_00[i] = Zm_dev_01[i] = Zm_dev_10[i] = Zm_dev_11[i] = nan_c
        Y_real = _MIMO_TINV @ Ym_dev @ _MIMO_T
        Ydev_dd[i], Ydev_dq[i] = Y_real[0, 0], Y_real[0, 1]
        Ydev_qd[i], Ydev_qq[i] = Y_real[1, 0], Y_real[1, 1]

        dY = Y_real[0, 0] * Y_real[1, 1] - Y_real[0, 1] * Y_real[1, 0]
        detYdev_abs[i] = float(abs(dY))
        kappa_Ydev[i] = _matrix_kappa_2x2(Y_real)
        y_norm_sq = float(np.linalg.norm(Y_real, ord="fro") ** 2)
        detYdev_norm[i] = (
            float(detYdev_abs[i] / y_norm_sq) if y_norm_sq > _RANK_TOL_MIMO else np.nan
        )
        if np.isfinite(detYdev_abs[i]) and detYdev_abs[i] > _RANK_TOL_MIMO:
            if np.isfinite(Zm_dev_00[i].real) and np.isfinite(Zm_dev_00[i].imag):
                Zm_full = np.array(
                    [[Zm_dev_00[i], Zm_dev_01[i]], [Zm_dev_10[i], Zm_dev_11[i]]],
                    dtype=complex,
                )
                Z_real = _MIMO_TINV @ Zm_full @ _MIMO_T
                Zdev_dd[i], Zdev_dq[i] = Z_real[0, 0], Z_real[0, 1]
                Zdev_qd[i], Zdev_qq[i] = Z_real[1, 0], Z_real[1, 1]
            else:
                Z_real = np.linalg.inv(Y_real)
                Zdev_dd[i], Zdev_dq[i] = Z_real[0, 0], Z_real[0, 1]
                Zdev_qd[i], Zdev_qq[i] = Z_real[1, 0], Z_real[1, 1]
        else:
            Zdev_dd[i] = Zdev_dq[i] = Zdev_qd[i] = Zdev_qq[i] = nan_c

    return {
        "Zbus_dd": Zbus_dd,
        "Zbus_dq": Zbus_dq,
        "Zbus_qd": Zbus_qd,
        "Zbus_qq": Zbus_qq,
        "detIpert_abs": detIpert_abs,
        "Ydev_dd": Ydev_dd,
        "Ydev_dq": Ydev_dq,
        "Ydev_qd": Ydev_qd,
        "Ydev_qq": Ydev_qq,
        "detVt_abs": detVt_abs,
        "Zdev_dd": Zdev_dd,
        "Zdev_dq": Zdev_dq,
        "Zdev_qd": Zdev_qd,
        "Zdev_qq": Zdev_qq,
        "detYdev_abs": detYdev_abs,
        "kappa_Ydev": kappa_Ydev,
        "detYdev_norm": detYdev_norm,
        "Zplus_bus": Zplus_bus,
        "Zminus_bus": Zminus_bus,
        "Zm_bus_00": Zm_bus_00,
        "Zm_bus_01": Zm_bus_01,
        "Zm_bus_10": Zm_bus_10,
        "Zm_bus_11": Zm_bus_11,
        "Yplus_dev": Yplus_dev,
        "Yminus_dev": Yminus_dev,
        "Ym_dev_00": Ym_dev_00,
        "Ym_dev_01": Ym_dev_01,
        "Ym_dev_10": Ym_dev_10,
        "Ym_dev_11": Ym_dev_11,
        "Zplus_dev": Zplus_dev,
        "Zminus_dev": Zminus_dev,
        "Zm_dev_00": Zm_dev_00,
        "Zm_dev_01": Zm_dev_01,
        "Zm_dev_10": Zm_dev_10,
        "Zm_dev_11": Zm_dev_11,
    }


def _mimo_impedance_results_to_dataframe(freqs: np.ndarray, res: dict[str, np.ndarray]) -> pd.DataFrame:
    Zbus_dd = res["Zbus_dd"]
    Zbus_dq = res["Zbus_dq"]
    Zbus_qd = res["Zbus_qd"]
    Zbus_qq = res["Zbus_qq"]
    Ydev_dd = res["Ydev_dd"]
    Ydev_dq = res["Ydev_dq"]
    Ydev_qd = res["Ydev_qd"]
    Ydev_qq = res["Ydev_qq"]
    Zdev_dd = res["Zdev_dd"]
    Zdev_dq = res["Zdev_dq"]
    Zdev_qd = res["Zdev_qd"]
    Zdev_qq = res["Zdev_qq"]
    Zplus_bus = res["Zplus_bus"]
    Zminus_bus = res["Zminus_bus"]
    Yplus_dev = res["Yplus_dev"]
    Yminus_dev = res["Yminus_dev"]
    Zplus_dev = res["Zplus_dev"]
    Zminus_dev = res["Zminus_dev"]
    return pd.DataFrame(
        {
            "f_Hz": freqs,
            "Zbus_dd_re": np.real(Zbus_dd),
            "Zbus_dd_im": np.imag(Zbus_dd),
            "Zbus_dq_re": np.real(Zbus_dq),
            "Zbus_dq_im": np.imag(Zbus_dq),
            "Zbus_qd_re": np.real(Zbus_qd),
            "Zbus_qd_im": np.imag(Zbus_qd),
            "Zbus_qq_re": np.real(Zbus_qq),
            "Zbus_qq_im": np.imag(Zbus_qq),
            "detIpert_abs": res["detIpert_abs"],
            "Ydev_dd_re": np.real(Ydev_dd),
            "Ydev_dd_im": np.imag(Ydev_dd),
            "Ydev_dq_re": np.real(Ydev_dq),
            "Ydev_dq_im": np.imag(Ydev_dq),
            "Ydev_qd_re": np.real(Ydev_qd),
            "Ydev_qd_im": np.imag(Ydev_qd),
            "Ydev_qq_re": np.real(Ydev_qq),
            "Ydev_qq_im": np.imag(Ydev_qq),
            "detVt_abs": res["detVt_abs"],
            "Zdev_dd_re": np.real(Zdev_dd),
            "Zdev_dd_im": np.imag(Zdev_dd),
            "Zdev_dq_re": np.real(Zdev_dq),
            "Zdev_dq_im": np.imag(Zdev_dq),
            "Zdev_qd_re": np.real(Zdev_qd),
            "Zdev_qd_im": np.imag(Zdev_qd),
            "Zdev_qq_re": np.real(Zdev_qq),
            "Zdev_qq_im": np.imag(Zdev_qq),
            "detYdev_abs": res["detYdev_abs"],
            "kappa_Ydev": res["kappa_Ydev"],
            "detYdev_norm": res["detYdev_norm"],
            "Zbus_plus_re": np.real(Zplus_bus),
            "Zbus_plus_im": np.imag(Zplus_bus),
            "Zbus_minus_re": np.real(Zminus_bus),
            "Zbus_minus_im": np.imag(Zminus_bus),
            "Zbus_m00_re": np.real(res["Zm_bus_00"]),
            "Zbus_m00_im": np.imag(res["Zm_bus_00"]),
            "Zbus_m01_re": np.real(res["Zm_bus_01"]),
            "Zbus_m01_im": np.imag(res["Zm_bus_01"]),
            "Zbus_m10_re": np.real(res["Zm_bus_10"]),
            "Zbus_m10_im": np.imag(res["Zm_bus_10"]),
            "Zbus_m11_re": np.real(res["Zm_bus_11"]),
            "Zbus_m11_im": np.imag(res["Zm_bus_11"]),
            "Ydev_plus_re": np.real(Yplus_dev),
            "Ydev_plus_im": np.imag(Yplus_dev),
            "Ydev_minus_re": np.real(Yminus_dev),
            "Ydev_minus_im": np.imag(Yminus_dev),
            "Ydev_m00_re": np.real(res["Ym_dev_00"]),
            "Ydev_m00_im": np.imag(res["Ym_dev_00"]),
            "Ydev_m01_re": np.real(res["Ym_dev_01"]),
            "Ydev_m01_im": np.imag(res["Ym_dev_01"]),
            "Ydev_m10_re": np.real(res["Ym_dev_10"]),
            "Ydev_m10_im": np.imag(res["Ym_dev_10"]),
            "Ydev_m11_re": np.real(res["Ym_dev_11"]),
            "Ydev_m11_im": np.imag(res["Ym_dev_11"]),
            "Zdev_plus_re": np.real(Zplus_dev),
            "Zdev_plus_im": np.imag(Zplus_dev),
            "Zdev_minus_re": np.real(Zminus_dev),
            "Zdev_minus_im": np.imag(Zminus_dev),
            "Zdev_m00_re": np.real(res["Zm_dev_00"]),
            "Zdev_m00_im": np.imag(res["Zm_dev_00"]),
            "Zdev_m01_re": np.real(res["Zm_dev_01"]),
            "Zdev_m01_im": np.imag(res["Zm_dev_01"]),
            "Zdev_m10_re": np.real(res["Zm_dev_10"]),
            "Zdev_m10_im": np.imag(res["Zm_dev_10"]),
            "Zdev_m11_re": np.real(res["Zm_dev_11"]),
            "Zdev_m11_im": np.imag(res["Zm_dev_11"]),
            "Zbus_sym_dd_minus_qq_abs": np.abs(Zbus_dd - Zbus_qq),
            "Zbus_sym_dq_plus_qd_abs": np.abs(Zbus_dq + Zbus_qd),
            "Ydev_sym_dd_minus_qq_abs": np.abs(Ydev_dd - Ydev_qq),
            "Ydev_sym_dq_plus_qd_abs": np.abs(Ydev_dq + Ydev_qd),
            "Zbus_mirror_ratio": np.abs(Zminus_bus) / np.maximum(np.abs(Zplus_bus), 1e-30),
            "Ydev_mirror_ratio": np.abs(Yminus_dev) / np.maximum(np.abs(Yplus_dev), 1e-30),
            "Zdev_mirror_ratio": np.abs(Zminus_dev) / np.maximum(np.abs(Zplus_dev), 1e-30),
        }
    )


def save_mimo_impedance_outputs(
    freqs: np.ndarray,
    res: dict[str, np.ndarray],
    out_dir: str,
    *,
    mark_hz: float | None = None,
    matrix_csv: str = "impedance_matrix.csv",
    plot_suffix: str = "",
    sequence_title_tag: str = "",
    mask_weak_cross_terms_db: float | None = None,
    plot_f_min_hz: float | None = None,
    plot_bus_only: bool = False,
) -> str:
    """Write impedance CSV and Bode PNGs from MIMO results.

    Set ``plot_bus_only=True`` to skip Z_dev / Y_dev figures (use for thesis replots).
    """
    os.makedirs(out_dir, exist_ok=True)
    suf = plot_suffix or ""
    tag = sequence_title_tag.strip()
    if tag and not tag.startswith(" "):
        tag = " " + tag

    out = _mimo_impedance_results_to_dataframe(freqs, res)
    out_csv = os.path.join(out_dir, matrix_csv)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    mr = out["Ydev_mirror_ratio"].to_numpy(dtype=float)
    mr_f = mr[np.isfinite(mr)]
    if mr_f.size:
        print(
            f"Ydev_mirror_ratio: median={float(np.median(mr_f)):.4g}, "
            f"max={float(np.max(mr_f)):.4g}",
            flush=True,
        )

    Zdev_dd = res["Zdev_dd"]
    Zdev_dq = res["Zdev_dq"]
    Zdev_qd = res["Zdev_qd"]
    Zdev_qq = res["Zdev_qq"]
    Ydev_dd = res["Ydev_dd"]
    Ydev_dq = res["Ydev_dq"]
    Ydev_qd = res["Ydev_qd"]
    Ydev_qq = res["Ydev_qq"]
    Zbus_dd = res["Zbus_dd"]
    Zbus_dq = res["Zbus_dq"]
    Zbus_qd = res["Zbus_qd"]
    Zbus_qq = res["Zbus_qq"]

    pmask = _plot_freq_mask(freqs, plot_f_min_hz)
    f_plot = np.asarray(freqs, dtype=float)[pmask]

    from casestudies.impedance_stability.plots.bode_axes import (
        apply_bode_mag_phase_axes,
        apply_matrix_bode_layout,
        matrix_bode_suptitle,
        phase_ylabel,
        semilogx_bode,
    )

    f_lo = float(np.nanmin(f_plot)) if f_plot.size else 0.1
    f_hi = float(np.nanmax(f_plot)) if f_plot.size else 10.0

    def _plot_one(ax_mag, ax_ph, f, Zc, title: str, *, ylab_mag: str) -> None:
        zc = np.asarray(Zc, dtype=complex)
        semilogx_bode(ax_mag, f, _mag_db_bode_plot(zc))
        ax_mag.set_title(title)
        ax_mag.set_ylabel(ylab_mag)
        ax_mag.grid(True, which="both", alpha=0.3)
        semilogx_bode(ax_ph, f, _phase_deg_bode_plot(zc))
        ax_ph.set_ylabel(phase_ylabel())
        ax_ph.grid(True, which="both", alpha=0.3)

    def _matrix_bode(
        elems: list[tuple[str, np.ndarray]],
        out_png: str,
        matrix_title: str,
        *,
        ylab_mag: str = "|Z|",
    ) -> None:
        fig = plt.figure(figsize=(12.0, 7.6))
        outer = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)
        layout = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (name, Zc), (r, c) in zip(elems, layout):
            sub = outer[r, c].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.10)
            ax_mag = fig.add_subplot(sub[0, 0])
            ax_ph = fig.add_subplot(sub[1, 0], sharex=ax_mag)
            _plot_one(ax_mag, ax_ph, f_plot, Zc[pmask], name, ylab_mag=ylab_mag)
            apply_bode_mag_phase_axes(ax_mag, ax_ph, f_lo, f_hi, show_freq_axis=(r == 1))
        apply_matrix_bode_layout(fig, matrix_bode_suptitle(matrix_title))
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    if not plot_bus_only:
        zdev_png = os.path.join(out_dir, f"Zdev_matrix_bode{suf}.png")
        _matrix_bode(
            [("Zdd", Zdev_dd), ("Zdq", Zdev_dq), ("Zqd", Zdev_qd), ("Zqq", Zdev_qq)],
            zdev_png,
            "Z_dev (dq)",
            ylab_mag="|Z|",
        )
        print(f"Saved {zdev_png}")

        ydev_png = os.path.join(out_dir, f"Ydev_matrix_bode{suf}.png")
        _matrix_bode(
            [("Ydd", Ydev_dd), ("Ydq", Ydev_dq), ("Yqd", Ydev_qd), ("Yqq", Ydev_qq)],
            ydev_png,
            "Y_dev (dq)",
            ylab_mag="|Y|",
        )
        print(f"Saved {ydev_png}")

    zbus_png = os.path.join(out_dir, f"Zbus_matrix_bode{suf}.png")
    _matrix_bode(
        [("Zdd", Zbus_dd), ("Zdq", Zbus_dq), ("Zqd", Zbus_qd), ("Zqq", Zbus_qq)],
        zbus_png,
        "Z_bus (dq)",
        ylab_mag="|Z|",
    )
    print(f"Saved {zbus_png}")

    det_ipert = res.get("detIpert_abs")
    cross_db = mask_weak_cross_terms_db

    def _seq_zm(z00, z01, z10, z11):
        if cross_db is None:
            return z00, z01, z10, z11
        return _zm_bode_plot_arrays(
            z00, z01, z10, z11, det_ipert_abs=det_ipert, cross_snr_db=float(cross_db)
        )

    zb00, zb01, zb10, zb11 = _seq_zm(
        res["Zm_bus_00"], res["Zm_bus_01"], res["Zm_bus_10"], res["Zm_bus_11"]
    )
    plot_sequence_four_zm_bode(
        freqs,
        zb00,
        zb01,
        zb10,
        zb11,
        os.path.join(out_dir, f"Zbus_sequence_bode{suf}.png"),
        title=f"Z_bus: Zm (++ / +- / -+ / --){tag} — magnitude (dB re 1 pu), phase (deg, unwrap)",
        mag_db_label="|Z|",
        phase_ylabel="",
        mark_hz=mark_hz,
        plot_f_min_hz=plot_f_min_hz,
        decimate_log_step=0.03,
    )
    print(f"Saved {os.path.join(out_dir, f'Zbus_sequence_bode{suf}.png')}")

    if not plot_bus_only:
        yd00, yd01, yd10, yd11 = _seq_zm(
            res["Ym_dev_00"], res["Ym_dev_01"], res["Ym_dev_10"], res["Ym_dev_11"]
        )
        plot_sequence_four_zm_bode(
            freqs,
            yd00,
            yd01,
            yd10,
            yd11,
            os.path.join(out_dir, f"Ydev_sequence_bode{suf}.png"),
            title=f"Y_dev: Ym (++ / +- / -+ / --){tag} — magnitude (dB re 1 pu), phase (deg, unwrap)",
            mag_db_label="|Y|",
            phase_ylabel="",
            legend_prefix="Y",
            mark_hz=mark_hz,
            plot_f_min_hz=plot_f_min_hz,
            decimate_log_step=0.04,
        )
        print(f"Saved {os.path.join(out_dir, f'Ydev_sequence_bode{suf}.png')}")

        zd00, zd01, zd10, zd11 = _seq_zm(
            res["Zm_dev_00"], res["Zm_dev_01"], res["Zm_dev_10"], res["Zm_dev_11"]
        )
        plot_sequence_four_zm_bode(
            freqs,
            zd00,
            zd01,
            zd10,
            zd11,
            os.path.join(out_dir, f"Zdev_sequence_bode{suf}.png"),
            title=f"Z_dev: Zm (++ / +- / -+ / --){tag} — magnitude (dB re 1 pu), phase (deg, unwrap)",
            mag_db_label="|Z|",
            phase_ylabel="",
            mark_hz=mark_hz,
            plot_f_min_hz=plot_f_min_hz,
            decimate_log_step=0.04,
        )
        print(f"Saved {os.path.join(out_dir, f'Zdev_sequence_bode{suf}.png')}")

        zmdev_png = os.path.join(out_dir, f"Zdev_dqcomplex_matrix_bode{suf}.png")
        _matrix_bode(
            [
                ("Zm_00", res["Zm_dev_00"]),
                ("Zm_01", res["Zm_dev_01"]),
                ("Zm_10", res["Zm_dev_10"]),
                ("Zm_11", res["Zm_dev_11"]),
            ],
            zmdev_png,
            "Z_dev mirror Zm (++ / +- / -+ / --), dB re 1 pu",
            ylab_mag="|Z_m|",
        )
        print(f"Saved {zmdev_png}")

    return out_csv

