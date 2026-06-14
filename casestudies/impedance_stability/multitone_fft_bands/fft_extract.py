from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from casestudies.impedance_stability.identification.impedance_matrix_uic_wt import (
    _mimo_impedance_at_freqs,
    save_mimo_impedance_outputs,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    DT_S,
    PlantId,
    T_MARGIN_S,
    BandSpec,
    get_band_spec,
    t_settle_s,
)
from casestudies.impedance_stability.multitone_fft_bands.singletone_specs import (
    SingleToneSpec,
)


def _load_run(log_dir: str, band_tag: str, axis_tag: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    vi_path = os.path.join(log_dir, f"uic_terminal_vi_multisine_{band_tag}_{axis_tag}.csv")
    tones_path = os.path.join(log_dir, f"injected_tones_{band_tag}_{axis_tag}.csv")
    if not os.path.isfile(vi_path):
        raise FileNotFoundError(f"Missing {vi_path}")
    if not os.path.isfile(tones_path):
        raise FileNotFoundError(f"Missing {tones_path}")

    df = pd.read_csv(vi_path)
    t = df["t"].to_numpy(dtype=float)
    tones = pd.read_csv(tones_path)
    freqs = pd.to_numeric(tones["f_Hz"], errors="coerce").to_numpy(dtype=float)
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    freqs = np.unique(freqs)
    if freqs.size == 0:
        raise ValueError(f"No valid f_Hz in {tones_path}")
    return t, df, freqs


def _series(df: pd.DataFrame, re_col: str, im_col: str) -> np.ndarray:
    return df[re_col].to_numpy(dtype=float) + 1j * df[im_col].to_numpy(dtype=float)


def _vt(df: pd.DataFrame) -> np.ndarray:
    return _series(df, "v_t_re_pu", "v_t_im_pu")


def _ipert(df: pd.DataFrame) -> np.ndarray:
    return _series(df, "i_pert_re_pu", "i_pert_im_pu")


def _ia_sys(df: pd.DataFrame) -> np.ndarray:
    if "i_a_sys_re_pu" in df.columns and "i_a_sys_im_pu" in df.columns:
        return _series(df, "i_a_sys_re_pu", "i_a_sys_im_pu")
    return _series(df, "i_a_re_pu", "i_a_im_pu")


def _analysis_window(
    t: np.ndarray, *, t_min: float, t_margin: float, t_win: float
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t_end = float(t[-1])
    t2 = t_end - float(t_margin)
    t1 = max(float(t_min), float(t2) - float(t_win))
    if t1 >= t2:
        raise ValueError(f"Invalid analysis window: t_min={t1} >= t_end - margin={t2}")
    return (t >= t1) & (t <= t2)


def _fft_bin_pair(
    t: np.ndarray,
    x: np.ndarray,
    f_hz: float,
    *,
    t_min: float,
    t_margin: float,
    t_win: float,
) -> tuple[complex, complex, dict[str, Any]]:
    nan = np.nan + 1j * np.nan
    m = _analysis_window(t, t_min=t_min, t_margin=t_margin, t_win=t_win)
    seg = np.asarray(x[m], dtype=complex)
    if seg.size < 32:
        return nan, nan, {"n": int(seg.size), "k": np.nan, "df_hz": np.nan}

    tt = np.asarray(t[m], dtype=float)
    dt = float(np.median(np.diff(tt))) if tt.size > 2 else DT_S
    if not np.isfinite(dt) or dt <= 0:
        dt = DT_S

    seg = seg - np.mean(seg)
    w = np.hanning(seg.size)
    norm = float(np.sum(w))
    X = np.fft.fft(seg * w) / (norm if norm > 0 else seg.size)
    n = int(seg.size)

    k = int(round(float(f_hz) * n * dt))
    if k <= 0 or k >= n // 2:
        return nan, nan, {"n": n, "k": k, "df_hz": (1.0 / (n * dt))}

    df_hz = 1.0 / (n * dt)
    bin_f = float(k) * df_hz
    info = {"n": n, "k": k, "df_hz": df_hz, "bin_f_hz": bin_f, "bin_err_hz": float(bin_f - f_hz)}
    return complex(X[k]), complex(X[n - k]), info


def run_fft_multitone_mimo_band(
    log_dir: str,
    band: str,
    *,
    out_dir: str | None = None,
    spec: BandSpec | SingleToneSpec | None = None,
    plant: PlantId = "wt",
) -> str:
    """FFT bin-pick MIMO ID for one band (re + im logs with band prefix)."""
    spec = spec or get_band_spec(band)
    band_tag = str(spec.band)
    out_dir = out_dir or log_dir
    os.makedirs(out_dir, exist_ok=True)

    t_re, df_re, freqs_re = _load_run(log_dir, band_tag, "re")
    t_im, df_im, freqs_im = _load_run(log_dir, band_tag, "im")
    if t_re.shape != t_im.shape or np.max(np.abs(t_re - t_im)) > 1e-6:
        raise ValueError("re and im runs must share the same time base.")
    if freqs_re.size != freqs_im.size or np.max(np.abs(freqs_re - freqs_im)) > 1e-12:
        raise ValueError("Injected frequency lists differ between re and im.")
    freqs = freqs_re

    vt_re, vt_im = _vt(df_re), _vt(df_im)
    ip_re, ip_im = _ipert(df_re), _ipert(df_im)
    ia_re, ia_im = _ia_sys(df_re), _ia_sys(df_im)

    n = int(freqs.size)
    nan_c = np.nan + 1j * np.nan

    def _buf() -> np.ndarray:
        return np.full(n, nan_c, dtype=complex)

    Vp_d, Vm_d, Vp_q, Vm_q = _buf(), _buf(), _buf(), _buf()
    Ip_p_d, Ip_m_d, Ip_p_q, Ip_m_q = _buf(), _buf(), _buf(), _buf()
    Ia_p_d, Ia_m_d, Ia_p_q, Ia_m_q = _buf(), _buf(), _buf(), _buf()
    bin_err = np.full(n, np.nan, dtype=float)
    df_fft_used = np.full(n, np.nan, dtype=float)

    t_min = float(t_settle_s(plant))
    t_margin = float(T_MARGIN_S)
    t_win = float(spec.t_win_s)

    for i, f in enumerate(freqs):
        Vp_d[i], Vm_d[i], info = _fft_bin_pair(
            t_re, vt_re, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win
        )
        df_fft_used[i] = float(info.get("df_hz") or np.nan)
        bin_err[i] = float(info.get("bin_err_hz") or np.nan)
        Vp_q[i], Vm_q[i], _ = _fft_bin_pair(t_im, vt_im, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win)
        Ip_p_d[i], Ip_m_d[i], _ = _fft_bin_pair(t_re, ip_re, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win)
        Ip_p_q[i], Ip_m_q[i], _ = _fft_bin_pair(t_im, ip_im, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win)
        Ia_p_d[i], Ia_m_d[i], _ = _fft_bin_pair(t_re, ia_re, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win)
        Ia_p_q[i], Ia_m_q[i], _ = _fft_bin_pair(t_im, ia_im, float(f), t_min=t_min, t_margin=t_margin, t_win=t_win)

    res = _mimo_impedance_at_freqs(
        freqs,
        Vp_d,
        Vm_d,
        Vp_q,
        Vm_q,
        Ip_p_d,
        Ip_m_d,
        Ip_p_q,
        Ip_m_q,
        Ia_p_d,
        Ia_m_d,
        Ia_p_q,
        Ia_m_q,
        mimo_reg_det_rel=0.0,
    )

    matrix_csv = f"impedance_matrix_fft_{band_tag}.csv"
    save_mimo_impedance_outputs(
        freqs,
        res,
        out_dir,
        mark_hz=None,
        matrix_csv=matrix_csv,
        plot_suffix=f"_multitone_fft_{band_tag}",
        sequence_title_tag=f"· multitone FFT {band_tag.upper()}",
        plot_f_min_hz=float(spec.f_min_hz),
    )

    out_csv = os.path.join(out_dir, matrix_csv)
    out = pd.read_csv(out_csv)
    out["id_method"] = "multitone_fft_bands_binpick"
    out["band"] = band_tag
    out["t_min_s"] = t_min
    out["t_margin_s"] = t_margin
    out["t_win_s"] = t_win
    out["df_fft_used_hz"] = df_fft_used
    out["tone_bin_err_hz"] = bin_err
    out.to_csv(out_csv, index=False)

    max_err = float(np.nanmax(np.abs(bin_err))) if np.any(np.isfinite(bin_err)) else np.nan
    lim = 0.5 * spec.df_fft_hz
    if np.isfinite(max_err) and max_err > lim:
        print(
            f"WARNING [{band_tag}]: max |tone_bin_err|={max_err:.4g} Hz > 0.5*df_fft={lim:.4g}",
            flush=True,
        )

    return out_csv
