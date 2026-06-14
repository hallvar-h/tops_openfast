"""
Merge LF / MF / HF multitone FFT impedance CSVs into one 0.1–10 Hz result.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from casestudies.impedance_stability.identification.impedance_matrix_uic_wt import save_mimo_impedance_outputs
from casestudies.impedance_stability.identification.merge_helpers import (
    _COMPLEX_PAIRS,
    _REAL_COLS,
    _dataframe_to_res,
    _interp_complex,
    _interp_real,
    _overlap_agreement_qa,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    F_MAX_HZ,
    F_MIN_HZ,
    LF_MF_BLEND_HI_HZ,
    LF_MF_BLEND_LO_HZ,
    MF_HF_BLEND_HI_HZ,
    MF_HF_BLEND_LO_HZ,
)

MERGED_CSV = "impedance_matrix_fft_merged.csv"


def _blend_weight_ramp(f_hz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 below lo, 1 at/above hi, linear between."""
    f = np.asarray(f_hz, dtype=float)
    w = (f - float(lo)) / max(float(hi) - float(lo), 1e-12)
    return np.clip(w, 0.0, 1.0)


def _z_at_freqs(df: pd.DataFrame, re_col: str, im_col: str, freqs: np.ndarray) -> np.ndarray:
    f_src = df["f_Hz"].to_numpy(dtype=float)
    z_src = df[re_col].to_numpy(dtype=float) + 1j * df[im_col].to_numpy(dtype=float)
    return _interp_complex(f_src, z_src, freqs, extrapolate=False)


def _y_at_freqs(df: pd.DataFrame, col: str, freqs: np.ndarray) -> np.ndarray:
    if col not in df.columns:
        return np.full(freqs.shape, np.nan, dtype=float)
    return _interp_real(df["f_Hz"].to_numpy(dtype=float), df[col].to_numpy(dtype=float), freqs)


def _merge_three_complex(
    f: np.ndarray,
    z_lf: np.ndarray,
    z_mf: np.ndarray,
    z_hf: np.ndarray,
) -> np.ndarray:
    w_mf = _blend_weight_ramp(f, LF_MF_BLEND_LO_HZ, LF_MF_BLEND_HI_HZ)
    w_hf = _blend_weight_ramp(f, MF_HF_BLEND_LO_HZ, MF_HF_BLEND_HI_HZ)

    z = np.full(f.shape, np.nan + 1j * np.nan, dtype=complex)
    low = f < LF_MF_BLEND_LO_HZ
    mid_lo = (f >= LF_MF_BLEND_LO_HZ) & (f < LF_MF_BLEND_HI_HZ)
    mid = (f >= LF_MF_BLEND_HI_HZ) & (f < MF_HF_BLEND_LO_HZ)
    mid_hi = (f >= MF_HF_BLEND_LO_HZ) & (f < MF_HF_BLEND_HI_HZ)
    high = f >= MF_HF_BLEND_HI_HZ

    z = np.where(low & np.isfinite(z_lf), z_lf, z)
    z = np.where(mid & np.isfinite(z_mf), z_mf, z)
    z = np.where(high & np.isfinite(z_hf), z_hf, z)

    both_lm = np.isfinite(z_lf) & np.isfinite(z_mf)
    z_lm = (1.0 - w_mf) * z_lf + w_mf * z_mf
    z = np.where(mid_lo & both_lm, z_lm, z)
    z = np.where(mid_lo & ~both_lm & np.isfinite(z_lf), z_lf, z)
    z = np.where(mid_lo & ~both_lm & np.isfinite(z_mf), z_mf, z)

    both_mh = np.isfinite(z_mf) & np.isfinite(z_hf)
    z_mh = (1.0 - w_hf) * z_mf + w_hf * z_hf
    z = np.where(mid_hi & both_mh, z_mh, z)
    z = np.where(mid_hi & ~both_mh & np.isfinite(z_mf), z_mf, z)
    z = np.where(mid_hi & ~both_mh & np.isfinite(z_hf), z_hf, z)

    return z


def _merge_real_three(
    f: np.ndarray,
    y_lf: np.ndarray,
    y_mf: np.ndarray,
    y_hf: np.ndarray,
) -> np.ndarray:
    w_mf = _blend_weight_ramp(f, LF_MF_BLEND_LO_HZ, LF_MF_BLEND_HI_HZ)
    w_hf = _blend_weight_ramp(f, MF_HF_BLEND_LO_HZ, MF_HF_BLEND_HI_HZ)
    y = np.full(f.shape, np.nan, dtype=float)
    low = f < LF_MF_BLEND_LO_HZ
    mid_lo = (f >= LF_MF_BLEND_LO_HZ) & (f < LF_MF_BLEND_HI_HZ)
    mid = (f >= LF_MF_BLEND_HI_HZ) & (f < MF_HF_BLEND_LO_HZ)
    mid_hi = (f >= MF_HF_BLEND_LO_HZ) & (f < MF_HF_BLEND_HI_HZ)
    high = f >= MF_HF_BLEND_HI_HZ

    y = np.where(low & np.isfinite(y_lf), y_lf, y)
    y = np.where(mid & np.isfinite(y_mf), y_mf, y)
    y = np.where(high & np.isfinite(y_hf), y_hf, y)

    both_lm = np.isfinite(y_lf) & np.isfinite(y_mf)
    y_blend = (1.0 - w_mf) * y_lf + w_mf * y_mf
    y = np.where(mid_lo & both_lm, y_blend, y)
    y = np.where(mid_lo & ~both_lm & np.isfinite(y_lf), y_lf, y)
    y = np.where(mid_lo & ~both_lm & np.isfinite(y_mf), y_mf, y)

    both_mh = np.isfinite(y_mf) & np.isfinite(y_hf)
    y_blend = (1.0 - w_hf) * y_mf + w_hf * y_hf
    y = np.where(mid_hi & both_mh, y_blend, y)
    y = np.where(mid_hi & ~both_mh & np.isfinite(y_mf), y_mf, y)
    y = np.where(mid_hi & ~both_mh & np.isfinite(y_hf), y_hf, y)

    return y


def _union_freqs(df_lf: pd.DataFrame, df_mf: pd.DataFrame, df_hf: pd.DataFrame) -> np.ndarray:
    parts = []
    for df in (df_lf, df_mf, df_hf):
        f = df["f_Hz"].to_numpy(dtype=float)
        parts.append(f[np.isfinite(f) & (f >= F_MIN_HZ) & (f <= F_MAX_HZ)])
    if not parts:
        raise ValueError("No frequencies to merge.")
    return np.unique(np.concatenate(parts))


def merge_three_band_csvs(
    log_dir: str,
    *,
    out_dir: str | None = None,
    matrix_csv: str = MERGED_CSV,
) -> dict[str, Any]:
    out_dir = out_dir or log_dir
    os.makedirs(out_dir, exist_ok=True)

    path_lf = os.path.join(log_dir, "impedance_matrix_fft_lf.csv")
    path_mf = os.path.join(log_dir, "impedance_matrix_fft_mf.csv")
    path_hf = os.path.join(log_dir, "impedance_matrix_fft_hf.csv")
    for p in (path_lf, path_mf, path_hf):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    df_lf = pd.read_csv(path_lf)
    df_mf = pd.read_csv(path_mf)
    df_hf = pd.read_csv(path_hf)
    freqs = _union_freqs(df_lf, df_mf, df_hf)

    merged: dict[str, np.ndarray] = {"f_Hz": freqs}
    for re_col, im_col in _COMPLEX_PAIRS:
        z_lf = _z_at_freqs(df_lf, re_col, im_col, freqs)
        z_mf = _z_at_freqs(df_mf, re_col, im_col, freqs)
        z_hf = _z_at_freqs(df_hf, re_col, im_col, freqs)
        z = _merge_three_complex(freqs, z_lf, z_mf, z_hf)
        merged[re_col] = np.real(z)
        merged[im_col] = np.imag(z)

    for col in _REAL_COLS:
        if col in df_lf.columns or col in df_mf.columns or col in df_hf.columns:
            y_lf = _y_at_freqs(df_lf, col, freqs)
            y_mf = _y_at_freqs(df_mf, col, freqs)
            y_hf = _y_at_freqs(df_hf, col, freqs)
            merged[col] = _merge_real_three(freqs, y_lf, y_mf, y_hf)

    out_df = pd.DataFrame(merged)
    out_df["id_method"] = "multitone_fft_bands_binpick"
    out_df["band"] = "merged"

    n_nan = 0
    for re_col, im_col in _COMPLEX_PAIRS:
        z = out_df[re_col].to_numpy(dtype=float) + 1j * out_df[im_col].to_numpy(dtype=float)
        n_nan += int(np.sum(~np.isfinite(z)))

    qa: dict[str, Any] = {
        "f_range_Hz": [F_MIN_HZ, F_MAX_HZ],
        "n_freq": int(freqs.size),
        "merged_nan_complex_samples": n_nan,
        "merged_complete": n_nan == 0,
        "lf_mf_blend_Hz": [LF_MF_BLEND_LO_HZ, LF_MF_BLEND_HI_HZ],
        "mf_hf_blend_Hz": [MF_HF_BLEND_LO_HZ, MF_HF_BLEND_HI_HZ],
    }

    zpp_lf = df_lf["Zbus_m00_re"].to_numpy(dtype=float) + 1j * df_lf["Zbus_m00_im"].to_numpy(dtype=float)
    zpp_mf = df_mf["Zbus_m00_re"].to_numpy(dtype=float) + 1j * df_mf["Zbus_m00_im"].to_numpy(dtype=float)
    zpp_hf = df_hf["Zbus_m00_re"].to_numpy(dtype=float) + 1j * df_hf["Zbus_m00_im"].to_numpy(dtype=float)
    f_lf = df_lf["f_Hz"].to_numpy(dtype=float)
    f_mf = df_mf["f_Hz"].to_numpy(dtype=float)
    f_hf = df_hf["f_Hz"].to_numpy(dtype=float)

    qa["lf_mf_overlap"] = _overlap_agreement_qa(
        f_lf, zpp_lf, f_mf, zpp_mf, ov_lo=LF_MF_BLEND_LO_HZ, ov_hi=LF_MF_BLEND_HI_HZ
    )
    qa["mf_hf_overlap"] = _overlap_agreement_qa(
        f_mf, zpp_mf, f_hf, zpp_hf, ov_lo=MF_HF_BLEND_LO_HZ, ov_hi=MF_HF_BLEND_HI_HZ
    )

    ypp_lf = df_lf["Ydev_m00_re"].to_numpy(dtype=float) + 1j * df_lf["Ydev_m00_im"].to_numpy(dtype=float)
    ypp_mf = df_mf["Ydev_m00_re"].to_numpy(dtype=float) + 1j * df_mf["Ydev_m00_im"].to_numpy(dtype=float)
    ypp_hf = df_hf["Ydev_m00_re"].to_numpy(dtype=float) + 1j * df_hf["Ydev_m00_im"].to_numpy(dtype=float)
    qa["lf_mf_overlap_Ypp"] = _overlap_agreement_qa(
        f_lf, ypp_lf, f_mf, ypp_mf, ov_lo=LF_MF_BLEND_LO_HZ, ov_hi=LF_MF_BLEND_HI_HZ
    )
    qa["mf_hf_overlap_Ypp"] = _overlap_agreement_qa(
        f_mf, ypp_mf, f_hf, ypp_hf, ov_lo=MF_HF_BLEND_LO_HZ, ov_hi=MF_HF_BLEND_HI_HZ
    )

    qa["warn"] = bool(
        not qa["merged_complete"]
        or qa["lf_mf_overlap"].get("warn")
        or qa["mf_hf_overlap"].get("warn")
    )

    out_path = os.path.join(out_dir, matrix_csv)
    out_df.to_csv(out_path, index=False)

    res = _dataframe_to_res(out_df)
    save_mimo_impedance_outputs(
        freqs,
        res,
        out_dir,
        mark_hz=None,
        matrix_csv=matrix_csv,
        plot_suffix="_multitone_fft_bands_merged",
        sequence_title_tag="· multitone FFT bands merged",
        plot_f_min_hz=F_MIN_HZ,
    )

    qa_path = os.path.join(out_dir, "merge_qa.json")
    with open(qa_path, "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=2, default=str)
    print(f"Wrote {out_path} and {qa_path}", flush=True)
    if qa.get("warn"):
        print(f"WARNING: merge QA flagged issues: {qa}", flush=True)
    else:
        print("Band-merge QA passed.", flush=True)

    return qa
