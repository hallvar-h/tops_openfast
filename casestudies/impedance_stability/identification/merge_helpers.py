"""Shared helpers for merging band-split impedance CSVs (multitone LF/MF/HF)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OVERLAP_PHASE_WARN_DEG: float = 20.0
OVERLAP_MAG_RATIO_WARN: float = 1.25

_COMPLEX_PAIRS: list[tuple[str, str]] = [
    ("Zbus_dd_re", "Zbus_dd_im"),
    ("Zbus_dq_re", "Zbus_dq_im"),
    ("Zbus_qd_re", "Zbus_qd_im"),
    ("Zbus_qq_re", "Zbus_qq_im"),
    ("Ydev_dd_re", "Ydev_dd_im"),
    ("Ydev_dq_re", "Ydev_dq_im"),
    ("Ydev_qd_re", "Ydev_qd_im"),
    ("Ydev_qq_re", "Ydev_qq_im"),
    ("Zdev_dd_re", "Zdev_dd_im"),
    ("Zdev_dq_re", "Zdev_dq_im"),
    ("Zdev_qd_re", "Zdev_qd_im"),
    ("Zdev_qq_re", "Zdev_qq_im"),
    ("Zbus_plus_re", "Zbus_plus_im"),
    ("Zbus_minus_re", "Zbus_minus_im"),
    ("Zbus_m00_re", "Zbus_m00_im"),
    ("Zbus_m01_re", "Zbus_m01_im"),
    ("Zbus_m10_re", "Zbus_m10_im"),
    ("Zbus_m11_re", "Zbus_m11_im"),
    ("Ydev_plus_re", "Ydev_plus_im"),
    ("Ydev_minus_re", "Ydev_minus_im"),
    ("Ydev_m00_re", "Ydev_m00_im"),
    ("Ydev_m01_re", "Ydev_m01_im"),
    ("Ydev_m10_re", "Ydev_m10_im"),
    ("Ydev_m11_re", "Ydev_m11_im"),
    ("Zdev_plus_re", "Zdev_plus_im"),
    ("Zdev_minus_re", "Zdev_minus_im"),
    ("Zdev_m00_re", "Zdev_m00_im"),
    ("Zdev_m01_re", "Zdev_m01_im"),
    ("Zdev_m10_re", "Zdev_m10_im"),
    ("Zdev_m11_re", "Zdev_m11_im"),
]

_REAL_COLS = ("detIpert_abs", "detVt_abs", "detYdev_abs", "kappa_Ydev", "detYdev_norm")


def _interp_complex(
    f_src: np.ndarray,
    z_src: np.ndarray,
    f_tgt: np.ndarray,
    *,
    extrapolate: bool = False,
) -> np.ndarray:
    f_src = np.asarray(f_src, dtype=float)
    z_src = np.asarray(z_src, dtype=complex)
    f_tgt = np.asarray(f_tgt, dtype=float)
    out = np.full(f_tgt.shape, np.nan + 1j * np.nan, dtype=complex)
    m = np.isfinite(f_src) & np.isfinite(np.real(z_src)) & np.isfinite(np.imag(z_src))
    if np.sum(m) < 1:
        return out
    order = np.argsort(f_src[m])
    fs = f_src[m][order]
    zs = z_src[m][order]
    if np.sum(m) == 1:
        out[np.isfinite(f_tgt)] = zs[0]
        return out
    left_re = zs.real[0] if extrapolate else np.nan
    right_re = zs.real[-1] if extrapolate else np.nan
    left_im = zs.imag[0] if extrapolate else np.nan
    right_im = zs.imag[-1] if extrapolate else np.nan
    ft = f_tgt[np.isfinite(f_tgt)]
    out[np.isfinite(f_tgt)] = np.interp(ft, fs, zs.real, left=left_re, right=right_re) + 1j * np.interp(
        ft, fs, zs.imag, left=left_im, right=right_im
    )
    return out


def _interp_real(f_src: np.ndarray, y_src: np.ndarray, f_tgt: np.ndarray) -> np.ndarray:
    f_src = np.asarray(f_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    m = np.isfinite(f_src) & np.isfinite(y_src)
    if np.sum(m) < 2:
        return np.full(f_tgt.shape, np.nan, dtype=float)
    order = np.argsort(f_src[m])
    return np.interp(f_tgt, f_src[m][order], y_src[m][order], left=np.nan, right=np.nan)


def _phase_delta_deg(z_lf: np.ndarray, z_hf: np.ndarray) -> np.ndarray:
    dph = np.degrees(np.angle(z_hf * np.conj(z_lf)))
    return (dph + 180.0) % 360.0 - 180.0


def _overlap_agreement_qa(
    f_lf: np.ndarray,
    z_lf: np.ndarray,
    f_hf: np.ndarray,
    z_hf: np.ndarray,
    *,
    ov_lo: float,
    ov_hi: float,
    n_check: int = 21,
    phase_warn_deg: float = OVERLAP_PHASE_WARN_DEG,
    mag_ratio_warn: float = OVERLAP_MAG_RATIO_WARN,
) -> dict[str, Any]:
    """Compare two band curves on a dense grid inside the measured overlap."""
    f_ov = np.linspace(float(ov_lo), float(ov_hi), int(n_check), dtype=float)
    zl = _interp_complex(f_lf, z_lf, f_ov, extrapolate=False)
    zh = _interp_complex(f_hf, z_hf, f_ov, extrapolate=False)
    m = np.isfinite(zl) & np.isfinite(zh)
    n_both = int(np.sum(m))
    out: dict[str, Any] = {
        "measured_overlap_Hz": [float(ov_lo), float(ov_hi)],
        "n_overlap_check": int(n_check),
        "n_both_finite": n_both,
        "overlap_coverage_ok": n_both == int(n_check),
    }
    if n_both < 2:
        out.update(
            {
                "max_abs_phase_diff_deg": None,
                "max_mag_ratio": None,
                "warn": True,
                "error": "insufficient_overlap_measurements",
            }
        )
        return out
    dph = _phase_delta_deg(zl[m], zh[m])
    mag_r = np.abs(zh[m]) / np.maximum(np.abs(zl[m]), 1e-30)
    max_dph = float(np.max(np.abs(dph)))
    max_mr = float(np.max(np.maximum(mag_r, 1.0 / mag_r)))
    out.update(
        {
            "max_abs_phase_diff_deg": max_dph,
            "max_mag_ratio": max_mr,
            "warn": max_dph > phase_warn_deg or max_mr > mag_ratio_warn,
        }
    )
    return out


def _dataframe_to_res(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def _c(re_col: str, im_col: str) -> np.ndarray:
        return df[re_col].to_numpy(dtype=float) + 1j * df[im_col].to_numpy(dtype=float)

    out: dict[str, np.ndarray] = {
        "Zbus_dd": _c("Zbus_dd_re", "Zbus_dd_im"),
        "Zbus_dq": _c("Zbus_dq_re", "Zbus_dq_im"),
        "Zbus_qd": _c("Zbus_qd_re", "Zbus_qd_im"),
        "Zbus_qq": _c("Zbus_qq_re", "Zbus_qq_im"),
        "Ydev_dd": _c("Ydev_dd_re", "Ydev_dd_im"),
        "Ydev_dq": _c("Ydev_dq_re", "Ydev_dq_im"),
        "Ydev_qd": _c("Ydev_qd_re", "Ydev_qd_im"),
        "Ydev_qq": _c("Ydev_qq_re", "Ydev_qq_im"),
        "Zdev_dd": _c("Zdev_dd_re", "Zdev_dd_im"),
        "Zdev_dq": _c("Zdev_dq_re", "Zdev_dq_im"),
        "Zdev_qd": _c("Zdev_qd_re", "Zdev_qd_im"),
        "Zdev_qq": _c("Zdev_qq_re", "Zdev_qq_im"),
        "detIpert_abs": df["detIpert_abs"].to_numpy(dtype=float),
        "detVt_abs": df["detVt_abs"].to_numpy(dtype=float),
        "detYdev_abs": df["detYdev_abs"].to_numpy(dtype=float),
        "Zplus_bus": _c("Zbus_plus_re", "Zbus_plus_im"),
        "Zminus_bus": _c("Zbus_minus_re", "Zbus_minus_im"),
        "Zm_bus_00": _c("Zbus_m00_re", "Zbus_m00_im"),
        "Zm_bus_01": _c("Zbus_m01_re", "Zbus_m01_im"),
        "Zm_bus_10": _c("Zbus_m10_re", "Zbus_m10_im"),
        "Zm_bus_11": _c("Zbus_m11_re", "Zbus_m11_im"),
        "Yplus_dev": _c("Ydev_plus_re", "Ydev_plus_im"),
        "Yminus_dev": _c("Ydev_minus_re", "Ydev_minus_im"),
        "Ym_dev_00": _c("Ydev_m00_re", "Ydev_m00_im"),
        "Ym_dev_01": _c("Ydev_m01_re", "Ydev_m01_im"),
        "Ym_dev_10": _c("Ydev_m10_re", "Ydev_m10_im"),
        "Ym_dev_11": _c("Ydev_m11_re", "Ydev_m11_im"),
        "Zplus_dev": _c("Zdev_plus_re", "Zdev_plus_im"),
        "Zminus_dev": _c("Zdev_minus_re", "Zdev_minus_im"),
        "Zm_dev_00": _c("Zdev_m00_re", "Zdev_m00_im"),
        "Zm_dev_01": _c("Zdev_m01_re", "Zdev_m01_im"),
        "Zm_dev_10": _c("Zdev_m10_re", "Zdev_m10_im"),
        "Zm_dev_11": _c("Zdev_m11_re", "Zdev_m11_im"),
    }
    if "kappa_Ydev" in df.columns:
        out["kappa_Ydev"] = df["kappa_Ydev"].to_numpy(dtype=float)
    if "detYdev_norm" in df.columns:
        out["detYdev_norm"] = df["detYdev_norm"].to_numpy(dtype=float)
    return out
