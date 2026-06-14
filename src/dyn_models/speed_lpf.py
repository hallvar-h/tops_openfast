"""ROSCO-style speed LPF helpers for grid–WT / grid–FMU coupling"""

from __future__ import annotations

import numpy as np


def resolve_speed_lpf_params(par, n_units: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read speed_lpf_* from a structured parameter array."""
    if n_units is None:
        n_units = int(np.asarray(par["S_n"]).size)
    par_names = getattr(getattr(par, "dtype", None), "names", None) or ()

    def _arr(key: str, default: float) -> np.ndarray:
        if key in par_names:
            return np.asarray(par[key]).ravel()
        return np.full(n_units, default)

    return (
        _arr("speed_lpf_type", 2.0),
        _arr("speed_lpf_corner_rad_s", 1.00810),
        _arr("speed_lpf_damping", 0.70000),
    )


def speed_pu_for_use(X, raw_key: str, filt_key: str, lpf_type: int) -> float:
    """LPF output in pu, or raw speed when speed_lpf_type == 0 (no filter)."""
    raw = float(np.asarray(X[raw_key]).ravel()[0])
    if lpf_type == 0:
        return raw
    return float(np.asarray(X[filt_key]).ravel()[0])


def apply_speed_lpf_dynamics(
    dX,
    X,
    u_pu,
    filt_key: str,
    filt_dot_key: str,
    lpf_type: int,
    omega_c: float,
    zeta: float,
) -> None:
    """2nd/1st-order speed LPF on u_pu (pu); states filt_key / filt_dot_key (pu, pu/s)."""
    u = np.asarray(u_pu, dtype=float).ravel()
    x1 = np.asarray(X[filt_key], dtype=float).ravel()
    x2 = np.asarray(X[filt_dot_key], dtype=float).ravel()
    if lpf_type == 2:
        dX[filt_key] = x2
        dX[filt_dot_key] = (omega_c**2) * (u - x1) - (2.0 * zeta * omega_c) * x2
    elif lpf_type == 1:
        dX[filt_key] = omega_c * (u - x1)
        dX[filt_dot_key] = np.zeros_like(u)
    else:
        dX[filt_key] = np.zeros_like(u)
        dX[filt_dot_key] = np.zeros_like(u)
