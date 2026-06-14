"""FMU co-simulation logging helpers (aligned with drivetrain multisine runners)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fmu_model_from_ps(ps) -> Any | None:
    for mdl in ps.dyn_mdls:
        if hasattr(mdl, "step_fmu"):
            return mdl
    return None


def seed_fmu_output_row(fmu_mdl: Any, t: float) -> dict[str, float]:
    """t=0 row: NaN outputs (no valid post-step FMU sample yet)."""
    if hasattr(fmu_mdl, "FMU_OUTPUT_NAMES"):
        row = {
            name: np.nan
            for name in getattr(fmu_mdl, "FMU_OUTPUT_NAMES", [])
            if name in getattr(fmu_mdl, "vrs", {})
        }
    else:
        row = {}
    row["Time"] = float(t)
    row["Time_fmu"] = np.nan
    return row


def append_fmu_output_row(fmu_mdl: Any, t: float) -> dict[str, float]:
    """Post-step FMU outputs latched at TOPS time ``t``."""
    if hasattr(fmu_mdl, "get_all_fmu_outputs"):
        d = dict(fmu_mdl.get_all_fmu_outputs())
        if "Time" in d:
            d["Time_fmu"] = d.get("Time")
            d["Time"] = float(t)
        else:
            d["Time"] = float(t)
        return d
    return {"Time": float(t), "Time_fmu": np.nan}


def append_fmu_command_row(fmu_mdl: Any | None) -> dict[str, float]:
    """Wrapper command/debug scalars (one row per time step)."""
    if fmu_mdl is None:
        return {
            "Te_cmd_pu": np.nan,
            "Te_cmd_kNm": np.nan,
            "GenSpdOrTrq_set_kNm": np.nan,
            "omega_m_pu_meas": np.nan,
            "GenPwr_set_kW": np.nan,
            "ElecPwrCom_set_kW": np.nan,
        }
    te_pu = (
        float(np.asarray(fmu_mdl._Te_pu_cmd).ravel()[0])
        if getattr(fmu_mdl, "_Te_pu_cmd", None) is not None
        and np.asarray(fmu_mdl._Te_pu_cmd).size > 0
        else np.nan
    )
    te_knm = np.nan
    if hasattr(fmu_mdl, "_T_base_Nm") and np.isfinite(te_pu):
        te_knm = te_pu * float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0]) / 1e3
    return {
        "Te_cmd_pu": te_pu,
        "Te_cmd_kNm": te_knm,
        "GenSpdOrTrq_set_kNm": float(getattr(fmu_mdl, "_gen_spdortrq_kNm_set", np.nan)),
        "omega_m_pu_meas": float(getattr(fmu_mdl, "_omega_m_pu_meas", np.nan)),
        "GenPwr_set_kW": float(getattr(fmu_mdl, "_genpwr_kW_set", np.nan)),
        "ElecPwrCom_set_kW": float(getattr(fmu_mdl, "_elec_pwr_com_kW_last", np.nan)),
    }


def fmu_diag_dataframe(
    t: np.ndarray,
    fmu_rows: list[dict[str, float]],
    cmd_rows: list[dict[str, float]],
) -> pd.DataFrame | None:
    if not fmu_rows:
        return None
    n = int(len(t))
    if len(fmu_rows) != n or (cmd_rows and len(cmd_rows) != n):
        raise ValueError(
            f"FMU log length mismatch: t={n}, fmu={len(fmu_rows)}, cmd={len(cmd_rows)}"
        )
    df = pd.DataFrame({"t": np.asarray(t, dtype=float)})
    df_fmu = pd.DataFrame(fmu_rows)
    df_cmd = pd.DataFrame(cmd_rows) if cmd_rows else pd.DataFrame()
    return pd.concat([df, df_fmu, df_cmd], axis=1)


def injected_tones_csv_name(band_tag: str | None, axis: str) -> str:
    if band_tag and str(band_tag).strip():
        return f"injected_tones_{band_tag}_{axis}.csv"
    return f"injected_tones_{axis}.csv"


def vi_csv_name(band_tag: str | None, axis: str) -> str:
    if band_tag and str(band_tag).strip():
        return f"uic_terminal_vi_multisine_{band_tag}_{axis}.csv"
    return f"uic_terminal_vi_multisine_{axis}.csv"


def fmu_diag_csv_name(band_tag: str | None, axis: str) -> str:
    if band_tag and str(band_tag).strip():
        return f"fmu_diag_{band_tag}_{axis}.csv"
    return f"fmu_diag_{axis}.csv"
