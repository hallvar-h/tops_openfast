"""Shared UIC / WindTurbine / FMUtoUICdrivetrain parameters.

Single source aligned with ``test_WT.py`` and ``test_WT_FMU_drivetrain_.py``.
Impedance-stability PS builders import from here to avoid drift.
"""

from __future__ import annotations

import os
from typing import Any

GEN_HEADER: list[str] = [
    "name",
    "bus",
    "S_n",
    "V_n",
    "P",
    "V",
    "H",
    "D",
    "X_d",
    "X_q",
    "X_d_t",
    "X_q_t",
    "X_d_st",
    "X_q_st",
    "T_d0_t",
    "T_q0_t",
    "T_d0_st",
    "T_q0_st",
]

INFINITE_BUS_ROW: list[Any] = [
    "IB",
    "B1",
    10e8,
    22,
    0,
    1,
    1e5,
    0,
    1.05,
    0.66,
    0.328,
    0.66,
    1e-5,
    1e-5,
    1e5,
    10000,
    1e5,
    1e5,
]

THREE_BUS_BUSES: list[list[Any]] = [
    ["name", "V_n"],
    ["B1", 22],
    ["B2", 22],
    ["B3", 22],
]

THREE_BUS_LINES: list[list[Any]] = [
    ["name", "from_bus", "to_bus", "length", "S_n", "V_n", "unit", "R", "X", "B"],
    ["L1-2", "B1", "B2", 25, 10, 22, "PF", 1e-5, 1e-4, 0.0],
    ["L2-3", "B2", "B3", 25, 10, 22, "PF", 1e-5, 1e-4, 0.0],
]

THREE_BUS_LOADS: list[list[Any]] = [
    ["name", "bus", "P", "Q", "model"],
    ["L1", "B3", 20, 5, "Z"],
]

UIC_SIG_HEADER: list[str] = [
    "name",
    "bus",
    "S_n",
    "V_n",
    "v_ref",
    "p_ref",
    "q_ref",
    "Ki",
    "Kv",
    "xf",
    "perfect_tracking",
    "T_filter",
]

UIC_SIG_ROW: list[Any] = ["UIC1", "B2", 20, 22, 1.0, 0.5, 0.0, 0.03, 0.0, 0.1, 1, 0.1]

WIND_TURBINE_HEADER: list[str] = [
    "name",
    "UIC",
    "S_n",
    "V_n",
    "J_m",
    "J_e",
    "K",
    "D",
    "Kp_pitch",
    "Ki_pitch",
    "T_pitch",
    "max_pitch",
    "min_pitch",
    "max_pitch_rate",
    "rho",
    "R",
    "P_rated",
    "omega_m_rated",
    "wind_rated",
    "efficiency",
    "MPT_filename",
    "Cp_filename",
    "speed_lpf_type",
    "speed_lpf_corner_rad_s",
    "speed_lpf_damping",
]

# test_WT.py WindTurbine row (ROSCO pitch: Kp 0.6738 rad/pu, Ki 0.06, T_pitch 2.2 s, …)
WIND_TURBINE_ROW: list[Any] = [
    "WT1",
    "UIC1",
    15,
    22,
    352460500.0,
    1836784.0,
    69737644900.0 / 100.0,
    35698200.0 / 10.0,
    0.6738,
    0.06,
    2.2,
    30.0,
    0.0,
    10.0,
    1.225,
    120.97,
    1.0,
    7.559987120819503,
    10.6,
    0.95756,
    "MPT_Kopt2150.csv",
    "Cp_Ct_Cq.IEA15MW.ROSCO.txt",
    2,
    1.00810,
    0.70000,
]

FMU_DRIVETRAIN_HEADER: list[str] = [
    "name",
    "UIC",
    "S_n",
    "V_n",
    "FMU_path",
    "fmu_filename",
    "control_mode",
    "wd_path",
    "openfast_test_dir",
    "testNr",
    "J_m",
    "J_e",
    "K",
    "D",
    "omega_m_rated",
    "fmu_dt",
    "ElecPwrCom_kW",
    "efficiency",
    "speed_lpf_type",
    "speed_lpf_corner_rad_s",
    "speed_lpf_damping",
]


def fmu_drivetrain_row(*, fmu_path: str, project_root: str) -> list[Any]:
    """Data row for ``test_WT_FMU_drivetrain_.py`` FMUtoUICdrivetrain block."""
    root = str(project_root)
    return [
        "FMUtoUICdrivetrain1",
        "UIC1",
        15,
        22,
        fmu_path,
        "fast.fmu",
        3,
        os.path.join(root, "openfast_fmu", "resources", "wd.txt"),
        root,
        1002,
        352460500.0,
        1836784.0,
        69737644900.0 / 100.0,
        35698200.0 / 10.0,
        7.559987120819503,
        0.01,
        20000.0,
        0.95756,
        2,
        1.00810,
        0.70000,
    ]


def three_bus_network_core() -> dict[str, Any]:
    """Buses, lines, loads, infinite-bus generator (three-bus template)."""
    return {
        "base_mva": 10,
        "f": 50,
        "slack_bus": "B1",
        "buses": [list(r) for r in THREE_BUS_BUSES],
        "lines": [list(r) for r in THREE_BUS_LINES],
        "loads": [list(r) for r in THREE_BUS_LOADS],
        "generators": {"GEN": [list(GEN_HEADER), list(INFINITE_BUS_ROW)]},
    }


def uic_sig_vsc_block() -> dict[str, list[list[Any]]]:
    return {"UIC_sig": [list(UIC_SIG_HEADER), list(UIC_SIG_ROW)]}


def uic_sig_pref_vsc_block() -> dict[str, list[list[Any]]]:
    return {"UIC_sigPref": [list(UIC_SIG_HEADER), list(UIC_SIG_ROW)]}


def wind_turbine_block() -> dict[str, list[list[Any]]]:
    return {"WindTurbine": [list(WIND_TURBINE_HEADER), list(WIND_TURBINE_ROW)]}


def fmu_drivetrain_block(*, fmu_path: str, project_root: str) -> dict[str, list[list[Any]]]:
    return {
        "FMUtoUICdrivetrain": [
            list(FMU_DRIVETRAIN_HEADER),
            fmu_drivetrain_row(fmu_path=fmu_path, project_root=project_root),
        ],
    }
