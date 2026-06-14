"""OpenFAST FMU drivetrain block (replaces algebraic ``windturbine``)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from casestudies.impedance_stability.paths import REPO_ROOT
from casestudies.ps_data.wt_uic_fmu_shared import (
    fmu_drivetrain_block,
    three_bus_network_core,
    uic_sig_vsc_block,
)


def _fmu_path() -> str:
    for p in (REPO_ROOT / "OpenFAST" / "fast.fmu", REPO_ROOT / "fast.fmu"):
        if p.is_file():
            return str(p)
    return str(REPO_ROOT / "OpenFAST" / "fast.fmu")


def _multisine_pert_rows(*, axis: str, n_tones: int, target_total_rms: float) -> list[list[Any]]:
    freqs = np.linspace(0.1, 10.0, int(n_tones), dtype=float)
    amp = float(target_total_rms * np.sqrt(2.0 / float(n_tones)))
    phases = (2.0 * np.pi) * np.random.RandomState(1).rand(n_tones)
    rows: list[list[Any]] = []
    for k, (f_hz, ph) in enumerate(zip(freqs, phases)):
        rows.append([f"Ipert_{k:04d}", "B2", 0.0, 0.0, amp, float(f_hz), float(ph), axis])
    return rows


def load_fmu_multisine(*, axis: str = "re", n_tones: int = 100, target_total_rms: float = 0.01) -> dict:
    """FMU three-bus network template used by band-split and singletone loaders."""
    axis = str(axis).strip().lower()
    pert_rows = _multisine_pert_rows(axis=axis, n_tones=n_tones, target_total_rms=target_total_rms)
    root = str(REPO_ROOT)
    fmu_path = _fmu_path()
    model = three_bus_network_core()
    model["vsc"] = uic_sig_vsc_block()
    model["perturbations"] = {
        "BusCurrentPerturbation": [
            ["name", "bus", "I_re", "I_im", "amp", "f", "phase", "axis"],
            *pert_rows,
        ],
    }
    model["FMUtoUICdrivetrain"] = fmu_drivetrain_block(fmu_path=fmu_path, project_root=root)
    return model
