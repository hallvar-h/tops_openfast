"""
Single-tone current injection for band-split cross-checks.

Used by ``run_singletone.py`` (and optional CaseLoader ids ``multitone_singletone_*``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from casestudies.impedance_stability.multitone_fft_bands.band_specs import RANDOM_SEED, PlantId
from casestudies.impedance_stability.multitone_fft_bands.singletone_specs import (
    SingleToneSpec,
    singletone_spec_for_frequency,
)
from casestudies.impedance_stability.paths import REPO_ROOT
from casestudies.impedance_stability.ps_data.fmu_openfast import _fmu_path, load_fmu_multisine
from casestudies.impedance_stability.ps_data.uic_wt_three_bus import uic_wt_three_bus_model


def _build_singletone_rows(
    *,
    axis: str,
    spec: SingleToneSpec,
    random_seed: int = RANDOM_SEED,
) -> list[list[Any]]:
    axis = str(axis).strip().lower()
    phase = float((2.0 * np.pi) * np.random.RandomState(int(random_seed)).rand())
    return [
        ["name", "bus", "I_re", "I_im", "amp", "f", "phase", "axis"],
        [
            "Ipert_0000",
            "B2",
            0.0,
            0.0,
            float(spec.amp_per_tone_pu),
            float(spec.f_hz),
            phase,
            axis,
        ],
    ]


def load_multitone_fft_singletone(
    *,
    plant: PlantId,
    axis: str,
    f_hz: float,
    amp_per_tone_pu: float | None = None,
    t_win_s: float | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """WT or FMU with one injected tone at ``f_hz`` (snapped to FFT bin)."""
    spec = singletone_spec_for_frequency(
        f_hz, amp_per_tone_pu=amp_per_tone_pu, t_win_s=t_win_s
    )
    axis = str(axis).strip().lower()
    header = _build_singletone_rows(axis=axis, spec=spec, random_seed=int(random_seed))

    if plant == "wt":
        return uic_wt_three_bus_model(perturbation_rows=header)

    root = str(REPO_ROOT)
    fmu_path = _fmu_path()
    model = load_fmu_multisine(axis=axis)
    model["perturbations"] = {"BusCurrentPerturbation": header}
    if "FMUtoUICdrivetrain" in model and "FMUtoUICdrivetrain" in model["FMUtoUICdrivetrain"]:
        row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"][1]
        row[4] = fmu_path
        row[7] = os.path.join(root, "openfast_fmu", "resources", "wd.txt")
        row[8] = root
    return model
