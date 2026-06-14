"""Shared three-bus UIC + algebraic WT template (B1–B2–B3) for multisine / tone34 cases."""

from __future__ import annotations

from typing import Any

from casestudies.ps_data.wt_uic_fmu_shared import (
    three_bus_network_core,
    uic_sig_vsc_block,
    wind_turbine_block,
)


def uic_wt_three_bus_model(*, perturbation_rows: list[list[Any]]) -> dict[str, Any]:
    """Build PS dict with ``BusCurrentPerturbation`` rows (header + data)."""
    model = three_bus_network_core()
    model["vsc"] = uic_sig_vsc_block()
    model["windturbine"] = wind_turbine_block()
    model["perturbations"] = {"BusCurrentPerturbation": perturbation_rows}
    return model
