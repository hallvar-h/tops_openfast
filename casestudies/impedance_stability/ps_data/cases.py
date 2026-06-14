"""Registry of UIC+WT / FMU band-split multitone impedance-identification cases."""

from __future__ import annotations

from typing import Any, Callable

from casestudies.impedance_stability.ps_data.multitone_fft_bands import (
    load_fmu_multitone_fft_band,
    load_wt_multitone_fft_band,
)

_REGISTRY: dict[str, Callable[[], dict[str, Any]]] = {}

for _band in ("lf", "mf", "hf"):
    for _axis in ("re", "im"):
        _REGISTRY[f"multitone_bands_{_band}_{_axis}"] = (
            lambda b=_band, a=_axis: load_wt_multitone_fft_band(band=b, axis=a)  # type: ignore[arg-type]
        )
        _REGISTRY[f"fmu_multitone_bands_{_band}_{_axis}"] = (
            lambda b=_band, a=_axis: load_fmu_multitone_fft_band(band=b, axis=a)  # type: ignore[arg-type]
        )


def load(case_id: str) -> dict[str, Any]:
    key = str(case_id).strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown case_id={case_id!r}; known cases: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


class CaseLoader:
    """Adapter used by simulation runners: ``CaseLoader(case_id).load()``."""

    __slots__ = ("case_id",)

    def __init__(self, case_id: str) -> None:
        self.case_id = str(case_id).strip()

    def load(self) -> dict[str, Any]:
        return load(self.case_id)
