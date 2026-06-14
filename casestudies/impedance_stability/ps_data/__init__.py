"""Power-system builders for UIC+WT impedance identification."""

__all__ = ["CaseLoader", "load"]


def __getattr__(name: str):
    if name in __all__:
        from casestudies.impedance_stability.ps_data.cases import CaseLoader, load

        return CaseLoader if name == "CaseLoader" else load
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
