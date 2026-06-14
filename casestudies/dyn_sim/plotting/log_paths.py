"""Log/output paths for dyn_sim case studies (repo-relative)."""

from __future__ import annotations

from pathlib import Path

_PLOTTING_DIR = Path(__file__).resolve().parent
_DYN_SIM_DIR = _PLOTTING_DIR.parent
REPO_ROOT: Path = _DYN_SIM_DIR.parents[1]
LOGS_ROOT: Path = _DYN_SIM_DIR / "logs"

# Simulation result directories
WT_DIR: Path = LOGS_ROOT / "wt"
FMU_DRIVETRAIN_DIR: Path = LOGS_ROOT / "fmu_drivetrain"
FMU_UIC_DIR: Path = LOGS_ROOT / "fmu_uic"
COMPARE_DIR: Path = LOGS_ROOT / "compare"

# Primary CSV exports
WT_CSV: Path = WT_DIR / "wt_model.csv"
FMU_DRIVETRAIN_CSV: Path = FMU_DRIVETRAIN_DIR / "fmu_drivetrain.csv"
FMU_UIC_CSV: Path = FMU_UIC_DIR / "fmu_uic_sim.csv"


def ensure_log_dir(path: Path) -> Path:
    """Create parent directory for a log file or plot folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def wt_thesis_plots_dir() -> Path:
    return WT_DIR / "plots" / "thesis"


def fmu_drivetrain_thesis_plots_dir() -> Path:
    return FMU_DRIVETRAIN_DIR / "plots" / "thesis"


def compare_thesis_plots_dir() -> Path:
    return COMPARE_DIR / "plots" / "thesis"
