"""Log/output paths for band-split multitone impedance identification."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT: Path = _PACKAGE_DIR.parents[1]
LOGS_ROOT: Path = _PACKAGE_DIR / "logs"

# Band-split multitone (LF / MF / HF)
WT_UIC_MULTITONE_FFT_BANDS_DIR: Path = LOGS_ROOT / "wt_uic_multitone_fft_bands"
FMU_UIC_MULTITONE_FFT_BANDS_DIR: Path = LOGS_ROOT / "fmu_uic_multitone_fft_bands"

# Single-tone cross-checks
WT_UIC_MULTITONE_FFT_SINGLETONE_DIR: Path = LOGS_ROOT / "wt_uic_multitone_fft_singletone"
FMU_UIC_MULTITONE_FFT_SINGLETONE_DIR: Path = LOGS_ROOT / "fmu_uic_multitone_fft_singletone"

# WT vs FMU overlay from plot_wt_fmu_overlay.py
THESIS_MULTITONE_FFT_BANDS_COMPARE_DIR: Path = LOGS_ROOT / "thesis_multitone_fft_bands_compare"
