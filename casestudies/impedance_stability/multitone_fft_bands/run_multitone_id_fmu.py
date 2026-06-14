"""
FMU band-split multitone (0.1–10 Hz) impedance ID.

Run from repo root:
  python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_fmu.py
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from casestudies.impedance_stability.multitone_fft_bands.pipeline import run_multitone_bands_pipeline
from casestudies.impedance_stability.paths import FMU_UIC_MULTITONE_FFT_BANDS_DIR


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FMU band-split multitone FFT impedance ID.")
    parser.add_argument(
        "--no-sims",
        action="store_true",
        help="Skip time-domain runs; FFT ID, merge, and plots only (logs must exist).",
    )
    args = parser.parse_args()

    log_dir = str(FMU_UIC_MULTITONE_FFT_BANDS_DIR)
    run_multitone_bands_pipeline(
        plant="fmu",
        log_dir=log_dir,
        case_prefix="fmu_multitone_bands",
        label="FMU multitone FFT bands",
        run_sims=not args.no_sims,
    )


if __name__ == "__main__":
    main()
