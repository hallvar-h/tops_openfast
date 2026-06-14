"""
Rerun selected band-split multitone bands (sim + FFT ID + merge + plots).

Examples (repo root):
  python casestudies/impedance_stability/multitone_fft_bands/run_rerun_bands.py --plant fmu --bands lf
  python casestudies/impedance_stability/multitone_fft_bands/run_rerun_bands.py --plant wt --bands lf
"""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from casestudies.impedance_stability.multitone_fft_bands.pipeline import run_multitone_bands_pipeline
from casestudies.impedance_stability.paths import (
    FMU_UIC_MULTITONE_FFT_BANDS_DIR,
    WT_UIC_MULTITONE_FFT_BANDS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun selected multitone FFT bands.")
    parser.add_argument("--plant", choices=("wt", "fmu"), required=True)
    parser.add_argument(
        "--bands",
        required=True,
        help="Comma-separated band ids, e.g. lf or lf,mf",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip thesis PNG generation")
    args = parser.parse_args()

    bands = tuple(b.strip().lower() for b in args.bands.split(",") if b.strip())
    if not bands:
        raise SystemExit("--bands must list at least one band")

    if args.plant == "fmu":
        log_dir = str(FMU_UIC_MULTITONE_FFT_BANDS_DIR)
        case_prefix = "fmu_multitone_bands"
        label = "FMU multitone FFT bands"
    else:
        log_dir = str(WT_UIC_MULTITONE_FFT_BANDS_DIR)
        case_prefix = "multitone_bands"
        label = "WT multitone FFT bands"

    run_multitone_bands_pipeline(
        plant=args.plant,
        log_dir=log_dir,
        case_prefix=case_prefix,
        label=label,
        bands=bands,
        run_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
