"""
Overlay baseline vs coupled-model Bode plots from existing band-split merged CSVs.

Reads:
  logs/wt_uic_multitone_fft_bands/impedance_matrix_fft_merged.csv   (baseline)
  logs/fmu_uic_multitone_fft_bands/impedance_matrix_fft_merged.csv  (coupled model)

Writes comparison figures and trust report under:
  logs/thesis_multitone_fft_bands_compare/

Run from repo root:
  python casestudies/impedance_stability/multitone_fft_bands/plot_wt_fmu_overlay.py
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

from casestudies.impedance_stability.multitone_fft_bands.band_specs import F_MAX_HZ, F_MIN_HZ
from casestudies.impedance_stability.multitone_fft_bands.merge_bands import MERGED_CSV
from casestudies.impedance_stability.paths import (
    FMU_UIC_MULTITONE_FFT_BANDS_DIR,
    THESIS_MULTITONE_FFT_BANDS_COMPARE_DIR,
    WT_UIC_MULTITONE_FFT_BANDS_DIR,
)
from casestudies.impedance_stability.plots.plot_thesis_stability import (
    plot_ydev_dq_overlay,
    plot_ydev_pp_overlay,
    plot_zbus_dq_overlay,
)

OUT_DIR = THESIS_MULTITONE_FFT_BANDS_COMPARE_DIR
LABEL_BASELINE = "Baseline"
LABEL_COUPLED = "Coupled model"
TRUST_REPORT_TXT = "Ydev_low_trust_bins_baseline_vs_coupled.txt"


def _merged_csv(log_dir: os.PathLike[str]) -> str:
    path = os.path.join(str(log_dir), MERGED_CSV)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing merged CSV: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline vs coupled-model overlay plots from merged multitone FFT CSVs."
    )
    parser.add_argument("--f-min", type=float, default=float(F_MIN_HZ), help="Plot band lower edge (Hz)")
    parser.add_argument("--f-max", type=float, default=float(F_MAX_HZ), help="Plot band upper edge (Hz)")
    parser.add_argument(
        "--mark-hz",
        type=float,
        default=None,
        help="Optional vertical guide frequency (Hz); default none",
    )
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output directory for PNGs")
    parser.add_argument(
        "--phase-ref-hz",
        type=float,
        default=None,
        help="Phase overlay anchor (Hz); default 0.1 Hz",
    )
    parser.add_argument(
        "--no-phase-ref",
        action="store_true",
        help="Disable common phase reference + anchored unwrap on overlay phase panels",
    )
    args = parser.parse_args()

    baseline_csv = _merged_csv(WT_UIC_MULTITONE_FFT_BANDS_DIR)
    coupled_csv = _merged_csv(FMU_UIC_MULTITONE_FFT_BANDS_DIR)
    series = [
        (LABEL_BASELINE, baseline_csv),
        (LABEL_COUPLED, coupled_csv),
    ]
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    f_lo, f_hi = float(args.f_min), float(args.f_max)
    mark = float(args.mark_hz) if args.mark_hz is not None and float(args.mark_hz) > 0 else None
    phase_ref = not args.no_phase_ref
    phase_ref_hz = None if args.phase_ref_hz is None else float(args.phase_ref_hz)
    report_path = os.path.join(out_dir, TRUST_REPORT_TXT)

    plot_zbus_dq_overlay(
        series,
        os.path.join(out_dir, "Zbus_dq_baseline_vs_coupled.png"),
        f_min_hz=f_lo,
        f_max_hz=f_hi,
        mark_hz=mark,
        phase_ref_hz=phase_ref_hz,
        phase_ref=phase_ref,
    )

    plot_ydev_dq_overlay(
        series,
        os.path.join(out_dir, "Ydev_dq_baseline_vs_coupled.png"),
        f_min_hz=f_lo,
        f_max_hz=f_hi,
        mark_hz=mark,
        trust_report_path=report_path,
        phase_ref_hz=phase_ref_hz,
        phase_ref=phase_ref,
    )
    plot_ydev_pp_overlay(
        series,
        os.path.join(out_dir, "Ydev_pp_baseline_vs_coupled.png"),
        f_min_hz=f_lo,
        f_max_hz=f_hi,
        mark_hz=mark,
        phase_ref_hz=phase_ref_hz,
        phase_ref=phase_ref,
    )

    print(f"Wrote baseline vs coupled-model overlays to {out_dir}", flush=True)
    print(f"  Baseline (WT logs):  {baseline_csv}", flush=True)
    print(f"  Coupled (FMU logs):  {coupled_csv}", flush=True)
    print("  Z_bus dq: Zbus_dq_baseline_vs_coupled.png", flush=True)
    print("  Y_dev dq: Ydev_dq_baseline_vs_coupled.png", flush=True)
    print("  Y_dev ++: Ydev_pp_baseline_vs_coupled.png", flush=True)
    print(f"  Trust report: {TRUST_REPORT_TXT}", flush=True)


if __name__ == "__main__":
    main()
