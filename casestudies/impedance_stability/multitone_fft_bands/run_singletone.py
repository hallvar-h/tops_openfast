"""
Single-tone cross-check (one frequency, re + im sim, MIMO ID).

Examples (repo root):
  python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant fmu --freq 0.17 0.23
  python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant wt --freq 0.17 --amp 0.05
  python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant fmu --freq 0.38 --sim-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from casestudies.impedance_stability.multitone_fft_bands.band_specs import DT_S
from casestudies.impedance_stability.multitone_fft_bands.fft_extract import (
    run_fft_multitone_mimo_band,
)
from casestudies.impedance_stability.multitone_fft_bands.sim_axis import run_axis
from casestudies.impedance_stability.multitone_fft_bands.singletone_specs import (
    SingleToneSpec,
    singletone_spec_for_frequency,
)
from casestudies.impedance_stability.paths import (
    FMU_UIC_MULTITONE_FFT_BANDS_DIR,
    FMU_UIC_MULTITONE_FFT_SINGLETONE_DIR,
    WT_UIC_MULTITONE_FFT_BANDS_DIR,
    WT_UIC_MULTITONE_FFT_SINGLETONE_DIR,
)
from casestudies.impedance_stability.ps_data.multitone_fft_singletone import (
    load_multitone_fft_singletone,
)


def _bands_merged_csv(plant: str) -> str:
    root = (
        FMU_UIC_MULTITONE_FFT_BANDS_DIR
        if plant == "fmu"
        else WT_UIC_MULTITONE_FFT_BANDS_DIR
    )
    return os.path.join(str(root), "impedance_matrix_fft_merged.csv")


def _row_at_freq(df: pd.DataFrame, f_hz: float) -> pd.Series | None:
    if df.empty or "f_Hz" not in df.columns:
        return None
    f = df["f_Hz"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(f - float(f_hz))))
    if not np.isfinite(f[i]) or abs(f[i] - float(f_hz)) > 0.02:
        return None
    return df.iloc[i]


def _write_report(
    *,
    log_dir: str,
    plant: str,
    spec: SingleToneSpec,
    id_csv: str,
) -> None:
    df = pd.read_csv(id_csv)
    row = df.iloc[0]
    ydd = complex(row["Ydev_m00_re"], row["Ydev_m00_im"])
    report: dict[str, Any] = {
        "plant": plant,
        "f_requested_hz": spec.f_hz,
        "band_tag": spec.band_tag,
        "t_win_s": spec.t_win_s,
        "df_fft_hz": spec.df_fft_hz,
        "amp_per_tone_pu": spec.amp_per_tone_pu,
        "cycles": spec.cycles_at_f_min,
        "kappa_Ydev": float(row.get("kappa_Ydev", np.nan)),
        "detVt_abs": float(row.get("detVt_abs", np.nan)),
        "detIpert_abs": float(row.get("detIpert_abs", np.nan)),
        "Ydev_dd_abs_pu": float(abs(ydd)),
        "Ydev_dd_db": float(20.0 * np.log10(abs(ydd) + 1e-30)),
    }

    merged_path = _bands_merged_csv(plant)
    if os.path.isfile(merged_path):
        mdf = pd.read_csv(merged_path)
        mrow = _row_at_freq(mdf, spec.f_hz)
        if mrow is not None:
            my = complex(mrow["Ydev_m00_re"], mrow["Ydev_m00_im"])
            report["multitone_bands_merged"] = {
                "f_Hz": float(mrow["f_Hz"]),
                "kappa_Ydev": float(mrow.get("kappa_Ydev", np.nan)),
                "detVt_abs": float(mrow.get("detVt_abs", np.nan)),
                "Ydev_dd_abs_pu": float(abs(my)),
                "Ydev_dd_db": float(20.0 * np.log10(abs(my) + 1e-30)),
            }

    out_json = os.path.join(log_dir, f"singletone_report_{spec.band_tag}.json")
    out_txt = os.path.join(log_dir, f"singletone_report_{spec.band_tag}.txt")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Single-tone cross-check ({plant})\n")
        fh.write(f"  f = {spec.f_hz:g} Hz  tag = {spec.band_tag}\n")
        fh.write(f"  T_win = {spec.t_win_s:g} s  amp/tone = {spec.amp_per_tone_pu:.5f} pu\n")
        fh.write(
            f"  kappa(Y) = {report['kappa_Ydev']:.4g}  detVt = {report['detVt_abs']:.4g}  "
            f"|Ydd| = {report['Ydev_dd_abs_pu']:.4g} pu ({report['Ydev_dd_db']:.2f} dB)\n"
        )
        mb = report.get("multitone_bands_merged")
        if mb:
            fh.write("  vs multitone bands merged:\n")
            fh.write(
                f"    kappa(Y) = {mb['kappa_Ydev']:.4g}  |Ydd| = {mb['Ydev_dd_abs_pu']:.4g} pu "
                f"({mb['Ydev_dd_db']:.2f} dB)\n"
            )
    print(f"Wrote {out_txt}", flush=True)


def run_singletone_frequency(
    *,
    plant: str,
    f_hz: float,
    log_dir: str,
    amp_per_tone_pu: float | None = None,
    run_sim: bool = True,
    run_id: bool = True,
) -> SingleToneSpec:
    spec = singletone_spec_for_frequency(f_hz, amp_per_tone_pu=amp_per_tone_pu)
    errs = spec.validate()
    if errs:
        raise ValueError("; ".join(errs))

    print(
        f"\n=== {plant.upper()} single tone {spec.f_hz:g} Hz "
        f"(tag={spec.band_tag}, T_win={spec.t_win_s:g}s, "
        f"amp={spec.amp_per_tone_pu:.5f} pu) ===",
        flush=True,
    )
    if abs(spec.f_hz - float(f_hz)) > 1e-9:
        print(f"  (snapped from requested {float(f_hz):g} Hz to FFT bin)", flush=True)

    t_end = spec.t_end_s(plant)
    os.makedirs(log_dir, exist_ok=True)

    if run_sim:
        for axis in ("re", "im"):
            model = load_multitone_fft_singletone(
                plant=plant,  # type: ignore[arg-type]
                axis=axis,
                f_hz=spec.f_hz,
                amp_per_tone_pu=spec.amp_per_tone_pu,
            )
            run_axis(
                plant=plant,  # type: ignore[arg-type]
                case_id="singletone",
                run_axis=axis,
                band_tag=spec.band_tag,
                t_end_s=t_end,
                dt_s=DT_S,
                log_dir=log_dir,
                model=model,
            )

    id_csv = os.path.join(log_dir, f"impedance_matrix_fft_{spec.band_tag}.csv")
    if run_id:
        run_fft_multitone_mimo_band(
            log_dir,
            spec.band_tag,
            spec=spec,
            plant=plant,  # type: ignore[arg-type]
        )
        out = pd.read_csv(id_csv)
        out["id_method"] = "multitone_fft_singletone"
        out["band"] = spec.band_tag
        out.to_csv(id_csv, index=False)
        _write_report(log_dir=log_dir, plant=plant, spec=spec, id_csv=id_csv)

    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-tone sim + ID for cross-checking frequencies of interest."
    )
    parser.add_argument("--plant", choices=("wt", "fmu"), required=True)
    parser.add_argument(
        "--freq",
        type=float,
        nargs="+",
        required=True,
        help="One or more frequencies in Hz (e.g. 0.17 0.23 0.38)",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=None,
        help="Per-tone current amplitude (pu). Default: 0.05*sqrt(2) (~0.0707 pu)",
    )
    parser.add_argument("--sim-only", action="store_true", help="Run re/im sims only")
    parser.add_argument("--id-only", action="store_true", help="FFT ID only (logs must exist)")
    args = parser.parse_args()

    if args.sim_only and args.id_only:
        raise SystemExit("Use at most one of --sim-only and --id-only")

    log_dir = str(
        FMU_UIC_MULTITONE_FFT_SINGLETONE_DIR
        if args.plant == "fmu"
        else WT_UIC_MULTITONE_FFT_SINGLETONE_DIR
    )
    run_sim = not args.id_only
    run_id = not args.sim_only

    for f in args.freq:
        run_singletone_frequency(
            plant=args.plant,
            f_hz=float(f),
            log_dir=log_dir,
            amp_per_tone_pu=args.amp,
            run_sim=run_sim,
            run_id=run_id,
        )

    print(f"\nDone. Logs under {log_dir}", flush=True)


if __name__ == "__main__":
    main()
