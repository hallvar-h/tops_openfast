"""Orchestrate band-split multitone sims, FFT ID, merge, and thesis plots."""

from __future__ import annotations

import os
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from casestudies.impedance_stability.multitone_fft_bands.collision_check import check_msd_multitone_list
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    BAND_ORDER,
    DT_S,
    F1_HZ,
    F_MIN_HZ,
    TOL,
    get_band_spec,
    t_end_s,
)
from casestudies.impedance_stability.multitone_fft_bands.fft_extract import run_fft_multitone_mimo_band
from casestudies.impedance_stability.multitone_fft_bands.merge_bands import (
    MERGED_CSV,
    merge_three_band_csvs,
)
from casestudies.impedance_stability.multitone_fft_bands.sim_axis import run_axis
from casestudies.impedance_stability.multitone_fft_bands.pre_sim_collision_check import (
    run_pre_sim_collision_checks,
)
from casestudies.impedance_stability.multitone_fft_bands.validate_plan import print_and_assert
from casestudies.impedance_stability.plots.plot_thesis_stability import (
    plot_loop_eigenloci_dq,
    plot_nyquist_loop_pp,
    plot_nyquist_loop_qq,
    plot_ydev_pp,
)
from casestudies.impedance_stability.plots.plot_thesis_zbus import plot_zbus_pp

PlantId = Literal["wt", "fmu", "uic"]


def _write_collision_report(log_dir: str, band: str) -> None:
    path = os.path.join(log_dir, f"injected_tones_{band}_re.csv")
    tones = pd.read_csv(path)
    f_list = pd.to_numeric(tones["f_Hz"], errors="coerce").to_numpy(dtype=float)
    f_list = f_list[np.isfinite(f_list) & (f_list > 0.0)]
    coll = check_msd_multitone_list(f_list, f1=float(F1_HZ), tol=float(TOL))
    out = os.path.join(log_dir, f"multitone_collision_report_{band}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Band: {band}\n")
        f.write("Pairs with same mirror bin: " + (str(coll["same_mirror"]) if coll["same_mirror"] else "None") + "\n")
        f.write("Pairs with upper/mirror overlap: " + (str(coll["cross_coll"]) if coll["cross_coll"] else "None") + "\n")
        f.write("Special single-tone issues: " + (str(coll["special"]) if coll["special"] else "None") + "\n")
    print(f"Wrote {out}", flush=True)


def run_multitone_bands_pipeline(
    *,
    plant: PlantId,
    log_dir: str,
    case_prefix: str,
    label: str,
    bands: Iterable[str] | None = None,
    run_sims: bool = True,
    run_id: bool = True,
    run_merge: bool = True,
    run_plots: bool = True,
    pre_sim_collision_check: bool = True,
) -> str:
    """
    Full band-split workflow for WT or FMU.

    ``case_prefix`` is ``multitone_bands`` or ``fmu_multitone_bands``.
    ``bands`` defaults to all of ``BAND_ORDER``; pass e.g. ``("lf",)`` to rerun one band only.
    """
    os.makedirs(log_dir, exist_ok=True)
    print_and_assert()
    band_list = tuple(bands) if bands is not None else BAND_ORDER
    for b in band_list:
        if b not in BAND_ORDER:
            raise ValueError(f"Unknown band {b!r}; expected one of {BAND_ORDER}")

    if pre_sim_collision_check:
        run_pre_sim_collision_checks(log_dir, fail_on_collision=True, bands=band_list)

    for band in band_list:
        spec = get_band_spec(band)
        t_end = float(t_end_s(spec, plant))
        if run_sims:
            print(f"\n=== {plant.upper()} band={band} sims t_end={t_end:g}s ===", flush=True)
            for axis in ("re", "im"):
                case_id = f"{case_prefix}_{band}_{axis}"
                run_axis(
                    plant=plant,
                    case_id=case_id,
                    run_axis=axis,
                    band_tag=band,
                    t_end_s=t_end,
                    dt_s=float(DT_S),
                    log_dir=log_dir,
                )
            _write_collision_report(log_dir, band)

        if run_id:
            print(f"\n=== FFT ID band={band} ===", flush=True)
            run_fft_multitone_mimo_band(log_dir, band, spec=spec, plant=plant)

    merged_csv = os.path.join(log_dir, MERGED_CSV)
    if run_merge:
        print("\n=== Merge LF/MF/HF ===", flush=True)
        merge_three_band_csvs(log_dir)

    if run_plots and os.path.isfile(merged_csv):
        plot_zbus_pp(
            merged_csv,
            os.path.join(log_dir, "Zbus_pp_multitone_fft_bands.png"),
            label=label,
            f_min_hz=F_MIN_HZ,
            f_max_hz=10.0,
        )
        plot_ydev_pp(
            merged_csv,
            os.path.join(log_dir, "Ydev_pp_multitone_fft_bands.png"),
            label=label,
            f_min_hz=F_MIN_HZ,
            f_max_hz=10.0,
        )
        plot_nyquist_loop_pp(
            merged_csv,
            os.path.join(log_dir, "nyquist_loop_multitone_fft_bands.png"),
            label=label,
            f_min_hz=F_MIN_HZ,
            f_max_hz=10.0,
        )
        plot_nyquist_loop_qq(
            merged_csv,
            os.path.join(log_dir, "nyquist_loop_qq_multitone_fft_bands.png"),
            label=label,
            f_min_hz=F_MIN_HZ,
            f_max_hz=10.0,
        )
        stats = plot_loop_eigenloci_dq(
            merged_csv,
            os.path.join(log_dir, "loop_eigenloci_dq_multitone_fft_bands.png"),
            label=label,
            f_min_hz=F_MIN_HZ,
            f_max_hz=10.0,
        )
        print(f"dq eigenloci: max|lambda|={stats['max_abs_lambda']:.4g}", flush=True)
        print(f"Thesis plots saved under {log_dir}", flush=True)

    return merged_csv
