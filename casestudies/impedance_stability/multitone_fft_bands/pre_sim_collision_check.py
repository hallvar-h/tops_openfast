"""
Pre-simulation modified-sequence-domain (MSD) collision check for all bands.

Same test as the legacy single-band workflow, but run **before** sims using
the exact tone lists from ``ps_data.multitone_fft_bands`` (LF / MF / HF).

In this project this is the multitone "collision" / mirror-bin check at f1=50 Hz
(not a structural shaft/tower resonance scan).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

# Repo root on path when run as script (see ``if __name__`` block too).
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from casestudies.impedance_stability.multitone_fft_bands.collision_check import (
    check_msd_multitone_list,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (
    BAND_ORDER,
    F1_HZ,
    TOL,
    get_band_spec,
)
from casestudies.impedance_stability.ps_data.multitone_fft_bands import (
    get_band_injected_frequencies,
)


def _format_collision_report(*, band: str, freqs: np.ndarray, coll: dict[str, Any]) -> str:
    lines = [
        f"Band: {band}",
        f"f1_Hz (grid frequency): {F1_HZ:g}",
        f"n_tones: {int(freqs.size)}",
        f"f_Hz: {np.array2string(freqs, precision=4, separator=', ')}",
        "",
        "Modified sequence domain collision check (pre-simulation):",
        "  For each tone f_m: upper bin = f1 + f_m, mirror bin = |f1 - f_m|",
        "",
        "Pairs with same mirror bin: "
        + (str(coll["same_mirror"]) if coll["same_mirror"] else "None"),
        "Pairs with upper/mirror overlap: "
        + (str(coll["cross_coll"]) if coll["cross_coll"] else "None"),
        "Special single-tone issues: "
        + (str(coll["special"]) if coll["special"] else "None"),
        "",
    ]
    ok = not (coll["same_mirror"] or coll["cross_coll"] or coll["special"])
    lines.append("Status: PASS" if ok else "Status: FAIL")
    return "\n".join(lines) + "\n"


def _has_collisions(coll: dict[str, Any]) -> bool:
    return bool(coll["same_mirror"] or coll["cross_coll"] or coll["special"])


def run_pre_sim_collision_checks(
    log_dir: str | None = None,
    *,
    bands: Iterable[str] | None = None,
    fail_on_collision: bool = True,
    write_per_band: bool = True,
) -> dict[str, Any]:
    """
    Run MSD collision check on LF, MF, HF tone lists before time-domain sims.

    Writes:
      - ``multitone_collision_report_lf.txt`` (etc.) when ``log_dir`` is set
      - ``multitone_collision_report_pre_sim.txt`` summary when ``log_dir`` is set

    Returns a dict with per-band results and overall ``passed`` flag.
    """
    results: dict[str, Any] = {"bands": {}, "passed": True}

    band_list = tuple(bands) if bands is not None else BAND_ORDER
    for b in band_list:
        if b not in BAND_ORDER:
            raise ValueError(f"Unknown band {b!r}; expected one of {BAND_ORDER}")

    print("\n=== Pre-simulation MSD collision check ===", flush=True)
    for band in band_list:
        spec = get_band_spec(band)
        freqs = get_band_injected_frequencies(band)
        coll = check_msd_multitone_list(freqs, f1=float(F1_HZ), tol=float(TOL))
        ok = not _has_collisions(coll)
        results["bands"][band] = {
            "f_min_hz": spec.f_min_hz,
            "f_max_hz": spec.f_max_hz,
            "n_tones": int(freqs.size),
            "freqs_hz": freqs.tolist(),
            "collision": coll,
            "passed": ok,
        }
        if not ok:
            results["passed"] = False

        status = "PASS" if ok else "FAIL"
        print(
            f"  [{band}] {status}  n={freqs.size}  "
            f"f=[{spec.f_min_hz:g},{spec.f_max_hz:g}] Hz",
            flush=True,
        )
        if not ok:
            print(f"         details: {coll}", flush=True)

        if log_dir and write_per_band:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"multitone_collision_report_{band}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_format_collision_report(band=band, freqs=freqs, coll=coll))
            print(f"         wrote {path}", flush=True)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        summary_path = os.path.join(log_dir, "multitone_collision_report_pre_sim.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write("Pre-simulation MSD collision check (LF + MF + HF)\n")
            fh.write(f"f1_Hz = {F1_HZ:g}\n\n")
            for band in band_list:
                b = results["bands"][band]
                freqs = np.asarray(b["freqs_hz"], dtype=float)
                fh.write(_format_collision_report(band=band, freqs=freqs, coll=b["collision"]))
                fh.write("\n")
            fh.write(
                "Overall: PASS\n" if results["passed"] else "Overall: FAIL — do not start sims\n"
            )
        print(f"Wrote {summary_path}", flush=True)

    if results["passed"]:
        print("Pre-simulation MSD collision check: all bands PASS.\n", flush=True)
    else:
        print("Pre-simulation MSD collision check: FAIL on one or more bands.\n", flush=True)
        if fail_on_collision:
            raise SystemExit(1)

    return results


if __name__ == "__main__":
    from casestudies.impedance_stability.paths import WT_UIC_MULTITONE_FFT_BANDS_DIR

    log = str(WT_UIC_MULTITONE_FFT_BANDS_DIR) if len(sys.argv) < 2 else sys.argv[1]
    run_pre_sim_collision_checks(log, fail_on_collision=True)
