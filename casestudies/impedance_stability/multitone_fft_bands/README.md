# Multitone FFT — band-split (0.05–10 Hz)

Reliable multitone impedance identification by running **three band-limited** simulations per plant (LF / MF / HF), each with a long FFT window tuned to that band.

## Bands

**Source of truth:** all band limits, FFT windows, tone grids, settle/margin times, injection RMS, and merge overlap bands are defined in [`band_specs.py`](band_specs.py) (`BandSpec`, `LF_TONE_SEGMENTS`, `TARGET_TOTAL_RMS`, `T_SETTLE_S_*`, `LF_MF_*` / `MF_HF_*`). The table below is the **current thesis default**; change those constants (then run `validate_plan.py`) to reconfigure — runners and `ps_data/multitone_fft_bands.py` read from `get_band_spec()` / `t_end_s()`, not from this README.

| Band | f (Hz) | T_win (s) | df_fft (Hz) | N tones | t_end WT (s) | t_end FMU (s) |
|------|--------|-----------|-------------|---------|--------------|---------------|
| LF | 0.05 – 1.0 | 500 | 0.002 | **63** (0.01 Hz to 0.5 Hz, 0.03 Hz above) | 525 | 555 |
| MF | 0.9 – 4.0 | 200 | 0.005 | 24 | 225 | 255 |
| HF | 3.5 – 10.0 | 120 | 0.00833 | 22 | 145 | 175 |

- **Settle before FFT window:** WT 20 s, FMU 50 s (`T_SETTLE_S_FMU`). Same `T_win` and 5 s margin for both.

- **LF grid (one band, one sim):** **0.05–0.5 Hz @ 0.01 Hz** (46 tones, includes **0.16 / 0.17 / 0.18 Hz**), then **0.53–1.0 Hz @ 0.03 Hz** (17 tones). Same `T_win=500` s and MSD pre-check: `validate_plan.py`. Not a separate band — variable spacing inside `lf`.
- **Injection:** `target_total_rms = 0.05` pu on every band (per-tone `amp = RMS × √(2/N)`).
- **Merge:** LF–MF blend 0.9–1.0 Hz; MF–HF blend 3.5–4.2 Hz. Authoritative curve: `impedance_matrix_fft_merged.csv`.

## Run (repo root)

```text
python casestudies/impedance_stability/multitone_fft_bands/validate_plan.py
python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_wt.py
python casestudies/impedance_stability/multitone_fft_bands/run_multitone_id_fmu.py
python casestudies/impedance_stability/multitone_fft_bands/run_rerun_bands.py --plant fmu --bands lf
```

### Single-tone cross-check (one frequency)

Same UIC+WT/FMU setup as band-split, but **one tone** per run (stronger per-tone amp: `0.05×√2` pu by default; amplitude and window snap to the parent band via [`singletone_specs.py`](singletone_specs.py)). Use to verify LF peaks (e.g. 0.17, 0.23, 0.38 Hz) without multitone grid noise.

```text
python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant fmu --freq 0.17 0.23
python casestudies/impedance_stability/multitone_fft_bands/run_singletone.py --plant wt --freq 0.17
```

Logs: `logs/wt_uic_multitone_fft_singletone/`, `logs/fmu_uic_multitone_fft_singletone/`

Outputs per frequency `f0p17` (tag from Hz): `injected_tones_f0p17_{re,im}.csv`, `uic_terminal_vi_multisine_f0p17_{re,im}.csv`, `impedance_matrix_fft_f0p17.csv`, `singletone_report_f0p17.txt` (includes comparison to multitone **merged** CSV when present).

### Pre-simulation MSD collision check

Runs automatically at the start of each runner, before any band sim. Per-band reports:
`multitone_collision_report_lf.txt`, `_mf.txt`, `_hf.txt`, plus `multitone_collision_report_pre_sim.txt`.
Standalone: `python casestudies/impedance_stability/multitone_fft_bands/pre_sim_collision_check.py`

Logs:

- `casestudies/impedance_stability/logs/wt_uic_multitone_fft_bands/`
- `casestudies/impedance_stability/logs/fmu_uic_multitone_fft_bands/`

## Outputs per model

- Sim: `injected_tones_{band}_{axis}.csv`, `uic_terminal_vi_multisine_{band}_{axis}.csv`
- ID: `impedance_matrix_fft_lf.csv`, `_mf.csv`, `_hf.csv`
- Merged: `impedance_matrix_fft_merged.csv`, `merge_qa.json`
- Thesis: `Zbus_pp_multitone_fft_bands.png`, `Ydev_pp_multitone_fft_bands.png`, `nyquist_loop_multitone_fft_bands.png`
