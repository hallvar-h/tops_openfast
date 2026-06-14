# Impedance stability (band-split multitone FFT)

UIC+WT baseline vs FMU coupled-model impedance identification at the UIC terminal (bus B2), with a stiff grid (infinite bus) at B1, using three band-limited multitone runs (LF / MF / HF) plus optional singletone cross-checks.

## Layout

```
impedance_stability/
├── paths.py                 # log directory constants
├── identification/          # MIMO FFT impedance math + band merge helpers
├── plots/                   # thesis Bode/Nyquist figures (used by pipeline)
├── ps_data/                 # CaseLoader registry + WT/FMU power-system builders
├── multitone_fft_bands/     # sim → FFT extract → merge → plot pipeline
└── logs/                    # run outputs (regenerable)
```

## Run (repo root)

See [`multitone_fft_bands/README.md`](multitone_fft_bands/README.md) for band specs and commands:

- `validate_plan.py` — pre-flight tone grid + MSD collision check
- `run_multitone_id_wt.py` / `run_multitone_id_fmu.py` — full band pipeline
- `run_rerun_bands.py` — rerun selected bands
- `run_singletone.py` — single-tone cross-check at chosen frequencies

## Active log directories

| Directory | Purpose |
|-----------|---------|
| `logs/wt_uic_multitone_fft_bands/` | Baseline WT band runs + merged CSV |
| `logs/fmu_uic_multitone_fft_bands/` | FMU band runs + merged CSV |
| `logs/wt_uic_multitone_fft_singletone/` | WT singletone cross-checks |
| `logs/fmu_uic_multitone_fft_singletone/` | FMU singletone cross-checks |
| `logs/thesis_multitone_fft_bands_compare/` | WT vs FMU overlay (`plot_wt_fmu_overlay.py`) |

## Optional (not in main pipeline)

- `multitone_fft_bands/plot_wt_fmu_overlay.py` — baseline vs coupled Bode overlay
- `plots/plot_perturbation_signal.py` — visualize injected multitone spectrum
