# dyn_sim plotting

Thesis time-domain figures and path constants for `dyn_sim/logs/`.

| File | Role |
|------|------|
| `log_paths.py` | CSV + PNG paths under `../logs/` |
| `thesis_plot_style.py` | Baseline / coupled / compare figure builders |
| `regenerate_thesis_plots_from_csv.py` | Replot from CSVs (no re-simulation) |
| `compare_WT_vs_FMU_results.py` | Baseline vs coupled overlays |

Run from repo root:

```bash
python casestudies/dyn_sim/plotting/regenerate_thesis_plots_from_csv.py
python casestudies/dyn_sim/plotting/compare_WT_vs_FMU_results.py
```

Sim scripts (`test_WT_sim.py`, `test_WT_FMU_drivetrain_sim.py`) import from this package and write CSVs to `dyn_sim/logs/`.
