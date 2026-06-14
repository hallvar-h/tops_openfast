# dyn_sim

Time-domain wind turbine simulations (baseline WT vs OpenFAST-coupled FMU).

## Run (repo root)

```bash
python casestudies/dyn_sim/test_WT_sim.py
python casestudies/dyn_sim/test_WT_FMU_drivetrain_sim.py
```

Coupled run needs `fast.fmu` at repo root or `OpenFAST/fast.fmu`, and OpenFAST case `test1002/`.

## Outputs

| Run | CSV | Figures |
|-----|-----|---------|
| Baseline | `logs/wt/wt_model.csv` | `logs/wt/plots/thesis/` |
| Coupled | `logs/fmu_drivetrain/fmu_drivetrain.csv` | `logs/fmu_drivetrain/plots/thesis/` |

Replot from CSV (no re-simulation): `plotting/regenerate_thesis_plots_from_csv.py`  
Comparison overlays: `logs/compare/plots/thesis/` — see [`plotting/README.md`](plotting/README.md).
