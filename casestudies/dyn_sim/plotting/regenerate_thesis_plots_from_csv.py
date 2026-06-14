"""Regenerate thesis time-domain PNGs from saved CSV logs (no re-simulation)."""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd

from casestudies.dyn_sim.plotting.compare_WT_vs_FMU_results import main as compare_main
from casestudies.dyn_sim.plotting.log_paths import (
    FMU_DRIVETRAIN_CSV,
    WT_CSV,
    fmu_drivetrain_thesis_plots_dir,
    wt_thesis_plots_dir,
)
from casestudies.dyn_sim.plotting.thesis_plot_style import (
    clear_thesis_plot_dir,
    save_baseline_thesis_plots,
    save_coupled_thesis_plots,
)

# Columns in fmu_drivetrain.csv that are not raw OpenFAST FMU outputs.
_COUPLED_NON_FMU_COLS = frozenset({
    't', 'P_e_sys_pu', 'P_ref_sys_pu', 'P_e_uic_pu_raw', 'P_ref_uic_pu_raw',
    'v_bus_pu', 'vi_mag_pu', 'i_a_mag_pu_uic', 'i_a_angle_deg',
    'P_uic_bus_actual_sys_pu', 'Q_uic_bus_actual_sys_pu',
    'P_uic_bus_ref_sys_pu', 'Q_uic_bus_ref_sys_pu', 'P_inf_sys_pu', 'Q_inf_sys_pu',
    'omega_base_rpm', 'omega_e_tops_pu', 'Te_cmd_pu', 'Te_cmd_kNm', 'GenSpdOrTrq_set_kNm', 'omega_m_pu_meas',
    'GenPwr_set_kW', 'ElecPwrCom_set_kW', 'P_cmd_kW', 'P_cmd_pu',
    'Mode_write', 'Mode_readback', 'testNr_write', 'testNr_readback',
})


def _plot_baseline_from_csv(show: bool = False) -> str:
    if not WT_CSV.is_file():
        raise FileNotFoundError(f"Missing {WT_CSV} — run test_WT_sim.py first.")
    wt = pd.read_csv(WT_CSV)
    thesis_dir = wt_thesis_plots_dir()
    n_rm = clear_thesis_plot_dir(str(thesis_dir))
    paths = save_baseline_thesis_plots(
        str(thesis_dir), wt['t'].to_numpy(dtype=float), wt, clean_first=False, show=show,
    )
    print(f"  removed {n_rm} old PNG(s), wrote {len(paths)} baseline figure(s)")
    return str(thesis_dir)


def _plot_coupled_from_csv(show: bool = False) -> str:
    if not FMU_DRIVETRAIN_CSV.is_file():
        raise FileNotFoundError(f"Missing {FMU_DRIVETRAIN_CSV} — run test_WT_FMU_drivetrain_sim.py first.")
    df = pd.read_csv(FMU_DRIVETRAIN_CSV)
    thesis_dir = fmu_drivetrain_thesis_plots_dir()
    omega_base_rpm = float(df['omega_base_rpm'].iloc[0]) if 'omega_base_rpm' in df.columns else None
    fmu_cols = [c for c in df.columns if c not in _COUPLED_NON_FMU_COLS]
    df_fmu = df[fmu_cols] if fmu_cols else None
    n_rm = clear_thesis_plot_dir(str(thesis_dir))
    paths = save_coupled_thesis_plots(
        str(thesis_dir),
        df['t'].to_numpy(dtype=float),
        df,
        df_fmu=df_fmu,
        omega_base_rpm=omega_base_rpm,
        clean_first=False,
        show=show,
    )
    print(f"  removed {n_rm} old PNG(s), wrote {len(paths)} coupled figure(s)")
    return str(thesis_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--show', action='store_true', help='Open plot windows (default: save PNGs only).')
    ap.add_argument('--baseline-only', action='store_true')
    ap.add_argument('--coupled-only', action='store_true')
    ap.add_argument('--compare-only', action='store_true')
    args = ap.parse_args(argv)
    show = args.show
    cmp_args = ['--show'] if args.show else []

    if args.compare_only:
        compare_main(cmp_args)
        return

    if not args.coupled_only:
        bdir = _plot_baseline_from_csv(show=show)
        print(f"Baseline thesis figures written to {bdir}")

    if not args.baseline_only:
        cdir = _plot_coupled_from_csv(show=show)
        print(f"Coupled thesis figures written to {cdir}")
        if not args.coupled_only:
            print('Comparison figures:')
            compare_main(cmp_args)


if __name__ == '__main__':
    main()
