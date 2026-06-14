import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from casestudies.dyn_sim.plotting.log_paths import (
    FMU_DRIVETRAIN_CSV,
    WT_CSV,
    compare_thesis_plots_dir,
)
from casestudies.dyn_sim.plotting.thesis_plot_style import (
    COLOR_BASELINE_ALT,
    COLOR_COUPLED_ALT,
    COLOR_WIND,
    LS_REF,
    clear_thesis_plot_dir,
    legend_compare,
    save_compare_overlay,
)


def _ensure_increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0:
        return t
    _, idx = np.unique(t, return_index=True)
    idx = np.sort(idx)
    return t[idx]


def _align_df_on_time(df: pd.DataFrame, t_target: np.ndarray, t_col: str = 't') -> pd.DataFrame:
    if t_col not in df.columns:
        raise KeyError(f"Missing time column '{t_col}'")
    t_src = np.asarray(df[t_col], dtype=float).ravel()
    if t_src.size == 0:
        raise ValueError("Empty time vector")

    uniq, idx = np.unique(t_src, return_index=True)
    order = np.argsort(uniq)
    t_src = uniq[order]
    keep_rows = idx[order]
    d = df.iloc[keep_rows].copy()

    out = pd.DataFrame({t_col: np.asarray(t_target, dtype=float)})
    for col in d.columns:
        if col == t_col:
            continue
        s = d[col]
        if not np.issubdtype(s.dtype, np.number):
            s = pd.to_numeric(s, errors='coerce')
        if not np.issubdtype(s.dtype, np.number):
            continue
        y = np.asarray(s, dtype=float).ravel()
        if np.all(~np.isfinite(y)):
            continue
        good = np.isfinite(y)
        if np.any(good) and not np.all(good):
            y = np.interp(t_src, t_src[good], y[good])
        out[col] = np.interp(t_target, t_src, y)
    return out


def _pu_from_rpm(rpm: np.ndarray, omega_base_rpm: float) -> np.ndarray:
    rpm = np.asarray(rpm, dtype=float)
    if not np.isfinite(omega_base_rpm) or omega_base_rpm <= 0.0:
        return np.full_like(rpm, np.nan, dtype=float)
    return rpm / omega_base_rpm


def _load_mpt_torque_interp(project_root: str, mpt_filename: str = 'MPT_Kopt2150.csv') -> interp1d:
    """Mechanical torque (WT pu) vs rotor speed (rad/s), same table as WindTurbine."""
    mpt_t_filename = str(mpt_filename).replace('MPT_', 'MPT_T_', 1)
    path = os.path.join(project_root, 'wind_data', mpt_t_filename)
    data = np.loadtxt(path, delimiter='\t')
    rotor_speed_rad_s = data[2:, 0] * (2.0 * np.pi / 60.0)
    torque_mech_pu = data[2:, 1]
    return interp1d(
        rotor_speed_rad_s,
        torque_mech_pu,
        kind='linear',
        bounds_error=False,
        fill_value=(0.0, float(torque_mech_pu[-1])),
    )


def _omega_rad_s_from_pu(omega_pu: np.ndarray, omega_base_rpm: float) -> np.ndarray:
    omega_pu = np.asarray(omega_pu, dtype=float)
    omega_base_rad_s = float(omega_base_rpm) * (2.0 * np.pi / 60.0)
    return omega_pu * omega_base_rad_s


def _torque_wt_pu_from_knm(t_knm: np.ndarray, omega_base_rpm: float, s_wt_mva: float) -> np.ndarray:
    t_knm = np.asarray(t_knm, dtype=float)
    omega_base_rad_s = float(omega_base_rpm) * (2.0 * np.pi / 60.0)
    if not np.isfinite(omega_base_rad_s) or omega_base_rad_s <= 0.0 or s_wt_mva <= 0.0:
        return np.full_like(t_knm, np.nan, dtype=float)
    t_base_nm = float(s_wt_mva) * 1e6 / omega_base_rad_s
    return t_knm * 1e3 / t_base_nm


def _mpt_torque_wt_pu(omega_pu: np.ndarray, omega_base_rpm: float, mpt_interp: interp1d) -> np.ndarray:
    omega_rad_s = _omega_rad_s_from_pu(omega_pu, omega_base_rpm)
    return np.asarray(mpt_interp(omega_rad_s), dtype=float)


def _steady_means(t: np.ndarray, y: np.ndarray, t_min: float = 45.0) -> float:
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(y) & (t >= t_min)
    if not np.any(m):
        return float('nan')
    return float(np.mean(y[m]))


def _has_both(wt_a: pd.DataFrame, fmu_a: pd.DataFrame, wt_col: str, fmu_col: str) -> bool:
    return wt_col in wt_a.columns and fmu_col in fmu_a.columns


def _col(wt_a: pd.DataFrame, fmu_a: pd.DataFrame, wt_col: str, fmu_col: str, *, side: str) -> np.ndarray:
    df = wt_a if side == 'baseline' else fmu_a
    col = wt_col if side == 'baseline' else fmu_col
    return df[col].to_numpy(dtype=float)


def _ref_extra(
    wt_a: pd.DataFrame,
    fmu_a: pd.DataFrame,
    wt_col: str,
    fmu_col: str,
    *,
    detail: str,
) -> list[tuple[np.ndarray, str, str, str]]:
    extra: list[tuple[np.ndarray, str, str, str]] = []
    if wt_col in wt_a.columns:
        extra.append((
            wt_a[wt_col].to_numpy(dtype=float),
            legend_compare('baseline', detail=detail, ref=True),
            COLOR_WIND,
            LS_REF,
        ))
    if fmu_col in fmu_a.columns:
        extra.append((
            fmu_a[fmu_col].to_numpy(dtype=float),
            legend_compare('coupled', detail=detail, ref=True),
            COLOR_COUPLED_ALT,
            LS_REF,
        ))
    return extra


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Compare baseline vs coupled model result CSVs.")
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open interactive plot windows after saving (default: save PNGs only).",
    )
    args = ap.parse_args(argv)

    print(f"Using baseline CSV: {WT_CSV}")
    print(f"Using coupled CSV:  {FMU_DRIVETRAIN_CSV}")

    if not WT_CSV.is_file():
        raise FileNotFoundError(f"Missing baseline results CSV: {WT_CSV}")
    if not FMU_DRIVETRAIN_CSV.is_file():
        raise FileNotFoundError(
            f"Missing coupled results CSV: {FMU_DRIVETRAIN_CSV}\n"
            "Run `casestudies/dyn_sim/test_WT_FMU_drivetrain_sim.py` once to generate it."
        )

    wt = pd.read_csv(WT_CSV)
    fmu = pd.read_csv(FMU_DRIVETRAIN_CSV)

    t_fmu = _ensure_increasing(np.asarray(fmu['t'], dtype=float))
    if t_fmu.size == 0:
        raise ValueError("Coupled CSV has empty time vector")

    wt_a = _align_df_on_time(wt, t_fmu, t_col='t')
    fmu_a = _align_df_on_time(fmu, t_fmu, t_col='t')

    wt_base_rpm = float(wt['omega_base_rpm'].iloc[0]) if 'omega_base_rpm' in wt.columns and len(wt) else np.nan
    fmu_base_rpm = float(fmu['omega_base_rpm'].iloc[0]) if 'omega_base_rpm' in fmu.columns and len(fmu) else np.nan
    if (not np.isfinite(fmu_base_rpm)) or fmu_base_rpm <= 0.0:
        fmu_base_rpm = np.nan
        if 'GenSpeed' in fmu.columns:
            g = np.asarray(fmu['GenSpeed'], dtype=float)
            g = g[np.isfinite(g)]
            if g.size:
                fmu_base_rpm = float(np.median(g[: min(g.size, 200)]))
        if (not np.isfinite(fmu_base_rpm)) or fmu_base_rpm <= 0.0:
            fmu_base_rpm = 1.0
    common_base_rpm = float(fmu_base_rpm)

    if 'omega_e_pu' in wt_a.columns and np.isfinite(wt_base_rpm) and wt_base_rpm > 0:
        wt_a['omega_e_rpm'] = wt_a['omega_e_pu'].to_numpy(dtype=float) * wt_base_rpm
        wt_a['omega_e_pu_common'] = wt_a['omega_e_rpm'].to_numpy(dtype=float) / common_base_rpm
    if 'omega_m_pu' in wt_a.columns and np.isfinite(wt_base_rpm) and wt_base_rpm > 0:
        wt_a['omega_m_rpm'] = wt_a['omega_m_pu'].to_numpy(dtype=float) * wt_base_rpm
        wt_a['omega_m_pu_common'] = wt_a['omega_m_rpm'].to_numpy(dtype=float) / common_base_rpm

    if 'GenSpeed' in fmu_a.columns:
        fmu_a['omega_e_rpm'] = fmu_a['GenSpeed'].to_numpy(dtype=float)
        fmu_a['omega_e_pu_common'] = _pu_from_rpm(fmu_a['omega_e_rpm'].to_numpy(dtype=float), common_base_rpm)
    if 'RotSpeed' in fmu_a.columns:
        fmu_a['omega_m_rpm'] = fmu_a['RotSpeed'].to_numpy(dtype=float)
        fmu_a['omega_m_pu_common'] = _pu_from_rpm(fmu_a['omega_m_rpm'].to_numpy(dtype=float), common_base_rpm)

    plots_dir = compare_thesis_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_rm = clear_thesis_plot_dir(str(plots_dir))
    if n_rm:
        print(f"Removed {n_rm} old comparison PNG(s) from {plots_dir}")
    show = args.show

    saved: list[str] = []

    def _save(
        stem,
        title,
        ylabel,
        baseline=None,
        coupled=None,
        extra=None,
        baseline_ls=None,
        coupled_ls=None,
    ):
        p = save_compare_overlay(
            str(plots_dir),
            stem,
            title,
            t_fmu,
            ylabel=ylabel,
            baseline=baseline,
            coupled=coupled,
            extra=extra,
            baseline_ls=baseline_ls or "-",
            coupled_ls=coupled_ls or "-",
            show=show,
        )
        if p:
            saved.append(p)

    def _pair(
        wt_col: str,
        fmu_col: str,
        stem: str,
        title: str,
        ylabel: str,
        *,
        baseline_detail: str | None = None,
        coupled_detail: str | None = None,
        ref: bool = False,
    ) -> None:
        if not _has_both(wt_a, fmu_a, wt_col, fmu_col):
            return
        ls = LS_REF if ref else "-"
        _save(
            stem,
            title,
            ylabel,
            baseline=(
                _col(wt_a, fmu_a, wt_col, fmu_col, side='baseline'),
                legend_compare('baseline', detail=baseline_detail, ref=ref),
            ),
            coupled=(
                _col(wt_a, fmu_a, wt_col, fmu_col, side='coupled'),
                legend_compare('coupled', detail=coupled_detail, ref=ref),
            ),
            baseline_ls=ls,
            coupled_ls=ls,
        )

    # --- Direct column pairs (same quantity, both logs) ---
    _DIRECT_PAIRS: list[tuple[str, str, str, str, str]] = [
        ('P_e_sys_pu', 'P_e_sys_pu', 'compare_P_e_sys_pu', 'Electrical power at UIC', 'Active power (p.u., system base)'),
        ('P_ref_sys_pu', 'P_ref_sys_pu', 'compare_P_ref_sys_pu', 'Active-power reference', 'Power reference (p.u., system base)'),
        ('P_uic_bus_actual_sys_pu', 'P_uic_bus_actual_sys_pu', 'compare_P_uic_bus_actual_sys_pu', 'UIC bus active power', 'Active power (p.u., system base)'),
        ('P_uic_bus_ref_sys_pu', 'P_uic_bus_ref_sys_pu', 'compare_P_uic_bus_ref_sys_pu', 'UIC bus active-power reference', 'Active power (p.u., system base)'),
        ('Q_uic_bus_actual_sys_pu', 'Q_uic_bus_actual_sys_pu', 'compare_Q_uic_bus_actual_sys_pu', 'UIC bus reactive power', 'Reactive power (p.u., system base)'),
        ('Q_uic_bus_ref_sys_pu', 'Q_uic_bus_ref_sys_pu', 'compare_Q_uic_bus_ref_sys_pu', 'UIC bus reactive-power reference', 'Reactive power (p.u., system base)'),
        ('vi_mag_pu', 'vi_mag_pu', 'compare_vi_mag_pu', r'UIC internal voltage $|v_i|$', 'Voltage magnitude (p.u.)'),
        ('v_bus_pu', 'v_bus_pu', 'compare_Vt_pu', r'UIC terminal voltage $|V_t|$', 'Voltage magnitude (p.u.)'),
        ('i_a_mag_pu_uic', 'i_a_mag_pu_uic', 'compare_i_a_mag_pu', 'Armature current magnitude', 'Current (p.u., UIC base)'),
        ('i_a_angle_deg', 'i_a_angle_deg', 'compare_i_a_angle_deg', 'Armature current angle', 'Current angle (deg)'),
        # P_inf / Q_inf: use compare_inf_bus_PQ_pu only (same as baseline/coupled inf_bus_PQ_pu single plots).
    ]
    for wt_col, fmu_col, stem, title, ylabel in _DIRECT_PAIRS:
        ref = 'ref' in stem or 'reference' in title.lower()
        _pair(wt_col, fmu_col, stem, title, ylabel, ref=ref)

    # --- Renamed / derived pairs ---
    if 'omega_e_pu_common' in wt_a.columns and 'omega_e_pu_common' in fmu_a.columns:
        _pair(
            'omega_e_pu_common', 'omega_e_pu_common',
            'compare_omega_e_pu', 'Generator (electrical) speed',
            r'Speed $\omega_e$ (p.u., rated $\omega_m$ base)',
        )
    if 'omega_m_pu_common' in wt_a.columns and 'omega_m_pu_common' in fmu_a.columns:
        _pair(
            'omega_m_pu_common', 'omega_m_pu_common',
            'compare_omega_m_pu', 'Rotor (mechanical) speed',
            r'Speed $\omega_m$ (p.u., rated $\omega_m$ base)',
        )
    if 'omega_e_rpm' in wt_a.columns and 'omega_e_rpm' in fmu_a.columns:
        _pair('omega_e_rpm', 'omega_e_rpm', 'compare_omega_e_rpm', 'Generator speed', 'Speed (rpm)')
    if 'omega_m_rpm' in wt_a.columns and 'omega_m_rpm' in fmu_a.columns:
        _pair('omega_m_rpm', 'omega_m_rpm', 'compare_omega_m_rpm', 'Rotor speed', 'Speed (rpm)')

    _pair('pitch_deg', 'BldPitch1', 'compare_pitch_deg', 'Blade pitch angle', 'Pitch angle (deg)')

    wind_extra = []
    if 'RtVAvgxh' in fmu_a.columns:
        wind_extra.append((
            fmu_a['RtVAvgxh'].to_numpy(dtype=float),
            legend_compare('coupled', detail='hub-averaged'),
            COLOR_COUPLED_ALT,
            '-',
        ))
    if _has_both(wt_a, fmu_a, 'wind_speed_mps', 'Wind1VelX'):
        _save(
            'compare_wind_mps',
            'Wind speed',
            'Wind speed (m/s)',
            baseline=(_col(wt_a, fmu_a, 'wind_speed_mps', 'Wind1VelX', side='baseline'),
                      legend_compare('baseline', detail='wind speed')),
            coupled=(_col(wt_a, fmu_a, 'wind_speed_mps', 'Wind1VelX', side='coupled'),
                     legend_compare('coupled', detail='inflow')),
            extra=wind_extra or None,
        )

    # --- Combined overlays (fewer figures, same signals) ---
    if _has_both(wt_a, fmu_a, 'vi_mag_pu', 'vi_mag_pu') and _has_both(wt_a, fmu_a, 'v_bus_pu', 'v_bus_pu'):
        _save(
            'compare_voltages_pu',
            'UIC voltages',
            'Voltage magnitude (p.u.)',
            baseline=(_col(wt_a, fmu_a, 'vi_mag_pu', 'vi_mag_pu', side='baseline'),
                      legend_compare('baseline', detail=r'$|v_i|$')),
            coupled=(_col(wt_a, fmu_a, 'vi_mag_pu', 'vi_mag_pu', side='coupled'),
                     legend_compare('coupled', detail=r'$|v_i|$')),
            extra=[
                (_col(wt_a, fmu_a, 'v_bus_pu', 'v_bus_pu', side='baseline'),
                 legend_compare('baseline', detail=r'$|V_t|$'),
                 COLOR_BASELINE_ALT, '-'),
                (_col(wt_a, fmu_a, 'v_bus_pu', 'v_bus_pu', side='coupled'),
                 legend_compare('coupled', detail=r'$|V_t|$'),
                 COLOR_COUPLED_ALT, '-'),
            ],
        )

    p_act, p_ref = 'P_uic_bus_actual_sys_pu', 'P_uic_bus_ref_sys_pu'
    q_act, q_ref = 'Q_uic_bus_actual_sys_pu', 'Q_uic_bus_ref_sys_pu'
    if _has_both(wt_a, fmu_a, p_act, p_act) and _has_both(wt_a, fmu_a, q_act, q_act):
        pq_extra = _ref_extra(wt_a, fmu_a, p_ref, p_ref, detail=r'$P_{\mathrm{ref},t}$')
        pq_extra.extend(_ref_extra(wt_a, fmu_a, q_ref, q_ref, detail=r'$Q_{\mathrm{ref},t}$'))
        pq_extra.extend([
            (_col(wt_a, fmu_a, q_act, q_act, side='baseline'),
             legend_compare('baseline', detail=r'$Q_t$'),
             COLOR_BASELINE_ALT, '-'),
            (_col(wt_a, fmu_a, q_act, q_act, side='coupled'),
             legend_compare('coupled', detail=r'$Q_t$'),
             COLOR_COUPLED_ALT, '-'),
        ])
        _save(
            'compare_uic_bus_PQ_pu',
            'UIC bus power',
            'Power (p.u., system base)',
            baseline=(_col(wt_a, fmu_a, p_act, p_act, side='baseline'),
                      legend_compare('baseline', detail=r'$P_t$')),
            coupled=(_col(wt_a, fmu_a, p_act, p_act, side='coupled'),
                     legend_compare('coupled', detail=r'$P_t$')),
            extra=pq_extra,
        )

    if _has_both(wt_a, fmu_a, 'P_e_sys_pu', 'P_e_sys_pu'):
        pwr_extra = _ref_extra(wt_a, fmu_a, 'P_ref_sys_pu', 'P_ref_sys_pu', detail=r'$P_{\mathrm{ref}}$')
        if 'P_aero_sys_pu' in wt_a.columns:
            pwr_extra.insert(0, (
                wt_a['P_aero_sys_pu'].to_numpy(dtype=float),
                legend_compare('baseline', detail=r'$P_{\mathrm{aero}}$'),
                COLOR_WIND,
                '-',
            ))
        _save(
            'compare_wt_power_pu',
            'Wind-turbine electrical power',
            'Power (p.u., system base)',
            baseline=(_col(wt_a, fmu_a, 'P_e_sys_pu', 'P_e_sys_pu', side='baseline'),
                      legend_compare('baseline', detail=r'$P_e$')),
            coupled=(_col(wt_a, fmu_a, 'P_e_sys_pu', 'P_e_sys_pu', side='coupled'),
                     legend_compare('coupled', detail=r'$P_e$')),
            extra=pwr_extra or None,
        )

    if _has_both(wt_a, fmu_a, 'P_inf_sys_pu', 'P_inf_sys_pu') and _has_both(wt_a, fmu_a, 'Q_inf_sys_pu', 'Q_inf_sys_pu'):
        _save(
            'compare_inf_bus_PQ_pu',
            'Infinite-bus power',
            'Power (p.u., system base)',
            baseline=(_col(wt_a, fmu_a, 'P_inf_sys_pu', 'P_inf_sys_pu', side='baseline'),
                      legend_compare('baseline', detail='P')),
            coupled=(_col(wt_a, fmu_a, 'P_inf_sys_pu', 'P_inf_sys_pu', side='coupled'),
                     legend_compare('coupled', detail='P')),
            extra=[
                (_col(wt_a, fmu_a, 'Q_inf_sys_pu', 'Q_inf_sys_pu', side='baseline'),
                 legend_compare('baseline', detail='Q'),
                 COLOR_BASELINE_ALT, '-'),
                (_col(wt_a, fmu_a, 'Q_inf_sys_pu', 'Q_inf_sys_pu', side='coupled'),
                 legend_compare('coupled', detail='Q'),
                 COLOR_COUPLED_ALT, '-'),
            ],
        )

    # FMU-only diagnostics overlaid when present (no baseline counterpart).
    if 'P_cmd_pu' in fmu_a.columns and _has_both(wt_a, fmu_a, 'P_ref_sys_pu', 'P_ref_sys_pu'):
        _save(
            'compare_P_cmd_vs_Pref',
            'Implied mechanical power vs reference',
            'Power (p.u., system base)',
            baseline=(_col(wt_a, fmu_a, 'P_ref_sys_pu', 'P_ref_sys_pu', side='baseline'),
                      legend_compare('baseline', detail=r'$P_{\mathrm{ref}}$', ref=True)),
            coupled=(fmu_a['P_cmd_pu'].to_numpy(dtype=float),
                     legend_compare('coupled', detail=r'$T\omega$ cmd')),
            baseline_ls=LS_REF,
        )

    # Mechanical torque: baseline MPT table vs coupled OpenFAST GenTq (WT torque base, pu).
    s_wt_mva = 15.0
    wt_omega_base_rpm = wt_base_rpm if np.isfinite(wt_base_rpm) and wt_base_rpm > 0 else common_base_rpm
    mpt_interp = _load_mpt_torque_interp(project_root)
    omega_for_mpt = None
    if 'T_mpt_wt_pu' in wt_a.columns:
        t_mpt_wt = wt_a['T_mpt_wt_pu'].to_numpy(dtype=float)
    elif 'omega_e_pu' in wt_a.columns and np.isfinite(wt_omega_base_rpm):
        omega_for_mpt = wt_a['omega_e_pu'].to_numpy(dtype=float)
        t_mpt_wt = _mpt_torque_wt_pu(omega_for_mpt, wt_omega_base_rpm, mpt_interp)
        wt_a['T_mpt_wt_pu'] = t_mpt_wt
    else:
        t_mpt_wt = None

    t_gentq_wt = None
    if 'GenTq' in fmu_a.columns and np.isfinite(common_base_rpm):
        t_gentq_wt = _torque_wt_pu_from_knm(
            fmu_a['GenTq'].to_numpy(dtype=float), common_base_rpm, s_wt_mva
        )
        fmu_a['GenTq_wt_pu'] = t_gentq_wt

    if t_mpt_wt is not None and t_gentq_wt is not None:
        _save(
            'compare_T_mech_wt_pu',
            'Generator torque (mechanical)',
            r'Torque (p.u., WT shaft base)',
            baseline=(t_mpt_wt, legend_compare('baseline', detail='MPT table')),
            coupled=(t_gentq_wt, legend_compare('coupled', detail='GenTq (ROSCO)')),
        )

        # Implied mechanical power from T*omega at generator speed (system base, pu).
        sys_s_mva = 10.0
        wt_to_sys = s_wt_mva / sys_s_mva
        if 'omega_e_pu' in wt_a.columns and np.isfinite(wt_omega_base_rpm):
            om_e_wt = wt_a['omega_e_pu'].to_numpy(dtype=float)
        elif 'omega_e_pu_common' in wt_a.columns:
            om_e_wt = (
                wt_a['omega_e_pu_common'].to_numpy(dtype=float)
                * common_base_rpm
                / wt_omega_base_rpm
            )
        else:
            om_e_wt = None
        if om_e_wt is not None:
            p_mpt_sys = t_mpt_wt * om_e_wt * wt_to_sys
            wt_a['P_mech_from_T_sys_pu'] = p_mpt_sys
            if 'GenSpeed' in fmu_a.columns and np.isfinite(common_base_rpm):
                om_e_fmu = _pu_from_rpm(
                    fmu_a['GenSpeed'].to_numpy(dtype=float), common_base_rpm
                )
                p_gentq_sys = t_gentq_wt * om_e_fmu * wt_to_sys
                fmu_a['P_mech_from_T_sys_pu'] = p_gentq_sys
                _save(
                    'compare_P_mech_from_T_sys_pu',
                    r'Implied mechanical power ($T\,\omega$)',
                    'Power (p.u., system base)',
                    baseline=(p_mpt_sys, legend_compare('baseline', detail=r'MPT $T\omega$')),
                    coupled=(p_gentq_sys, legend_compare('coupled', detail=r'GenTq $\omega$')),
                )

        t_ss = t_fmu
        print('\nSteady-state torque / speed (mean t >= 45 s):')
        print(f"  Baseline  omega_e_pu = {_steady_means(t_ss, wt_a['omega_e_pu'].to_numpy() if 'omega_e_pu' in wt_a.columns else np.full_like(t_ss, np.nan)):.4f}")
        print(f"  Coupled   omega_e_pu = {_steady_means(t_ss, fmu_a['omega_e_pu_common'].to_numpy() if 'omega_e_pu_common' in fmu_a.columns else np.full_like(t_ss, np.nan)):.4f}")
        print(f"  Baseline  T_MPT_wt_pu = {_steady_means(t_ss, t_mpt_wt):.4f}")
        print(f"  Coupled   GenTq_wt_pu = {_steady_means(t_ss, t_gentq_wt):.4f}")
        if 'P_e_sys_pu' in wt_a.columns and 'P_e_sys_pu' in fmu_a.columns:
            print(f"  Baseline  P_e_sys_pu  = {_steady_means(t_ss, wt_a['P_e_sys_pu'].to_numpy()):.4f}")
            print(f"  Coupled   P_e_sys_pu  = {_steady_means(t_ss, fmu_a['P_e_sys_pu'].to_numpy()):.4f}")
        if 'P_mech_from_T_sys_pu' in wt_a.columns and 'P_mech_from_T_sys_pu' in fmu_a.columns:
            print(f"  Baseline  T*omega sys = {_steady_means(t_ss, wt_a['P_mech_from_T_sys_pu'].to_numpy()):.4f}")
            print(f"  Coupled   T*omega sys = {_steady_means(t_ss, fmu_a['P_mech_from_T_sys_pu'].to_numpy()):.4f}")

    print(f"Saved {len(saved)} thesis comparison figure(s) to {plots_dir}")
    for p in saved:
        print(f"  {p}")

    if show and saved:
        import matplotlib.pyplot as plt

        plt.show(block=True)


if __name__ == '__main__':
    main()
