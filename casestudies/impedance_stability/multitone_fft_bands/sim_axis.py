"""Time-domain simulation for one multitone band (re or im axis)."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

import src.dynamic as dps
import src.solvers as dps_sol

PlantId = Literal["wt", "fmu", "uic"]


def _safe_legend(ax, *args, **kwargs):
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(*args, **kwargs)


def extract_injected_tones_from_model(model: dict) -> pd.DataFrame:
    pert = model.get("perturbations", {}) if isinstance(model, dict) else {}
    bc = pert.get("BusCurrentPerturbation") if isinstance(pert, dict) else None
    if not bc:
        return pd.DataFrame(columns=["name", "bus", "f_Hz", "amp_pu", "phase_rad", "axis"])
    rows = list(bc)
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    out = pd.DataFrame(
        {
            "name": df["name"] if "name" in df.columns else np.nan,
            "bus": df["bus"] if "bus" in df.columns else np.nan,
            "f_Hz": pd.to_numeric(df["f"], errors="coerce") if "f" in df.columns else np.nan,
            "amp_pu": pd.to_numeric(df["amp"], errors="coerce") if "amp" in df.columns else np.nan,
            "phase_rad": pd.to_numeric(df["phase"], errors="coerce") if "phase" in df.columns else np.nan,
            "axis": df["axis"] if "axis" in df.columns else np.nan,
        }
    )
    out = out[np.isfinite(out["f_Hz"].to_numpy(dtype=float)) & (out["f_Hz"].to_numpy(dtype=float) > 0.0)]
    return out.sort_values(["f_Hz", "name"], kind="stable").reset_index(drop=True)


def run_axis(
    *,
    plant: PlantId,
    case_id: str,
    run_axis: str,
    band_tag: str,
    t_end_s: float,
    dt_s: float,
    log_dir: str,
    plot_sanity: bool = True,
    model: dict | None = None,
) -> None:
    t_start_wall = time.perf_counter()
    if model is None:
        from casestudies.impedance_stability.ps_data.cases import CaseLoader

        model = CaseLoader(case_id).load()
    os.makedirs(log_dir, exist_ok=True)
    print(
        f"Log directory: {log_dir} (plant={plant}, band={band_tag}, axis={run_axis}, case={case_id})",
        flush=True,
    )

    ps = dps.PowerSystemModel(model=model)
    uic_model = ps.vsc["UIC_sig"]

    if plant == "wt":
        wt_model = ps.windturbine["WindTurbine"]
        wind_speed = wt_model.wind_speed_init()
        P_ref = wt_model.P_ref_from_wind(wind_speed, uic_model.par["S_n"])
        uic_model.par["p_ref"][:] = P_ref
        uic_model.par["q_ref"][:] = 0.0
    elif plant == "uic":
        # Standalone UIC: use p_ref / q_ref from the PS case (vsc UIC_sig parameters).
        # No windturbine block; do not overwrite setpoints from MPPT.
        pass
    else:
        uic_model.par["p_ref"][:] = 0.75
        uic_model.par["q_ref"][:] = 0.0

    print("Running power flow...", flush=True)
    ps.power_flow()
    print("Initializing dynamic simulation (init_dyn_sim)...", flush=True)
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    fmu_models = [mdl for mdl in ps.dyn_mdls if hasattr(mdl, "step_fmu")] if plant == "fmu" else []
    fmu_mdl = fmu_models[0] if fmu_models else None
    fmu_outputs_stored: list[dict] = []
    fmu_cmd_stored: list[dict] = []

    result_dict = defaultdict(list)
    f_ode = lambda t_, x_: ps.state_derivatives(t_, x_, ps.solve_algebraic(t_, x_))
    sol = dps_sol.SimpleRK4(f_ode, 0.0, x0, float(t_end_s), max_step=float(dt_s))

    v_t_re: list[float] = []
    v_t_im: list[float] = []
    v_t_abs: list[float] = []
    v_t_ang: list[float] = []
    i_a_re: list[float] = []
    i_a_im: list[float] = []
    i_a_abs: list[float] = []
    i_a_ang: list[float] = []
    i_pert_re: list[float] = []
    i_pert_im: list[float] = []
    i_pert_abs: list[float] = []
    i_pert_ang: list[float] = []

    pert_mdl = None
    if hasattr(ps, "perturbations") and isinstance(ps.perturbations, dict):
        pert_mdl = ps.perturbations.get("BusCurrentPerturbation")

    sys_s_n = ps.sys_data["s_n"]
    uic_s_n = uic_model.par["S_n"][0]
    i_uic_to_sys = float(uic_s_n / sys_s_n)

    v = ps.solve_algebraic(0.0, x0)
    result_dict["Global", "t"].append(0.0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]

    if plant == "fmu" and fmu_mdl is not None:
        from casestudies.impedance_stability.multitone_fft_bands.fmu_log import (
            append_fmu_command_row,
            seed_fmu_output_row,
        )

        fmu_outputs_stored.append(seed_fmu_output_row(fmu_mdl, 0.0))
        fmu_cmd_stored.append(append_fmu_command_row(fmu_mdl))

    i_a0 = uic_model.i_a(x0, v)[0]
    v_t0 = uic_model.v_t(x0, v)[0]
    v_t_re.append(float(np.real(v_t0)))
    v_t_im.append(float(np.imag(v_t0)))
    v_t_abs.append(float(np.abs(v_t0)))
    v_t_ang.append(float(np.angle(v_t0)))
    i_a_re.append(float(np.real(i_a0)))
    i_a_im.append(float(np.imag(i_a0)))
    i_a_abs.append(float(np.abs(i_a0)))
    i_a_ang.append(float(np.angle(i_a0)))

    if pert_mdl is not None:
        _, i_pert_vec0 = pert_mdl.current_injections(x0, v)
        i_pert0 = np.sum(np.asarray(i_pert_vec0, dtype=complex))
    else:
        i_pert0 = 0.0 + 0.0j
    i_pert_re.append(float(np.real(i_pert0)))
    i_pert_im.append(float(np.imag(i_pert0)))
    i_pert_abs.append(float(np.abs(i_pert0)))
    i_pert_ang.append(float(np.angle(i_pert0)))

    t = 0.0
    while t < float(t_end_s):
        sys.stdout.write("\r%d%%" % int(t / float(t_end_s) * 100))
        sol.step()
        x = sol.x
        t = float(sol.t)
        v = ps.solve_algebraic(t, x)
        for mdl in fmu_models:
            mdl.step_fmu(x, v, t, float(dt_s))

        if plant == "fmu" and fmu_mdl is not None:
            from casestudies.impedance_stability.multitone_fft_bands.fmu_log import (
                append_fmu_command_row,
                append_fmu_output_row,
            )

            fmu_outputs_stored.append(append_fmu_output_row(fmu_mdl, t))
            fmu_cmd_stored.append(append_fmu_command_row(fmu_mdl))

        result_dict["Global", "t"].append(t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        i_a = uic_model.i_a(x, v)[0]
        v_t = uic_model.v_t(x, v)[0]
        v_t_re.append(float(np.real(v_t)))
        v_t_im.append(float(np.imag(v_t)))
        v_t_abs.append(float(np.abs(v_t)))
        v_t_ang.append(float(np.angle(v_t)))
        i_a_re.append(float(np.real(i_a)))
        i_a_im.append(float(np.imag(i_a)))
        i_a_abs.append(float(np.abs(i_a)))
        i_a_ang.append(float(np.angle(i_a)))

        if pert_mdl is not None:
            _, i_pert_vec = pert_mdl.current_injections(x, v)
            i_pert = np.sum(np.asarray(i_pert_vec, dtype=complex))
        else:
            i_pert = 0.0 + 0.0j
        i_pert_re.append(float(np.real(i_pert)))
        i_pert_im.append(float(np.imag(i_pert)))
        i_pert_abs.append(float(np.abs(i_pert)))
        i_pert_ang.append(float(np.angle(i_pert)))

    print("\r100%", flush=True)

    for mdl in fmu_models:
        if hasattr(mdl, "terminate_fmu"):
            mdl.terminate_fmu()

    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    t_stored = result[("Global", "t")].to_numpy(dtype=float)
    ic = np.asarray(i_a_re, dtype=float) + 1j * np.asarray(i_a_im, dtype=float)

    if plant == "wt":
        # Log WT shaft states for tone-based excitation plots.
        wt_name = "WT1"
        required = ("omega_m", "omega_e", "theta_m", "theta_e")
        missing = [k for k in required if (wt_name, k) not in result.columns]
        if missing:
            raise KeyError(
                f"WT shaft logging missing columns for {wt_name}: {missing}. "
                f"Available WT columns: {[c for c in result.columns if c and c[0] == wt_name][:20]} ..."
            )
        wt_df = pd.DataFrame(
            {
                "t": t_stored,
                "omega_m_pu": result[(wt_name, "omega_m")].to_numpy(dtype=float),
                "omega_e_pu": result[(wt_name, "omega_e")].to_numpy(dtype=float),
                "theta_m_pu": result[(wt_name, "theta_m")].to_numpy(dtype=float),
                "theta_e_pu": result[(wt_name, "theta_e")].to_numpy(dtype=float),
            }
        )
        wt_df["omega_s_pu"] = wt_df["omega_m_pu"] - wt_df["omega_e_pu"]
        wt_df["theta_s_pu"] = wt_df["theta_m_pu"] - wt_df["theta_e_pu"]
        wt_path = os.path.join(log_dir, f"wt_shaft_states_multisine_{band_tag}_{run_axis}.csv")
        wt_df.to_csv(wt_path, index=False)
        print(f"WT shaft states saved to {wt_path}", flush=True)
    elif plant == "fmu":
        # Log FMU drivetrain states for tone-based excitation plots.
        # omega_e and theta_s are DAE states in FMUtoUICdrivetrain; omega_m is measured from the FMU.
        required = ("omega_e", "theta_s")
        candidates = sorted({c[0] for c in result.columns if len(c) >= 2 and c[1] in required})
        fmu_name = None
        for name in candidates:
            if all((name, k) in result.columns for k in required):
                fmu_name = name
                break
        if fmu_name is None:
            raise KeyError(
                f"FMU shaft logging missing columns {required}. "
                f"Available model/state pairs (first 20): {list(result.columns)[:20]} ..."
            )

        omega_e_pu = result[(fmu_name, "omega_e")].to_numpy(dtype=float)
        theta_s_pu = result[(fmu_name, "theta_s")].to_numpy(dtype=float)

        omega_m_pu = np.full_like(omega_e_pu, np.nan, dtype=float)
        if fmu_cmd_stored:
            cmd_df = pd.DataFrame(fmu_cmd_stored)
            if "omega_m_pu_meas" in cmd_df.columns:
                om = pd.to_numeric(cmd_df["omega_m_pu_meas"], errors="coerce").to_numpy(dtype=float)
                if om.size == omega_m_pu.size:
                    omega_m_pu = om

        fmu_shaft_df = pd.DataFrame(
            {
                "t": t_stored,
                "omega_m_pu": omega_m_pu,
                "omega_e_pu": omega_e_pu,
                "theta_s_pu": theta_s_pu,
            }
        )
        fmu_shaft_df["omega_s_pu"] = fmu_shaft_df["omega_m_pu"] - fmu_shaft_df["omega_e_pu"]
        fmu_path = os.path.join(log_dir, f"fmu_shaft_states_multisine_{band_tag}_{run_axis}.csv")
        fmu_shaft_df.to_csv(fmu_path, index=False)
        print(f"FMU shaft states saved to {fmu_path}", flush=True)

    uic_vi_df = pd.DataFrame(
        {
            "t": t_stored,
            "v_t_re_pu": np.asarray(v_t_re, dtype=float),
            "v_t_im_pu": np.asarray(v_t_im, dtype=float),
            "v_t_abs_pu": np.asarray(v_t_abs, dtype=float),
            "v_t_ang_rad": np.asarray(v_t_ang, dtype=float),
            "i_a_re_pu": np.asarray(i_a_re, dtype=float),
            "i_a_im_pu": np.asarray(i_a_im, dtype=float),
            "i_a_abs_pu": np.asarray(i_a_abs, dtype=float),
            "i_a_ang_rad": np.asarray(i_a_ang, dtype=float),
            "i_a_sys_re_pu": np.real(ic * i_uic_to_sys).astype(float),
            "i_a_sys_im_pu": np.imag(ic * i_uic_to_sys).astype(float),
            "i_a_sys_abs_pu": np.abs(ic * i_uic_to_sys).astype(float),
            "i_pert_re_pu": np.asarray(i_pert_re, dtype=float),
            "i_pert_im_pu": np.asarray(i_pert_im, dtype=float),
            "i_pert_abs_pu": np.asarray(i_pert_abs, dtype=float),
            "i_pert_ang_rad": np.asarray(i_pert_ang, dtype=float),
            "S_sys_MVA": float(sys_s_n),
            "S_uic_MVA": float(uic_s_n),
        }
    )

    from casestudies.impedance_stability.multitone_fft_bands.fmu_log import (
        fmu_diag_csv_name,
        fmu_diag_dataframe,
        injected_tones_csv_name,
        vi_csv_name,
    )

    tones_df = extract_injected_tones_from_model(model)
    tones_path = os.path.join(log_dir, injected_tones_csv_name(band_tag, run_axis))
    tones_df.to_csv(tones_path, index=False)
    print(f"Injected tones saved to {tones_path} ({len(tones_df)} rows)", flush=True)

    vi_path = os.path.join(log_dir, vi_csv_name(band_tag, run_axis))
    uic_vi_df.to_csv(vi_path, index=False)
    print(f"UIC terminal V/I saved to {vi_path}", flush=True)

    if plant == "fmu" and fmu_outputs_stored:
        fmu_df = fmu_diag_dataframe(t_stored, fmu_outputs_stored, fmu_cmd_stored)
        if fmu_df is not None:
            fmu_path = os.path.join(log_dir, fmu_diag_csv_name(band_tag, run_axis))
            fmu_df.to_csv(fmu_path, index=False)
            print(f"FMU diagnostics saved to {fmu_path}", flush=True)

    if plot_sanity:
        import matplotlib.pyplot as plt

        t_arr = uic_vi_df["t"].to_numpy(dtype=float)
        v_dev = uic_vi_df["v_t_abs_pu"].to_numpy(dtype=float) - float(
            np.mean(uic_vi_df["v_t_abs_pu"])
        )
        ip_abs_arr = uic_vi_df["i_pert_abs_pu"].to_numpy(dtype=float)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5.0))
        ax[0].plot(t_arr, v_dev, lw=1.0, label="|V_t| - mean")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_ylabel("Δ|V_t| (pu)")
        _safe_legend(ax[0], loc="best", fontsize=8)
        ax[1].plot(t_arr, ip_abs_arr, lw=1.0, label="|i_pert|")
        ax[1].grid(True, alpha=0.3)
        ax[1].set_ylabel("|i_pert| (pu)")
        ax[1].set_xlabel("Time (s)")
        _safe_legend(ax[1], loc="best", fontsize=8)
        fig.tight_layout()
        vi_png = vi_path.replace(".csv", ".png")
        fig.savefig(vi_png, dpi=150)
        plt.close(fig)
        print(f"Sanity plot saved to {vi_png}", flush=True)

        if plant == "fmu" and fmu_outputs_stored:
            fmu_df = fmu_diag_dataframe(t_stored, fmu_outputs_stored, fmu_cmd_stored)
            if fmu_df is not None and "YawBrTAyp" in fmu_df.columns:
                y = fmu_df["YawBrTAyp"].to_numpy(dtype=float)
                if np.isfinite(y).any():
                    fig_ss, ax_ss = plt.subplots(2, 1, sharex=True, figsize=(10, 5.0))
                    if "RtAeroMxh" in fmu_df.columns and np.isfinite(fmu_df["RtAeroMxh"]).any():
                        ax_ss[0].plot(
                            fmu_df["t"],
                            fmu_df["RtAeroMxh"],
                            lw=1.0,
                            label="RtAeroMxh (kN·m)",
                        )
                        ax_ss[0].set_ylabel("Hub Mx")
                        _safe_legend(ax_ss[0], loc="best", fontsize=8)
                        ax_ss[0].grid(True, alpha=0.3)
                    ax_ss[1].plot(
                        fmu_df["t"],
                        y,
                        lw=1.0,
                        label="YawBrTAyp (tower side-side, m/s²)",
                    )
                    ax_ss[1].set_ylabel("Tower accel")
                    ax_ss[1].set_xlabel("Time (s)")
                    _safe_legend(ax_ss[1], loc="best", fontsize=8)
                    ax_ss[1].grid(True, alpha=0.3)
                    fig_ss.tight_layout()
                    if band_tag and str(band_tag).strip():
                        ss_name = f"fmu_tower_ss_{band_tag}_{run_axis}.png"
                    else:
                        ss_name = f"fmu_tower_ss_{run_axis}.png"
                    ss_png = os.path.join(log_dir, ss_name)
                    fig_ss.savefig(ss_png, dpi=150)
                    plt.close(fig_ss)
                    print(f"FMU side-side sanity plot saved to {ss_png}", flush=True)

    print(
        f"Axis {run_axis} band {band_tag} wall time: {time.perf_counter() - t_start_wall:.2f}s",
        flush=True,
    )
