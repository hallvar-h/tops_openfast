from src.dyn_models.utils import DAEModel
from src.dyn_models.speed_lpf import (
    apply_speed_lpf_dynamics,
    resolve_speed_lpf_params,
    speed_pu_for_use,
)
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq
import os

class WindTurbine(DAEModel):
    """
    'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'S_n', 'V_n',         'J_m',             'J_e',             'K',          'D',        'Kp_pitch',     'Ki_pitch',   'T_pitch', 'max_pitch', 'min_pitch', 'max_pitch_rate',     'rho',     'R',      'P_rated', 'omega_m_rated', 'wind_rated', 'efficiency','MPT_filename', 'Cp_filename', 'speed_lpf_type', 'speed_lpf_corner_rad_s', 'speed_lpf_damping'],
            ['WT1', 'UIC1',  15,    22,          310619488.,        1836784,        697376449.,    71186519.,       0.66,           0.2,           0.1,         90,           0,           2,              1.225,    120.97,       1.0,       7.53,      10.6,   0.95,           'MPT_Kopt2150.csv', 'Cp_Kopt2150.csv', 2,               1.00810,                0.70000]
            # [-,     -,     MW,     kV,           kg m^2,           kg m^2,          Nm/rad,       Nms/rad,        rad/pu,         rad/pu,        s,            deg,         deg,         deg/s,          kg/m^3,     m,          pu,         RPM,        m/s,  efficiency (0-1), -, -, 0/1/2,              rad/s,                  -]
        ],
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        sn = self.par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        self.par['S_n'] = sn
        self._sys_to_local = self.sys_par['s_n'] / self.par['S_n']
        self._local_to_sys = self.par['S_n'] / self.sys_par['s_n']

        # Convert omega_m_rated from RPM to rad/s
        RPM_to_rad_per_s = 2 * np.pi / 60  # 1 RPM = 2π/60 rad/s
        self.par['omega_m_rated'] = self.par['omega_m_rated'] * RPM_to_rad_per_s
        
        self._debug_counter = 0
        
        # Load wind data from .hh file for variable wind speed
        # File format: first line is number of columns, then time (col 1) and wind speed in m/s (col 2)
        wind_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'wind_data', '10mps_NTM_3xDTU10MW_IECKAI_VS_T1.hh')
        wind_data = np.loadtxt(wind_file_path, skiprows=1, usecols=(0, 1))
        wind_times = wind_data[:, 0]  # First column: time in seconds
        wind_speeds = wind_data[:, 1]  # Second column: wind speed in m/s
        # Create interpolation function for smooth wind speed transitions
        # Use linear interpolation for natural smooth transitions
        self._wind_interp = interp1d(wind_times, wind_speeds, kind='linear', 
                                     bounds_error=False, fill_value=(wind_speeds[0], wind_speeds[-1]))

        # convert all WT params to pu:
        w_m_base = self.par['omega_m_rated']  # rad/s
        T_base = self.par['S_n'] * 1e6 / w_m_base  # Nm
        
        # Calculate H_m and H_e from J_m and J_e as instance variables (arrays, one per unit)
        self.H_m = 0.5 * self.par['J_m'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.H_e = 0.5 * self.par['J_e'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.par['K'] = self.par['K'] / T_base
        self.par['D'] = self.par['D'] * w_m_base / T_base
        self.par['max_pitch'] = self.par['max_pitch'] * np.pi / 180
        self.par['min_pitch'] = self.par['min_pitch'] * np.pi / 180
        self.par['max_pitch_rate'] = self.par['max_pitch_rate'] * np.pi / 180

        # Speed LPF on rotor speed for MPT lookup (optional; speed_lpf_type=0 disables).
        # Keep safe defaults if the parameters are not provided in the model data.
        #
        # NOTE: `self.par` is typically a NumPy structured array; we cannot add new
        # fields at runtime, so we store resolved defaults in instance attributes.
        n_units = int(np.asarray(self.par['S_n']).size)
        (
            self._speed_lpf_type,
            self._speed_lpf_corner_rad_s,
            self._speed_lpf_damping,
        ) = resolve_speed_lpf_params(self.par, n_units)

        # Gen-speed LPF in Te denominator (ROSCO); keeps MPT on raw omega_m when speed_lpf_type=0.
        self._te_speed_lpf_type = 2

    def connections(self):
        return [
            {
                'input': 'P_e',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'p_e',
            },
            {
                'input': 'S_n_UIC',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'S_n',
            },
            {
                'output': 'P_ref',
                'destination': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'input': 'p_ref',
            }
        ]

    def state_list(self):
        # speed_lpf_* applies to omega_m_filt only (rotor / MPT path).
        #   omega_m_filt[*]  — rotor speed for MPT (input omega_m)
        # Modal note: torsional mode ~3–4 Hz remains in eigs; mode shape is gen-dominated
        # (H_e << H_m), not removed by LPF. Use freq_range > 3 Hz in get_mode_idx to list it.
        return [
            'omega_m',
            'omega_e',
            'theta_m',
            'theta_e',
            'pitch_PI_integral_state',
            'pitch_angle',
            'omega_m_filt',
            'omega_m_filt_dot',
            'omega_e_filt',
            'omega_e_filt_dot',
            # --- Previous: P_ref on filtered gen speed (MPT now on omega_m_filt):
            # (omega_e_filt was also used here)
        ]

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par
        P_aero = self.P_aero(x, v)
        Pe = self.P_e(x, v) * self.S_n_UIC(x, v) / par['S_n'] # UIC pu -> WT pu
        Ta = P_aero / X['omega_m'] if X['omega_m'] > 0 else 0
        # Backwards compatible: accept legacy 'gb_gen_efficiency' if present
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0
        Tm = Ta # could have rotor efficiency here
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0])
        omega_c = float(np.asarray(self._speed_lpf_corner_rad_s).ravel()[0])
        zeta = float(np.asarray(self._speed_lpf_damping).ravel()[0])

        # --- Torque coupling: Te = Pe / (omega_e_filt * eta) ---
        te_lpf_type = int(np.asarray(self._te_speed_lpf_type).ravel()[0])
        omega_e_filtered = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', te_lpf_type)
        if not np.isfinite(omega_e_filtered) or omega_e_filtered <= 1e-3:
            omega_e_filtered = 1e-3
        Te = Pe / (omega_e_filtered * eta) if np.isfinite(Pe) else 0.0
        Te_for_swing = Te

        # shaft torque
        theta_s = X['theta_m'] - X['theta_e']
        omega_s = X['omega_m'] - X['omega_e']
        T_shaft = par['K'] * theta_s + par['D'] * omega_s

        # swing eqs for wt dynamics
        dX['omega_m'] = (1/(2*self.H_m)) * (Tm - T_shaft)
        dX['omega_e'] = (1/(2*self.H_e)) * (T_shaft - Te_for_swing)
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']

        max_pitch = par['max_pitch'][0]
        min_pitch = par['min_pitch'][0]
        max_pitch_rate = par['max_pitch_rate'][0]
        omega_ref = 1.0 # 'hardcoded' as the rated speed from init willl always be 1 pu
        e_omega = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type) - omega_ref
        pitch_reference_pi = 0.0

        if e_omega < 0.:
            # Region 2: below rated speed - MPPT, reset integral
            dX_pitch_integral = 0.0
            pitch_reference_pi = min_pitch
        else:  # Region 3: at or above rated speed - pitch to limit power
            # Calculate controller output to check for anti-windup
            PIctrl_integral_term = par['Ki_pitch'][0] * X['pitch_PI_integral_state'][0]
            PIctrl_proportional_term = par['Kp_pitch'][0] * e_omega
            pitch_reference_unclamped = PIctrl_integral_term + PIctrl_proportional_term
            
            # Anti-windup -> stops integration term when reference is at limit to prevent over- and undershoots
            if pitch_reference_unclamped >= max_pitch or pitch_reference_unclamped <= min_pitch:
                dX_pitch_integral = 0.0  # Stop integrating when output hits limits
            else:
                dX_pitch_integral = e_omega  # Normal integration
            
            # Clamp pitch_reference to max and min pitch angle
            pitch_reference_pi = np.clip(pitch_reference_unclamped, min_pitch, max_pitch)
        
        dX['pitch_PI_integral_state'] = dX_pitch_integral
        # DTU-style servo: T_pitch drives pitch_angle toward PI demand, subject to rate limit
        delta_pitch_angle = (1/par['T_pitch'][0]) * (pitch_reference_pi - X['pitch_angle'][0])
        dX['pitch_angle'] = np.clip(delta_pitch_angle, -max_pitch_rate, max_pitch_rate)

        omega_m = np.asarray(X['omega_m']).ravel()
        apply_speed_lpf_dynamics(dX, X, omega_m, 'omega_m_filt', 'omega_m_filt_dot', lpf_type, omega_c, zeta)
        te_lpf_type = int(np.asarray(self._te_speed_lpf_type).ravel()[0])
        omega_e = np.asarray(X['omega_e']).ravel()
        apply_speed_lpf_dynamics(dX, X, omega_e, 'omega_e_filt', 'omega_e_filt_dot', te_lpf_type, omega_c, zeta)

        """ self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 2 or self._debug_counter == 3 or self._debug_counter == 4 or self._debug_counter == 5 or self._debug_counter == 6 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            print('Debug values (iteration', self._debug_counter, '):')
            print('  X[omega_m]:', X['omega_m'])
            print('  X[omega_e]:', X['omega_e'])
            print('  X[theta_m]:', X['theta_m'])
            print('  X[theta_e]:', X['theta_e'])
            print('dX[omega_m]:', dX['omega_m'])
            print('dX[omega_e]:', dX['omega_e'])
            print('dX[theta_m]:', dX['theta_m'])
            print('dX[theta_e]:', dX['theta_e'])
            #print('dX[pitch_PI_integral_state]:', dX['pitch_PI_integral_state'])
            #print('  X[pitch_PI_integral_state]:', X['pitch_PI_integral_state'])
            #print('dX[pitch_angle]:', dX['pitch_angle'])
            print('  X[pitch_angle]:', X['pitch_angle'])
            #print('  pitch_angle:', self._pitch_angle)
            print('  P_aero (WT local pu):', P_aero)
            print('  Pe (WT local pu):', Pe)
            print('  P_ref (UIC pu):', self.P_ref(x, v))
            print('cp: ', self.load_and_set_Cp(x, v))
            print('H_m:', np.asarray(self.H_m, dtype=float))
            print('H_e:', np.asarray(self.H_e, dtype=float))
            print('par[K]:', np.asarray(par['K'], dtype=float))
            print('par[D]:', np.asarray(par['D'], dtype=float))
            #print('  omega_m_ref:', omega_m_ref) """

        if self._debug_counter > 100 and X['omega_m'][0] > 10:
            print('solution blowing up, omega_m:', X['omega_m'][0])

        return

    def _pe_local_wt_pu(self, x, v) -> float:
        """Electrical power from UIC (WT local pu on S_n)."""
        pe_uic = float(np.asarray(self.P_e(x, v)).ravel()[0])
        s_n_uic = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        s_n_loc = float(np.asarray(self.par['S_n']).ravel()[0])
        return pe_uic * (s_n_uic / s_n_loc) if s_n_loc > 0 else pe_uic

    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        self._input_values['P_e'] = self.P_e(x_0, v_0)
        self._input_values['S_n_UIC'] = self.S_n_UIC(x_0, v_0)

        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        u_rated = float(np.asarray(par['wind_rated']).ravel()[0])
        u_start = float(np.asarray(self.wind_speed(x_0, v_0)).ravel()[0])
        K = float(np.asarray(par['K']).ravel()[0])

        self._load_MPT_table()
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0

        if u_start >= u_rated * 0.99:
            omega_m_init_pu = 1.0
        else:
            # Region 2: P_aero = T_mech*omega_pu (MPPT mechanical target).
            def _res(om):
                X['omega_m'] = om
                X['omega_e'] = om
                X['pitch_angle'] = 0.0
                return float(self.P_aero(x_0, v_0).ravel()[0]) - self._mpt_power_mech_pu(
                    om * w_rated, om
                )

            try:
                omega_m_init_pu = float(brentq(_res, 0.05, 1.0))
            except ValueError:
                omega_m_init_pu = float(np.clip(u_start / u_rated, 0.05, 0.98))
                print(
                    'Brentq omega_m init failed; using u/u_rated =',
                    omega_m_init_pu,
                )
        X['omega_m'] = omega_m_init_pu
        X['omega_e'] = omega_m_init_pu
        X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))

        pe_loc_pu = self._pe_local_wt_pu(x_0, v_0)

        if omega_m_init_pu >= 0.99:
            # Region 3: rated speed + pitch for P_aero = P_e/eta.
            min_pitch = float(np.asarray(par['min_pitch']).ravel()[0])
            max_pitch = float(np.asarray(par['max_pitch']).ravel()[0])
            Ki = float(np.asarray(par['Ki_pitch']).ravel()[0])
            X['omega_m'] = 1.0
            X['omega_e'] = 1.0
            self.load_and_set_Cp(x_0, v_0)

            def _res_pitch(pitch_rad):
                X['pitch_angle'] = pitch_rad
                # P_aero (mech) = P_e/eta at rated speed
                return float(self.P_aero(x_0, v_0).ravel()[0]) - pe_loc_pu / eta
            try:
                pitch_eq = brentq(_res_pitch, min_pitch, max_pitch) # again brentq solves for where _res_pitch func is 0 -> pitch is the right val for P_aero = 1 pu
            except ValueError:
                # Fallback: choose the pitch end that best matches the target
                r_min = _res_pitch(min_pitch)
                r_max = _res_pitch(max_pitch)
                pitch_eq = min_pitch if abs(r_min) <= abs(r_max) else max_pitch
                print('Brentq pitch init failed, using endpoint closest to power balance')
            pitch_eq = np.clip(pitch_eq, min_pitch, max_pitch)
            X['pitch_angle'] = pitch_eq
            X['pitch_PI_integral_state'] = pitch_eq / Ki if Ki > 0 else 0.0
            self._pitch_angle = pitch_eq
            # Recompute shaft twist so that T_shaft = T_e at init (omega_s=0).
            # This makes omega_m/omega_e start with (approximately) zero acceleration.
            Te = pe_loc_pu / (float(np.asarray(X['omega_e']).ravel()[0]) * eta) if float(np.asarray(X['omega_e']).ravel()[0]) > 0 else 0.0
            theta_s = Te / K
            X['theta_m'] = 0.0
            X['theta_e'] = -theta_s
        else:
            # Region 2: MPPT, pitch at minimum (typically 0)
            X['pitch_PI_integral_state'] = 0.0
            X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))
            self._pitch_angle = float(np.asarray(X['pitch_angle']).ravel()[0])
            om_e0 = float(np.asarray(X['omega_e']).ravel()[0])
            Te = pe_loc_pu / (om_e0 * eta) if om_e0 > 0 else 0.0
            theta_s = Te / K
            X['theta_m'] = 0.0
            X['theta_e'] = -theta_s

        om_m = np.asarray(X['omega_m'], dtype=float).ravel()
        X['omega_m_filt'] = om_m.copy()
        X['omega_m_filt_dot'] = np.zeros_like(om_m)
        om_e = np.asarray(X['omega_e'], dtype=float).ravel()
        X['omega_e_filt'] = om_e.copy()
        X['omega_e_filt_dot'] = np.zeros_like(om_e)

        return

    def P_aero(self, x, v):
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.load_and_set_Cp(x, v)
        
        # Aerodynamic power from wind: P = 0.5 * rho * A * v^3 * Cp
        # A = pi * R^2 (swept area)
        P_aero_watts = 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp
        
        # Convert to per-unit using local base power
        S_base_watts = self.par['S_n'] * 1e6  # Convert MVA to Watts, WT local base
        P_aero_pu = P_aero_watts / S_base_watts
        
        return P_aero_pu  # WT pu
 
    def P_ref(self, x, v):
        X = self.local_view(x)
        par = self.par
        lpf_type = int(np.asarray(self._speed_lpf_type).ravel()[0]) if hasattr(self, '_speed_lpf_type') else 1
        filtered_omega_e_pu = speed_pu_for_use(X, 'omega_e', 'omega_e_filt', lpf_type)
        omega_rated = float(np.asarray(self.par['omega_m_rated']).ravel()[0])
        gen_speed_rad_s = filtered_omega_e_pu * omega_rated
        P_elec_wt_pu = self._mpt_power_elec_pu(gen_speed_rad_s, filtered_omega_e_pu)
        return np.atleast_1d(P_elec_wt_pu * par['S_n'] / self.S_n_UIC(x, v))

    def P_ref_from_wind(self, wind_speed_mps, S_n_UIC):
        """P_ref in UIC pu. Same MPPT root as init (P_aero = T_mech*omega_pu)."""
        wind_speed_mps = float(np.asarray(wind_speed_mps).ravel()[0])
        self._load_MPT_table()
        self.load_and_set_Cp(None, None)  # load Cp table only
        par = self.par
        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        R = float(np.asarray(par['R']).ravel()[0])
        rho = float(np.asarray(par['rho']).ravel()[0])
        S_n = float(np.asarray(par['S_n']).ravel()[0])
        eta = float(np.asarray(par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0

        def _res(om):
            omega_rad = om * w_rated
            tsr = omega_rad * R / wind_speed_mps if wind_speed_mps > 0 else 0
            pa = np.clip(0.0, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
            tsr_c = np.clip(tsr, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())
            Cp = float(self._cp_interp(np.array([pa, tsr_c]))[0])
            P_aero = 0.5 * rho * np.pi * R**2 * wind_speed_mps**3 * Cp / (S_n * 1e6)
            return P_aero - self._mpt_power_mech_pu(omega_rad, om)

        try:
            omega_init = float(brentq(_res, 0.05, 1.0))
        except ValueError:
            lam_ref = R * w_rated / float(np.asarray(par['wind_rated']).ravel()[0])
            omega_init = float(np.clip(lam_ref * wind_speed_mps / R / w_rated, 0.05, 1.0))
        P_elec_wt_pu = self._mpt_power_elec_pu(omega_init * w_rated, omega_init)
        return P_elec_wt_pu * S_n / float(np.asarray(S_n_UIC).ravel()[0])

    def _mpt_power_mech_pu(self, omega_rad_s, omega_pu):
        """Mechanical power (WT pu): T_mech * omega_pu. No eta here."""
        return self._mpt_torque_mech_pu(omega_rad_s) * float(omega_pu)

    def _mpt_power_elec_pu(self, omega_rad_s, omega_pu):
        """Electrical power (WT pu) for UIC p_ref: P_e = eta * P_m. Only eta in the chain."""
        eta = float(np.asarray(self.par['efficiency']).ravel()[0])
        eta = eta if np.isfinite(eta) and eta > 0 else 1.0
        return eta * self._mpt_power_mech_pu(omega_rad_s, omega_pu)

    def _mpt_torque_mech_pu(self, omega_rad_s):
        """Mechanical torque (WT pu) from MPT_T_* (table values are T_mech, user-scaled)."""
        self._load_MPT_table()
        return float(self._mpt_torque_interp(omega_rad_s))

    def _load_MPT_table(self):
        """Load MPT_T_* torque (pu on WT shaft base). Values are T_mech; P_e = eta*T_mech*omega_pu."""
        if hasattr(self, '_mpt_torque_interp'):
            return
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mpt_filename = self.par['MPT_filename'][0] if isinstance(self.par['MPT_filename'], np.ndarray) else self.par['MPT_filename']
        mpt_t_filename = str(mpt_filename).replace('MPT_', 'MPT_T_', 1)
        path = os.path.join(project_root, 'wind_data', mpt_t_filename)
        data = np.loadtxt(path, delimiter='\t')
        rotor_speed_RPM = data[2:, 0]
        torque_mech_pu = data[2:, 1]
        rotor_speed_rad_s = rotor_speed_RPM * (2 * np.pi / 60)
        self._mpt_torque_interp = interp1d(
            rotor_speed_rad_s,
            torque_mech_pu,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, torque_mech_pu[-1]),
        )

    def load_and_set_Cp(self, x, v):
        par = self.par
        # Load Cp data if not already loaded
        if not hasattr(self, '_cp_data'):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            cp_filename = self.par['Cp_filename'][0] if isinstance(self.par['Cp_filename'], np.ndarray) else self.par['Cp_filename']
            path = os.path.join(project_root, 'wind_data', cp_filename)
            with open(path, 'r') as f:
                lines = f.readlines()
            pitch_line = lines[4].strip()
            if pitch_line.startswith('#'):
                pitch_line = pitch_line[1:].strip()
            pitch_angles = np.array([float(x) for x in pitch_line.split()])
            tsr_line = lines[6].strip()
            if tsr_line.startswith('#'):
                tsr_line = tsr_line[1:].strip()
            tip_speed_ratios = np.array([float(x) for x in tsr_line.split()])
            cp_start_idx = None
            for i, line in enumerate(lines):
                if '# Power coefficient' in line:
                    cp_start_idx = i + 1
                    break
            if cp_start_idx is None:
                raise ValueError("Could not find '# Power coefficient' section in Cp file")
            while cp_start_idx < len(lines) and not lines[cp_start_idx].strip():
                cp_start_idx += 1
            cp_values = []
            for i in range(len(tip_speed_ratios)):
                if cp_start_idx + i >= len(lines):
                    break
                line = lines[cp_start_idx + i].strip()
                if not line:
                    continue
                cp_row = np.array([float(x) for x in line.split() if x.strip()])
                cp_values.append(cp_row)
            if len(cp_values) > 0:
                expected_length = len(pitch_angles)
                cp_values_fixed = []
                for row in cp_values:
                    if len(row) > expected_length:
                        cp_values_fixed.append(row[:expected_length])
                    elif len(row) < expected_length:
                        padded = np.zeros(expected_length)
                        padded[:len(row)] = row
                        cp_values_fixed.append(padded)
                    else:
                        cp_values_fixed.append(row)
                cp_values = np.array(cp_values_fixed)
            else:
                raise ValueError("No Cp data found in Cp file")
            self._cp_interp = RegularGridInterpolator(
                (pitch_angles, tip_speed_ratios),
                cp_values.T,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._cp_data = True
        if x is None:
            return 0.0

        X = self.local_view(x)
        # omega_m is stored in per-unit (base = omega_m_rated in rad/s)
        omega_m_rad_s = X['omega_m'] * par['omega_m_rated'] # pu speed * base speed -> rad/s
        wind_speed = self.wind_speed(x, v) # m/s
        # Handle array case: use np.where for element-wise conditional
        tip_speed_ratio = np.where(wind_speed > 0, omega_m_rad_s * par['R'] / wind_speed, 0)

        # Interpolate Cp value - pass as 1D array of length 2 for a single point
        # Convert to Python floats first to avoid any array issues
        tsr = float(tip_speed_ratio) if np.isscalar(tip_speed_ratio) else float(tip_speed_ratio.item())
        # Use state variable pitch_angle
        X = self.local_view(x)
        pitch_angle_val = X['pitch_angle'] 
        pa = float(pitch_angle_val*180/np.pi) if np.isscalar(pitch_angle_val) else float(pitch_angle_val.item()*180/np.pi)
        
        # Clamp values to be within grid bounds to avoid extrapolation returning fill_value (0)
        # Grid order is (pitch_angles, tip_speed_ratios), so grid[0] = pitch, grid[1] = TSR
        pa_clamped = np.clip(pa, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
        tsr_clamped = np.clip(tsr, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())
        
        # Point order must match grid order: (pitch, TSR)
        point = np.array([pa_clamped, tsr_clamped], dtype=np.float64)
        # RegularGridInterpolator returns an array, so we need to extract the scalar
        Cp_table = float(self._cp_interp(point)[0])
        
        # Cp is dimensionless - return the coefficient directly
        return Cp_table

    def wind_speed_init(self):
        """Wind speed at t=0 (m/s). Defaults to rated for load-flow / modal OP."""
        #return float(np.asarray(self.par['wind_rated']).ravel()[0])
        return 14.0

    def wind_speed(self, x, v):
        """Returns wind speed in m/s. Uses _sim_time (s) when set by sim loop, else 0 (init)."""
        t = getattr(self, '_sim_time', 0)
        if t < 120:
            return 14.0
        else:
            return 14.0
        """ t = getattr(self, '_sim_time', 0)
        return float(self._wind_interp(t)) """
