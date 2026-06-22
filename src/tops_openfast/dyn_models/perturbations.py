import numpy as np
from tops.dyn_models.utils import DAEModel


class BusCurrentPerturbation(DAEModel):
    """
    Minimal current injection model for frequency-response / impedance experiments.

    Injects a complex current in system p.u. at a specified bus. The perturbation is
    generated internally using a phase state

      theta_dot = 2*pi*f
      I_inj = (I_re + j*I_im) + amp * sin(theta + phase) * axis

    One unit row per tone (multitone) or a single row (singletone).

    Axis options:
      - "re": inject perturbation on real axis only
      - "im": inject perturbation on imag axis only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus_idx = np.array(
            np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )
        self.bus_idx_red = np.array(
            np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )

    def bus_ref_spec(self):
        return {"terminal": self.par["bus"]}

    def reduced_system(self):
        return self.par["bus"]

    def state_list(self):
        return ["theta"]

    def init_from_load_flow(self, x_0, v_0, S):
        # Keep oscillator deterministic across runs
        X0 = self.local_view(x_0)
        X0["theta"][:] = 0.0

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        p = self.par
        f = np.asarray(p["f"], dtype=float)
        dX["theta"][:] = 2.0 * np.pi * f

    def current_injections(self, x, v):
        p = self.par
        X = self.local_view(x)

        i_base = np.asarray(p["I_re"], dtype=float) + 1j * np.asarray(p["I_im"], dtype=float)
        amp = np.asarray(p["amp"], dtype=float)
        phase = np.asarray(p["phase"], dtype=float)

        # Axis selection: 're' (default) or 'im'
        axis = p["axis"] if "axis" in p.dtype.names else np.array(["re"] * self.n_units)
        axis = np.asarray(axis).astype(str)
        use_imag = axis == "im"

        # perturbation current: magnitude * sin(phase + theta)
        # phase is the phase of the perturbation given the frequency and the time
        # theta is the phase of the system
        i_pert = amp * np.sin(np.asarray(X["theta"], dtype=float) + phase)
        # re-only: 1, im-only: 1j
        if np.any((axis != "re") & (axis != "im")):
            bad = sorted(set(axis[(axis != "re") & (axis != "im")].tolist()))
            raise ValueError(f"BusCurrentPerturbation axis must be 're' or 'im'. Got {bad}")
        inj_factor = np.where(use_imag, 1j, 1.0)
        i_total = i_base + i_pert * inj_factor

        return self.bus_idx_red["terminal"], i_total
