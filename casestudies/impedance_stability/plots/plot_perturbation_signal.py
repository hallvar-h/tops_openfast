"""
Visualise the disturbance (perturbation) current injected for impedance identification.

Band-split **multitone** — sum of sinusoids at bin-aligned tones in the LF / MF / HF bands
(see ``ps_data.multitone_fft_bands`` and ``multitone_fft_bands.band_specs``). Each tone has
amplitude ``amp = TARGET_TOTAL_RMS * sqrt(2 / n_tones)`` and an independent random phase
(RandomState seed = ``RANDOM_SEED``).

The signals are reconstructed analytically from those configs so the figure always matches
what is actually injected at bus B2 in the time-domain simulations.

Run with no arguments to write the default ``perturbation_signal.png`` thesis figure, or
use the CLI flags to plot a single band / axis.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from casestudies.dyn_sim.plotting.thesis_plot_style import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_COUPLED,
    COLOR_WIND,
    THESIS_DPI,
    apply_thesis_td_style,
    plain_y_axis,
    style_time_axis,
    xlim_time,
    ylim_nice,
)
from casestudies.impedance_stability.multitone_fft_bands.band_specs import (  # noqa: E402
    BAND_ORDER,
    RANDOM_SEED,
    TARGET_TOTAL_RMS,
    BandId,
)
from casestudies.impedance_stability.paths import LOGS_ROOT  # noqa: E402
from casestudies.impedance_stability.ps_data.multitone_fft_bands import (  # noqa: E402
    get_band_injected_frequencies,
)

# Output
DEFAULT_OUT_DIR: Path = LOGS_ROOT / "perturbation_signal"
DEFAULT_OUT_PNG: Path = DEFAULT_OUT_DIR / "perturbation_signal.png"

# Default short windows for each multitone band (so cycles of the lowest tone are visible).
# LF f_min = 0.05 Hz → 20 s period; show 2 periods. MF/HF: a few seconds is enough.
_BAND_T_MAX_S: dict[str, float] = {
    "lf": 40.0,
    "mf": 6.0,
    "hf": 3.0,
}

# Time-step for the analytic reconstruction (much finer than dt_sim=0.01 s to draw a smooth curve).
_DT_PLOT_S: float = 1e-3

# Band-specific colours (re-use the muted thesis palette).
_BAND_COLOR: dict[str, str] = {
    "lf": COLOR_BASELINE,
    "mf": COLOR_COUPLED,
    "hf": COLOR_WIND,
}

_BAND_TITLE: dict[str, str] = {
    "lf": "Low band (0.05--1 Hz)",
    "mf": "Mid band (0.9--4 Hz)",
    "hf": "High band (3.5--10 Hz)",
}


@dataclass(frozen=True)
class MultitoneSignal:
    """Analytic multitone perturbation: ``i_pert(t) = sum_k A cos(2*pi*f_k t + phi_k)``."""

    band: BandId
    axis: str
    freqs_hz: np.ndarray
    amp_per_tone_pu: float
    phases_rad: np.ndarray
    target_total_rms_pu: float

    @property
    def n_tones(self) -> int:
        return int(self.freqs_hz.size)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        phase = 2.0 * np.pi * self.freqs_hz[:, None] * t[None, :] + self.phases_rad[:, None]
        return float(self.amp_per_tone_pu) * np.cos(phase).sum(axis=0)


def build_multitone_signal(
    band: str,
    *,
    axis: str = "re",
    target_total_rms: float = TARGET_TOTAL_RMS,
    random_seed: int = RANDOM_SEED,
) -> MultitoneSignal:
    """Reconstruct the multitone injection used by ``load_multitone_fft_band``."""
    freqs = get_band_injected_frequencies(band)
    n_t = int(freqs.size)
    if n_t == 0:
        raise ValueError(f"No tones defined for band {band!r}")
    amp_per_tone = float(target_total_rms) * float(np.sqrt(2.0 / n_t))
    phases = (2.0 * np.pi) * np.random.RandomState(int(random_seed)).rand(n_t)
    return MultitoneSignal(
        band=band,  # type: ignore[arg-type]
        axis=str(axis).strip().lower(),
        freqs_hz=np.asarray(freqs, dtype=float),
        amp_per_tone_pu=amp_per_tone,
        phases_rad=np.asarray(phases, dtype=float),
        target_total_rms_pu=float(target_total_rms),
    )


def _draw_multitone(ax: plt.Axes, sig: MultitoneSignal, *, t_max: float) -> None:
    t = np.arange(0.0, float(t_max) + _DT_PLOT_S, _DT_PLOT_S)
    y = sig(t)
    color = _BAND_COLOR.get(sig.band, COLOR_BASELINE)
    ax.plot(t, y, color=color, linewidth=1.0)
    ax.axhline(0.0, color="0.6", linewidth=0.45, linestyle=":")
    label = (
        rf"$N={sig.n_tones}$, $A_k={sig.amp_per_tone_pu:.4f}$ pu, "
        rf"$I_{{\mathrm{{RMS}}}}={sig.target_total_rms_pu:.3f}$ pu"
    )
    ax.text(
        0.985,
        0.92,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="0.25",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="0.8",
            lw=0.4,
            alpha=0.92,
        ),
    )
    style_time_axis(ax)
    plain_y_axis(ax)
    xlim_time(ax, t)
    ylim_nice(ax, y, pad_frac=0.15)


def plot_multitone(
    band: str = "mf",
    *,
    axis: str = "re",
    t_max: float | None = None,
    out_png: str | Path | None = None,
) -> Path:
    """Single-band multitone perturbation figure."""
    apply_thesis_td_style()
    sig = build_multitone_signal(band, axis=axis)
    t_max = float(t_max if t_max is not None else _BAND_T_MAX_S[band])

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.6))
    _draw_multitone(ax, sig, t_max=t_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(rf"$i_{{\mathrm{{pert}},{sig.axis}}}$ (p.u.)")
    ax.set_title(_BAND_TITLE.get(band, band), fontsize=10, pad=6)

    out = Path(out_png) if out_png is not None else DEFAULT_OUT_DIR / f"multitone_{band}_{axis}.png"
    return _save(fig, out)


def plot_overview(
    *,
    axis: str = "re",
    out_png: str | Path | None = None,
) -> Path:
    """Stacked overview: LF, MF, and HF multitone bands."""
    apply_thesis_td_style()
    bands = list(BAND_ORDER)
    height_per_panel = 1.55
    fig, axes = plt.subplots(
        len(bands),
        1,
        figsize=(7.0, 0.5 + height_per_panel * len(bands)),
        sharey=False,
    )
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        sig = build_multitone_signal(band, axis=axis)
        _draw_multitone(ax, sig, t_max=_BAND_T_MAX_S[band])
        ax.set_ylabel(rf"$i_{{\mathrm{{pert}},{sig.axis}}}$ (p.u.)")
        ax.set_title(_BAND_TITLE[band], fontsize=9.5, pad=4, loc="left")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Impedance-ID perturbation current (axis = "
        + ("real / d" if axis == "re" else "imag / q")
        + ")",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(out_png) if out_png is not None else DEFAULT_OUT_PNG
    return _save(fig, out)


def _save(fig: plt.Figure, out_png: Path) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot the impedance-ID multitone perturbation current.",
    )
    ap.add_argument(
        "--kind",
        choices=("overview", "multitone"),
        default="overview",
        help="Figure kind. 'overview' = LF+MF+HF multitone.",
    )
    ap.add_argument(
        "--band",
        choices=BAND_ORDER,
        default=None,
        help="Multitone band (lf/mf/hf). Default mf for --kind multitone.",
    )
    ap.add_argument("--axis", choices=("re", "im"), default="re", help="Injection axis.")
    ap.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Window length (s). Default depends on band.",
    )
    ap.add_argument("--out", default=None, help="Output PNG path.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    if args.kind == "overview":
        out = plot_overview(axis=args.axis, out_png=args.out)
    else:
        band = args.band or "mf"
        out = plot_multitone(band=band, axis=args.axis, t_max=args.t_max, out_png=args.out)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
