"""All matplotlib output.  Nothing else in ``src/`` writes to disk.

Unit conversion for display (Hz -> GHz, s -> ns/ps, V -> mV) happens here and
only here; every other module works in SI.

Split by domain rather than by figure, following StatOpt's ``display*`` grouping
(docs/statopt_review.md section 2).  When this outgrows one module the seam is
response plots / distribution plots / result plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analog import SplitPulse, eye_vs_phase  # noqa: E402
from .config import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"


def _save(fig, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Frequency-domain responses
# --------------------------------------------------------------------------

def plot_channel_response(mm: dict, cfg, name: str = "channel_response.png") -> Path:
    """Mixed-mode magnitudes vs. frequency, with the Nyquist point marked."""
    f_ghz = mm["f"] / 1e9

    def db(x):
        return 20 * np.log10(np.abs(x) + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_ghz, db(mm["sdd21"]), label="Sdd21 (insertion)", lw=1.4)
    ax.plot(f_ghz, db(mm["sdd11"]), label="Sdd11 (return)", lw=1.0, alpha=0.8)
    ax.plot(f_ghz, db(mm["scd21"]), label="Scd21 (mode conversion)", lw=1.0, alpha=0.8)

    fn = cfg.nyquist / 1e9
    il_nyq = db(np.interp(cfg.nyquist, mm["f"], np.abs(mm["sdd21"])))
    ax.axvline(fn, color="k", ls="--", lw=0.9)
    ax.plot([fn], [il_nyq], "ro", ms=5)
    ax.annotate(
        f"Nyquist {fn:.0f} GHz\n{il_nyq:.1f} dB",
        xy=(fn, il_nyq),
        xytext=(fn + 3, il_nyq + 10),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=9,
    )

    ax.set_xlim(0, 40)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Channel mixed-mode response")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Time-domain responses
# --------------------------------------------------------------------------

def plot_impulse_pulse(
    t: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    sp: SplitPulse,
    cfg,
    j: int | None = None,
    name: str = "channel_time.png",
) -> Path:
    """Impulse response, and pulse response with cursors at sampling phase ``j``."""
    j = sp.peak_phase if j is None else j
    m = cfg.samples_per_ui

    span = int(40 * m)
    lo = max(0, sp.k_peak - 10 * m)
    hi = min(t.size, lo + span)
    sl = slice(lo, hi)
    t_ns = t[sl] * 1e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    ax1.plot(t_ns, h[sl] * 1e3, lw=1.2)
    ax1.set_ylabel("h(t) (mV per sample)")
    ax1.set_title("Impulse response")
    ax1.grid(alpha=0.3)

    ax2.plot(t_ns, p[sl] * 1e3, lw=1.2, label="pulse response")
    idx = sp.k_peak + (j - sp.peak_phase) + np.arange(-sp.n_pre, sp.n_post + 1) * m
    keep = (idx >= lo) & (idx < hi)
    ax2.plot(
        t[idx[keep]] * 1e9, sp.at(j)[keep] * 1e3, "o", ms=4.5, color="crimson",
        label=f"cursors at phase {sp.phase_ui(j):+.3f} UI",
    )
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axvline(t[sp.k_peak] * 1e9, color="k", ls="--", lw=0.9, label="peak")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("p(t) (mV)")
    ax2.set_title(f"Pulse response, 1 UI = {cfg.ui * 1e12:.1f} ps")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return _save(fig, name)


def plot_cursor_stem(
    sp: SplitPulse, j: int | None = None, n_show: int = 20,
    name: str = "channel_cursors.png",
) -> Path:
    """Cursor magnitudes at phase ``j``, main cursor highlighted."""
    j = sp.peak_phase if j is None else j
    n_show = min(n_show, sp.n_post)
    c = sp.at(j)[: sp.n_pre + n_show + 1]
    idx = np.arange(-sp.n_pre, n_show + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(idx, c * 1e3, basefmt=" ")
    ax.plot([0], [c[sp.n_pre] * 1e3], "o", color="crimson", ms=7, label="main cursor")
    ax.set_xlabel("Cursor index (UI relative to main)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(
        f"Pulse response cursors at phase {sp.phase_ui(j):+.3f} UI "
        f"(showing {n_show} of {sp.n_post} postcursors)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, name)


def plot_eye_vs_phase(
    sp: SplitPulse,
    cfg,
    tap_counts: tuple[int, ...] = (0, 1, 2, 4),
    lock_phase_ui: float | None = None,
    name: str = "eye_vs_phase.png",
) -> Path:
    """Worst-case eye height across sampling phase, for several DFE tap counts.

    The peak of each curve is the best available sampling point; the MM lock
    phase is where the CDR actually settles.  Any gap between them is margin
    the design gives away.
    """
    phase = sp.phase_ui(np.arange(sp.samples_per_ui))
    order = np.argsort(phase)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for n in tap_counts:
        eye = eye_vs_phase(sp, n_taps=n) * 1e3
        ax.plot(phase[order], eye[order], lw=1.4,
                label=f"{n}-tap DFE" if n else "no DFE")

    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.9, label="pulse peak")
    if lock_phase_ui is not None:
        ax.axvline(lock_phase_ui, color="crimson", ls=":", lw=1.4,
                   label=f"MM lock ({lock_phase_ui:+.3f} UI)")

    ax.set_xlabel("Sampling phase relative to pulse peak (UI)")
    ax.set_ylabel("Worst-case eye height (mV)")
    ax.set_title(f"Peak-distortion eye vs. sampling phase, ISI depth {sp.n_post} UI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, name)
