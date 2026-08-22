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

from .channel import SplitPulse, eye_vs_phase  # noqa: E402
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


def plot_verification(
    data: dict, name: str = "verify_channel.png"
) -> Path:
    """Four-panel physical-validity check of the channel conversion.

    Panels: (a) causality on a log magnitude axis, (b) cursor sum converging to
    the DC gain, (c) passivity vs. frequency, (d) cursor-sum error across
    sampling phase.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # -- (a) causality ----------------------------------------------------
    t_ns, h = data["t"] * 1e9, data["h"]
    peak = np.abs(h).max()
    db = 20 * np.log10(np.abs(h) / peak + 1e-18)
    k_on, k_pk = data["k_onset"], data["k_peak"]
    ax_a.plot(t_ns, db, lw=0.8)
    ax_a.axvline(t_ns[k_on], color="crimson", ls=":", lw=1.4,
                 label=f"onset {t_ns[k_on]:.3f} ns")
    ax_a.axvline(t_ns[k_pk], color="k", ls="--", lw=1.0,
                 label=f"peak {t_ns[k_pk]:.3f} ns")
    ax_a.axhline(-120, color="grey", lw=0.6)
    ax_a.set_xlim(0, min(t_ns[-1], t_ns[k_pk] * 2.5))
    ax_a.set_ylim(-200, 5)
    ax_a.set_xlabel("Time (ns)")
    ax_a.set_ylabel("|h(t)| relative to peak (dB)")
    ax_a.set_title(
        f"(a) Causality — pre-onset energy {data['causality']:.2e}"
    )
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8, loc="upper right")

    # -- (b) cursor sum -> DC gain ---------------------------------------
    dc = data["dc_gain"]
    for j, lbl in data["cumulative"]:
        m = np.arange(j.size) - data["n_pre"]
        ax_b.plot(m, j / dc, lw=1.2, label=lbl)
    ax_b.axhline(1.0, color="k", ls="--", lw=1.0, label="H(0) (DC gain)")
    ax_b.axvline(data["isi_depth"], color="crimson", ls=":", lw=1.2,
                 label=f"cfg.isi_depth = {data['isi_depth']}")
    ax_b.set_xscale("symlog", linthresh=10)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_xlabel("Cursor index m (symlog)")
    ax_b.set_ylabel("Cumulative sum / H(0)")
    ax_b.set_title("(b) Cursor sum converges to the low-frequency gain")
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8, loc="lower right")

    # -- (c) passivity ----------------------------------------------------
    f_ghz = data["f_s"] / 1e9
    ax_c.plot(f_ghz, data["max_abs_s"], lw=1.2, label=r"max$_{ij}$ |S$_{ij}$|")
    ax_c.plot(f_ghz, data["sigma_max"], lw=1.2, label=r"$\sigma_{max}$(S)")
    ax_c.axhline(1.0, color="crimson", ls="--", lw=1.2, label="passivity limit")
    ax_c.set_xlabel("Frequency (GHz)")
    ax_c.set_ylabel("Magnitude")
    ax_c.set_ylim(0, 1.15)
    ax_c.set_title(
        f"(c) Passivity — max σ = {data['sigma_max'].max():.4f}"
    )
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8, loc="lower right")

    # -- (d) phase independence of the cursor sum -------------------------
    ph, err = data["phase_ui"], data["sum_err_vs_phase"]
    order = np.argsort(ph)
    ax_d.semilogy(ph[order], np.abs(err[order]) + 1e-18, lw=1.4)
    ax_d.axhline(data["sum_tol"], color="crimson", ls="--", lw=1.2,
                 label=f"tolerance {data['sum_tol']:.0e}")
    ax_d.set_xlabel("Sampling phase relative to peak (UI)")
    ax_d.set_ylabel("|Σ cursors − H(0)| / H(0)")
    ax_d.set_title("(d) Identity holds at every sampling phase")
    ax_d.grid(alpha=0.3, which="both")
    ax_d.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, name)


def plot_ctle_verification(data: dict, name: str = "verify_ctle.png") -> Path:
    """Measured discrete-time CTLE response against the analog H(s) reference.

    Panels: (a) Bode magnitude with the analog reference and each discretized
    mode, (b) phase, (c) magnitude deviation from theory, (d) zoom on the peak
    showing where each mode actually lands.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    f_ghz = data["f"] / 1e9
    ref_db = data["analog_db"]
    target_peak = data["target_peak"] / 1e9
    nyq = data["nyquist"] / 1e9
    colors = ["#d62728", "#2ca02c", "#9467bd"]

    # -- (a) magnitude ----------------------------------------------------
    ax_a.semilogx(f_ghz, ref_db, "k-", lw=2.4, alpha=0.75, label="analog H(s)")
    for (tag, m), col in zip(data["modes"].items(), colors):
        ax_a.semilogx(f_ghz, m["mag_db"], lw=1.1, color=col, ls="--", label=f"measured: {tag}")
    ax_a.axvline(target_peak, color="grey", ls=":", lw=1.2)
    ax_a.axvline(nyq, color="k", ls="-.", lw=0.9, label=f"Nyquist {nyq:.0f} GHz")
    ax_a.axvline(data["fs"] / 2e9, color="crimson", ls="-.", lw=0.9,
                 label=f"fs/2 = {data['fs'] / 2e9:.0f} GHz")
    ax_a.set_xlim(0.05, f_ghz[-1])
    # The bilinear transform puts a zero at fs/2, so every discretized curve
    # dives to -inf there.  Clamp the axis to the band the link occupies rather
    # than let that artifact set the scale.
    ax_a.set_ylim(-40, 20)
    ax_a.set_ylabel("Magnitude (dB)")
    ax_a.set_title("(a) Bode magnitude — theory vs. measured")
    ax_a.grid(alpha=0.3, which="both")
    ax_a.legend(fontsize=8, loc="lower left")

    # -- (b) phase --------------------------------------------------------
    ax_b.semilogx(f_ghz, data["analog_phase"], "k-", lw=2.4, alpha=0.75,
                  label="analog H(s)")
    for (tag, m), col in zip(data["modes"].items(), colors):
        ax_b.semilogx(f_ghz, m["phase_deg"], lw=1.1, color=col, ls="--",
                      label=f"measured: {tag}")
    ax_b.axvline(nyq, color="k", ls="-.", lw=0.9)
    ax_b.set_xlim(0.05, f_ghz[-1])
    ax_b.set_ylabel("Phase (deg)")
    ax_b.set_title("(b) Bode phase")
    ax_b.grid(alpha=0.3, which="both")
    ax_b.legend(fontsize=8, loc="lower left")

    # -- (c) deviation ----------------------------------------------------
    # |deviation| on a log axis: the dynamic range runs from ~1e-14 dB in band
    # to hundreds of dB at fs/2, which no linear axis can show at once.
    for (tag, m), col in zip(data["modes"].items(), colors):
        ax_c.loglog(f_ghz, np.abs(m["mag_db"] - ref_db) + 1e-16, lw=1.3,
                    color=col, label=tag)
    ax_c.axvline(nyq, color="k", ls="-.", lw=0.9, label=f"Nyquist {nyq:.0f} GHz")
    ax_c.axvline(data["fs"] / 2e9, color="crimson", ls="-.", lw=0.9, label="fs/2")
    ax_c.set_xlim(0.05, f_ghz[-1])
    ax_c.set_ylim(1e-6, 1e3)
    ax_c.set_xlabel("Frequency (GHz)")
    ax_c.set_ylabel("|measured − theory| (dB)")
    ax_c.set_title("(c) Deviation from the analog reference")
    ax_c.grid(alpha=0.3, which="both")
    ax_c.legend(fontsize=8, loc="upper left")

    # -- (d) peak zoom ----------------------------------------------------
    lo, hi = 0.85 * target_peak, 1.15 * target_peak
    sel = (f_ghz >= lo) & (f_ghz <= hi)
    ax_d.plot(f_ghz[sel], ref_db[sel], "k-", lw=2.4, alpha=0.75, label="analog H(s)")
    for (tag, m), col in zip(data["modes"].items(), colors):
        ax_d.plot(f_ghz[sel], m["mag_db"][sel], lw=1.2, color=col, ls="--")
        ax_d.plot([m["peak_hz"] / 1e9], [m["boost_db"]], "o", ms=7, color=col,
                  label=f"{tag}: {m['peak_hz'] / 1e9:.4f} GHz "
                        f"({m['peak_err'] * 100:+.3f}%)")
    ax_d.axvline(target_peak, color="grey", ls=":", lw=1.6,
                 label=f"target {target_peak:.3f} GHz")
    ax_d.set_xlim(lo, hi)
    ax_d.set_xlabel("Frequency (GHz)")
    ax_d.set_ylabel("Magnitude (dB)")
    ax_d.set_title("(d) Where the peak actually lands")
    ax_d.grid(alpha=0.3)
    ax_d.legend(fontsize=8, loc="lower center")

    fig.tight_layout()
    return _save(fig, name)


def plot_cdr_verification(data: dict, name: str = "verify_cdr.png") -> Path:
    """CDR acquisition, jitter transfer, and jitter tolerance.

    Panels: (a) phase trajectory acquiring a frequency offset, (b) residual
    tracking error, (c) measured jitter transfer against the second-order
    model, (d) jitter tolerance with its linear and slew-rate asymptotes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # -- (a) acquisition ---------------------------------------------------
    acq = data["acquisition"]
    n_us = acq["n"] / data["symbol_rate"] * 1e6
    ax_a.plot(n_us, acq["phase"], lw=1.2, label="recovered clock phase")
    ax_a.plot(n_us, -acq["ideal"], lw=1.4, ls="--", color="crimson",
              label=f"{data['ppm']:.0f} ppm ramp (target)")
    ax_a.axvline(acq["lock_time_us"], color="k", ls=":", lw=1.4,
                 label=f"lock at {acq['lock_time_us']:.2f} us")
    ax_a.set_xlabel("Time (us)")
    ax_a.set_ylabel("Phase (UI)")
    ax_a.set_title(f"(a) Acquisition of a {data['ppm']:.0f} ppm offset")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8, loc="best")

    # -- (b) tracking error ------------------------------------------------
    ax_b.plot(n_us, acq["error"] * 1e3, lw=0.9)
    tol = acq["tol_ui"] * 1e3
    ax_b.axhspan(-tol, tol, color="green", alpha=0.12,
                 label=f"+/-{tol:.0f} mUI lock band")
    ax_b.axvline(acq["lock_time_us"], color="k", ls=":", lw=1.4)
    ax_b.set_xlabel("Time (us)")
    ax_b.set_ylabel("Tracking error (mUI)")
    ax_b.set_title("(b) Residual sampling error — type-II drives it to zero")
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8)

    # -- (c) jitter transfer -----------------------------------------------
    jt = data["jitter_transfer"]
    ax_c.semilogx(jt["f_theory"] / 1e6, jt["h_theory_db"], "k-", lw=2.2,
                  alpha=0.75, label="2nd-order model |H|")
    ax_c.semilogx(jt["f_theory"] / 1e6, jt["e_theory_db"], color="grey", lw=1.4,
                  ls="-.", label="model |E| = |1-H|")
    ax_c.semilogx(jt["f"] / 1e6, jt["h_db"], "o", ms=6, color="crimson",
                  label="measured |H|")
    ax_c.semilogx(jt["f"] / 1e6, jt["e_db"], "s", ms=5, color="#1f77b4",
                  alpha=0.8, label="measured |E|")
    ax_c.axhline(-3, color="k", ls=":", lw=1.0)
    ax_c.axvline(jt["f3db_meas"] / 1e6, color="crimson", ls="--", lw=1.3,
                 label=f"measured f_3dB {jt['f3db_meas'] / 1e6:.2f} MHz")
    ax_c.axvline(jt["f3db_theory"] / 1e6, color="k", ls="--", lw=1.1,
                 label=f"theory f_3dB {jt['f3db_theory'] / 1e6:.2f} MHz")
    ax_c.set_ylim(-30, 8)
    ax_c.set_xlabel("Jitter frequency (MHz)")
    ax_c.set_ylabel("Magnitude (dB)")
    ax_c.set_title("(c) Jitter transfer vs. the linearized model")
    ax_c.grid(alpha=0.3, which="both")
    ax_c.legend(fontsize=7.5, loc="lower left")

    # -- (d) jitter tolerance ----------------------------------------------
    jl = data["jtol"]
    ax_d.loglog(jl["f_theory"] / 1e6, jl["linear"], "k-", lw=2.0, alpha=0.75,
                label=r"linear: $\phi_{margin}/|E(f)|$")
    ax_d.loglog(jl["f_theory"] / 1e6, jl["slew"], color="darkorange", lw=2.0,
                ls="--", label=r"slew limit: $K_p/(2\pi f T_u)$")
    ax_d.loglog(jl["f_theory"] / 1e6, jl["combined"], color="grey", lw=3.0,
                alpha=0.35, label="min of the two")
    ax_d.loglog(jl["f"] / 1e6, jl["measured"], "o", ms=7, color="crimson",
                label="measured (bisection)")
    ax_d.set_xlabel("Jitter frequency (MHz)")
    ax_d.set_ylabel("Tolerable jitter (UI pp)")
    ax_d.set_title("(d) Jitter tolerance — slew limit sets the low-frequency slope")
    ax_d.grid(alpha=0.3, which="both")
    ax_d.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    return _save(fig, name)


def _draw_eye(ax, eye, title: str, max_traces: int = 1500) -> None:
    step = max(1, eye.traces.shape[0] // max_traces)
    t = eye.phase_ui
    ax.plot(t, eye.traces[::step].T * 1e3, color="#1f77b4", lw=0.35, alpha=0.06)
    ax.plot(t, eye.upper_min * 1e3, color="crimson", lw=1.6, label="worst 'one'")
    ax.plot(t, eye.lower_max * 1e3, color="darkorange", lw=1.6, label="worst 'zero'")
    ax.axhline(0, color="k", lw=0.7)
    h, ph = eye.eye_height()
    if h > 0:
        ax.annotate("", xy=(ph, eye.lower_max[np.argmin(np.abs(t - ph))] * 1e3),
                    xytext=(ph, eye.upper_min[np.argmin(np.abs(t - ph))] * 1e3),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1.4))
        ax.text(ph + 0.02, 0, f"  {h * 1e3:.1f} mV", fontsize=9, va="center")
    ax.set_xlabel("Phase (UI)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def plot_eye_analysis(data: dict, name: str = "eye_analysis.png") -> Path:
    """Six-panel eye analysis: time-domain eyes, BER contour, bathtubs, Q-fit."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    _draw_eye(ax_a, data["eye_off"], "(a) Time-domain eye — no EQ")
    _draw_eye(ax_b, data["eye_on"], "(b) Time-domain eye — CTLE + DFE")

    # -- (c) statistical BER contour --------------------------------------
    se = data["stat_eye"]
    floor = 1e-16
    logber = np.log10(np.maximum(se.ber.T, floor))
    im = ax_c.pcolormesh(se.phase_ui, se.threshold_v * 1e3, logber,
                         cmap="turbo", shading="auto", vmin=-16, vmax=0)
    levels = [-12, -9, -6, -3]
    ax_c.contour(se.phase_ui, se.threshold_v * 1e3, logber, levels=levels,
                 colors="white", linewidths=0.9)
    fig.colorbar(im, ax=ax_c, label="log10 BER")
    ax_c.set_xlabel("Phase (UI)")
    ax_c.set_ylabel("Threshold (mV)")
    ax_c.set_title("(c) Statistical eye — BER contour (white: 1e-12/-9/-6/-3)")

    # -- (d) vertical bathtub ---------------------------------------------
    j, vb = se.vertical_bathtub()
    ax_d.semilogy(se.threshold_v * 1e3, np.maximum(vb, floor), lw=1.6)
    ax_d.axhline(1e-12, color="crimson", ls="--", lw=1.2, label="BER 1e-12")
    eh = data["eye_height_1e12"]
    if eh > 0:
        ok = np.nonzero(vb < 1e-12)[0]
        ax_d.axvspan(se.threshold_v[ok[0]] * 1e3, se.threshold_v[ok[-1]] * 1e3,
                     color="green", alpha=0.13, label=f"EH = {eh * 1e3:.1f} mV")
    ax_d.set_ylim(1e-16, 1)
    ax_d.set_xlabel("Threshold (mV)")
    ax_d.set_ylabel("BER")
    ax_d.set_title(f"(d) Vertical bathtub at {se.phase_ui[j]:+.3f} UI")
    ax_d.grid(alpha=0.3, which="both")
    ax_d.legend(fontsize=8)

    # -- (e) horizontal bathtub -------------------------------------------
    hb = data["horizontal"]
    ax_e.semilogy(se.phase_ui, np.maximum(hb, floor), lw=1.6, label="statistical")
    if data.get("td_bathtub") is not None:
        tp, tb = data["td_bathtub"]
        ax_e.semilogy(tp, np.maximum(tb, floor), "o", ms=5, color="crimson",
                      label=f"time-domain ({data['n_td']:,} symbols)")
    ax_e.axhline(1e-12, color="crimson", ls="--", lw=1.2)
    ax_e.axhline(data["td_floor"], color="grey", ls=":", lw=1.2,
                 label=f"time-domain floor ~{data['td_floor']:.0e}")
    ew = data["eye_width_1e12"]
    if ew > 0:
        ok = np.nonzero(hb < 1e-12)[0]
        ax_e.axvspan(se.phase_ui[ok[0]], se.phase_ui[ok[-1]], color="green",
                     alpha=0.13, label=f"EW = {ew:.4f} UI")
    ax_e.set_ylim(1e-16, 1)
    ax_e.set_xlabel("Phase (UI)")
    ax_e.set_ylabel("BER")
    ax_e.set_title("(e) Horizontal bathtub — the overlap is the cross-check")
    ax_e.grid(alpha=0.3, which="both")
    ax_e.legend(fontsize=8, loc="lower center")

    # -- (f) Q-scale fit ---------------------------------------------------
    fit = data["fit"]
    q = data["q_curve"]
    ok = np.isfinite(q)
    ax_f.plot(se.phase_ui[ok], q[ok], "o", ms=4, color="#1f77b4",
              label="bathtub on the Q scale")
    tl = np.linspace(fit.mu_left, se.phase_ui[np.argmin(hb)], 50)
    tr = np.linspace(se.phase_ui[np.argmin(hb)], fit.mu_right, 50)
    ax_f.plot(tl, (tl - fit.mu_left) / fit.sigma_left, "k--", lw=1.6,
              label="dual-Dirac fit")
    ax_f.plot(tr, (fit.mu_right - tr) / fit.sigma_right, "k--", lw=1.6)
    ax_f.axhspan(data["q_lo"], data["q_hi"], color="green", alpha=0.10,
                 label=f"fit window Q={data['q_lo']:.0f}..{data['q_hi']:.0f}")
    for mu, lbl in ((fit.mu_left, r"$\mu_L$"), (fit.mu_right, r"$\mu_R$")):
        ax_f.axvline(mu, color="crimson", ls=":", lw=1.2)
        ax_f.text(mu, 0.4, lbl, color="crimson", fontsize=10, ha="center")
    ax_f.axhline(0, color="k", lw=0.7)
    ax_f.set_ylim(-0.5, 9)
    ax_f.set_xlabel("Phase (UI)")
    ax_f.set_ylabel("Q")
    ax_f.set_title(f"(f) Jitter decomposition — RJ {fit.rj_rms_ui * 1e3:.2f} mUI, "
                   f"DJ {fit.dj_pp_ui * 1e3:.1f} mUI")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper center")

    fig.tight_layout()
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
