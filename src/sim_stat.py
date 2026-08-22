"""Statistical eye: ISI distribution, BER contour, bathtubs, jitter fitting.

Computes the eye by propagating *distributions* rather than waveforms, which is
what makes BER 1e-12 reachable at all: a time-domain run of N symbols resolves
no lower than about 100/N, so 1e-12 would need ~1e14 symbols.

What this engine assumes, and therefore where it should not be believed:

- **Independent cursors.**  Each interfering symbol is an independent +/-1.
  Exact for a linear channel with random data; false once a DFE feeds decisions
  back (error propagation is not modelled at all) or a CDR makes the sampling
  phase data-dependent.
- **Steady state.**  Converged taps, fixed phase.  No acquisition, no bursts.
- **Modelled jitter.**  RJ and DJ are imposed as a distribution, not observed.

Against those it gains what time-domain cannot have: *every* ISI pattern,
weighted by probability, rather than the few a PRBS happened to emit.

All quantities are SI; phase is in UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from .config import LinkConfig

#: Q-factor for BER 1e-12, i.e. sqrt(2)*erfcinv(2e-12).
Q_1E12 = float(np.sqrt(2.0) * special.erfcinv(2.0 * 1e-12))


def q_of_ber(ber: np.ndarray | float) -> np.ndarray:
    """Q-factor corresponding to a BER, ``Q = sqrt(2)*erfcinv(2*BER)``."""
    return np.sqrt(2.0) * special.erfcinv(2.0 * np.asarray(ber, dtype=float))


def plateau_centre(y: np.ndarray) -> int:
    """Index of the centre of the minimum of ``y``, robust to a flat floor.

    In a well-opened eye the BER underflows to exactly 0 over a wide band of
    phases, so ``argmin`` returns the *first* index of that plateau -- which is
    the edge of the swept range, not the middle of the eye.  Every downstream
    quantity (best phase, bathtub, jitter fit) then anchors to the wrong place.
    Taking the middle of the tied region gives the eye centre instead.
    """
    y = np.asarray(y, dtype=float)
    tied = np.flatnonzero(y <= np.nanmin(y) * (1.0 + 1e-9) + 1e-300)
    return int(tied[tied.size // 2]) if tied.size else int(np.argmin(y))


def cursors_at_phase(
    p: np.ndarray,
    cfg: LinkConfig,
    k_peak: int,
    phase_ui: float,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> np.ndarray:
    """Cursor vector at an arbitrary sub-UI sampling phase.

        h_m = p(k_peak + phase*M + m*M),   m = -n_pre .. +n_post

    Linear interpolation between samples, so the phase grid is not limited to
    the ``samples_per_ui`` lattice -- needed because a bathtub curve has to be
    resolved much more finely than 1/32 UI.
    """
    n_pre = cfg.n_pre if n_pre is None else n_pre
    n_post = cfg.isi_depth if n_post is None else n_post
    m = cfg.samples_per_ui

    idx = k_peak + phase_ui * m + np.arange(-n_pre, n_post + 1) * m
    lo = np.floor(idx).astype(int)
    frac = idx - lo
    out = np.zeros(idx.size)
    ok = (lo >= 0) & (lo + 1 < p.size)
    out[ok] = p[lo[ok]] * (1.0 - frac[ok]) + p[lo[ok] + 1] * frac[ok]
    return out


def isi_pdf(
    interferers: np.ndarray, n_grid: int = 8192, span: float = 1.3
) -> tuple[np.ndarray, np.ndarray]:
    """Probability density of the total ISI voltage.

    The ISI is ``sum_m a_m*h_m`` with ``a_m`` independent and equiprobable in
    {-1,+1}.  Each term has characteristic function ``cos(w*h_m)``, and the
    characteristic function of a sum is the product, so

        Phi(w) = prod_m cos(w*h_m)

    and the density is its inverse transform.  This is exact and costs one FFT,
    whereas enumerating the 2^200 sign combinations is impossible and
    convolving 200 two-point kernels one at a time accumulates a rounding error
    per cursor.

    The support is bounded by ``sum |h_m|``; the grid spans ``span`` times that,
    so the transform does not alias.

    Returns
    -------
    v : np.ndarray
        Voltage grid, centred on zero.
    pdf : np.ndarray
        Density in 1/V, integrating to 1.
    """
    interferers = np.asarray(interferers, dtype=float)
    reach = float(np.abs(interferers).sum())
    if reach == 0.0:
        v = np.linspace(-1e-9, 1e-9, n_grid)
        pdf = np.zeros(n_grid)
        pdf[n_grid // 2] = 1.0 / (v[1] - v[0])
        return v, pdf

    half = span * reach
    dv = 2.0 * half / n_grid
    w = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=dv)

    phi = np.prod(np.cos(np.outer(w, interferers)), axis=1)
    pdf = np.fft.fftshift(np.fft.irfft(phi, n=n_grid)) / dv
    np.maximum(pdf, 0.0, out=pdf)          # clip FFT ringing at the 1e-16 level
    pdf /= pdf.sum() * dv

    v = (np.arange(n_grid) - n_grid // 2) * dv
    return v, pdf


def ber_at_thresholds(
    main: float,
    v_isi: np.ndarray,
    pdf: np.ndarray,
    thresholds: np.ndarray,
    sigma_n: float,
) -> np.ndarray:
    """BER as a function of slicer threshold, at one sampling phase.

    Conditioned on the transmitted symbol,

        P(err | +1) = P(main + isi + noise < V)
        P(err | -1) = P(-main + isi + noise > V)
        BER(V)      = 0.5*(P(err|+1) + P(err|-1))

    The Gaussian noise is integrated **analytically** against the ISI density
    rather than convolved numerically.  That matters: the 1e-12 contour lives
    entirely in the Gaussian tail, and an FFT convolution's own noise floor
    (~1e-16 relative) would swamp it.  The ISI density has bounded support, so
    it needs no tail accuracy of its own.
    """
    dv = float(v_isi[1] - v_isi[0])
    weight = pdf * dv
    keep = weight > 0.0
    isi, wgt = v_isi[keep], weight[keep]

    v = np.asarray(thresholds, dtype=float)[:, None]
    if sigma_n <= 0:
        p_hi = (main + isi[None, :] < v).astype(float)
        p_lo = (-main + isi[None, :] > v).astype(float)
    else:
        p_hi = stats.norm.cdf((v - main - isi[None, :]) / sigma_n)
        p_lo = stats.norm.sf((v + main - isi[None, :]) / sigma_n)
    return 0.5 * (p_hi @ wgt + p_lo @ wgt)


@dataclass
class StatEye:
    """Statistical eye: BER over sampling phase and slicer threshold."""

    phase_ui: np.ndarray            # (P,)
    threshold_v: np.ndarray         # (T,)
    ber: np.ndarray                 # (P, T)
    main: np.ndarray                # (P,) main cursor per phase
    sigma_n: float
    sigma_rj_ui: float
    dj_pp_ui: float

    def vertical_bathtub(self, j: int | None = None) -> tuple[int, np.ndarray]:
        """BER vs. threshold at the phase with the widest opening."""
        j = plateau_centre(self.ber.min(axis=1)) if j is None else j
        return j, self.ber[j]

    def horizontal_bathtub(self, v: float = 0.0) -> np.ndarray:
        """BER vs. sampling phase at a fixed threshold."""
        i = int(np.argmin(np.abs(self.threshold_v - v)))
        return self.ber[:, i]

    def eye_height(self, target: float = 1e-12, j: int | None = None) -> float:
        """Threshold span with BER below ``target``, at the best phase, in V."""
        j, curve = self.vertical_bathtub(j)
        ok = np.nonzero(curve < target)[0]
        if ok.size < 2:
            return 0.0
        return float(self.threshold_v[ok[-1]] - self.threshold_v[ok[0]])

    def eye_width(self, target: float = 1e-12, v: float = 0.0) -> float:
        """Phase span with BER below ``target``, at threshold ``v``, in UI."""
        curve = self.horizontal_bathtub(v)
        ok = np.nonzero(curve < target)[0]
        if ok.size < 2:
            return 0.0
        return float(self.phase_ui[ok[-1]] - self.phase_ui[ok[0]])


def build_stat_eye(
    p: np.ndarray,
    cfg: LinkConfig,
    k_peak: int,
    sigma_n: float,
    sigma_rj_ui: float = 0.0,
    dj_pp_ui: float = 0.0,
    n_phase: int = 129,
    n_threshold: int = 257,
    phase_span: float = 1.4,
    n_taps: int = 0,
    tap_phase_ui: float = 0.0,
) -> StatEye:
    """Build the statistical eye from a pulse response.

    Steps, following the standard statistical-eye pipeline:

        1. cursors at each sampling phase
        2. ISI density per phase, by characteristic function
        3. BER vs. threshold per phase, integrating Gaussian noise analytically
        4. convolve along the phase axis with the jitter density

    Step 4 is a *horizontal* convolution and step 3's noise a *vertical* one --
    the two impairments enter on different axes and must not be conflated.

    The DFE is modelled with **fixed** taps, solved once at ``tap_phase_ui`` and
    then subtracted at every phase as a residual ``h_m - c_m``.  Re-solving the
    taps at each phase would model a DFE that re-adapts as the sampling point
    moves, which no real receiver does -- it has one tap set and a CDR that
    picks one phase.  The difference is not cosmetic: re-solving makes the
    statistical bathtub far too optimistic away from the tap phase, and then
    disagrees with a time-domain run by many orders of magnitude for a reason
    that has nothing to do with either method.

    Correct decisions are assumed, so error propagation is outside this model.

    Parameters
    ----------
    sigma_n : float
        RMS voltage noise at the slicer, in V.  With no noise the ISI
        distribution is discrete and BER jumps from 0 to O(0.1), so a 1e-12
        contour would not exist -- a noiseless statistical eye cannot answer
        criterion 2.
    sigma_rj_ui, dj_pp_ui : float
        Random jitter (rms) and deterministic jitter (peak-to-peak), in UI.
        DJ is modelled dual-Dirac: two equal masses at +/- dj/2.
    """
    # The sweep spans more than one UI so that *both* bathtub edges are
    # captured.  A 1 UI sweep centred on zero misses the early-side edge
    # whenever the eye centre is itself offset from the pulse peak, and the
    # jitter fit then has only one edge to work with.
    phases = np.linspace(-0.5 * phase_span, 0.5 * phase_span, n_phase)

    # Fixed DFE taps, solved once at the tap phase.
    taps = np.zeros(0)
    if n_taps:
        c0 = cursors_at_phase(p, cfg, k_peak, tap_phase_ui)
        taps = c0[cfg.n_pre + 1: cfg.n_pre + 1 + n_taps].copy()

    mains = np.zeros(n_phase)
    reach = 0.0
    cur = []
    for ph in phases:
        c = cursors_at_phase(p, cfg, k_peak, ph)
        pre, main, post = c[: cfg.n_pre], c[cfg.n_pre], c[cfg.n_pre + 1 :].copy()
        post[: taps.size] -= taps            # residual, not deletion
        inter = np.concatenate([pre, post])
        cur.append((main, inter))
        reach = max(reach, abs(main) + float(np.abs(inter).sum()))

    thresholds = np.linspace(-reach, reach, n_threshold)
    ber = np.zeros((n_phase, n_threshold))
    for k, (main, inter) in enumerate(cur):
        v_isi, pdf = isi_pdf(inter)
        mains[k] = main
        ber[k] = ber_at_thresholds(main, v_isi, pdf, thresholds, sigma_n)

    if sigma_rj_ui > 0 or dj_pp_ui > 0:
        ber = _convolve_jitter(phases, ber, sigma_rj_ui, dj_pp_ui)

    return StatEye(phases, thresholds, ber, mains, sigma_n,
                   sigma_rj_ui, dj_pp_ui)


def _convolve_jitter(
    phases: np.ndarray, ber: np.ndarray, sigma_rj: float, dj_pp: float
) -> np.ndarray:
    """Smear the BER map along the phase axis by the jitter density.

    Dual-Dirac DJ plus Gaussian RJ:

        f(t) = 0.5*N(t - dj/2; sigma) + 0.5*N(t + dj/2; sigma)

    Done as a direct matrix product rather than an FFT: the bathtub tail at
    1e-12 is a Gaussian tail, and FFT round-off would floor it around 1e-16
    relative -- close enough to matter.  Edges are handled by clamping, i.e.
    the BER outside the computed span is taken as its edge value, which is
    conservative since BER only rises toward the eye edges.
    """
    dt = float(phases[1] - phases[0])
    half = dj_pp / 2.0
    span = max(6.0 * sigma_rj + half, dt)
    n_k = int(np.ceil(span / dt))
    tau = np.arange(-n_k, n_k + 1) * dt

    if sigma_rj > 0:
        kern = 0.5 * (stats.norm.pdf(tau, +half, sigma_rj)
                      + stats.norm.pdf(tau, -half, sigma_rj))
    else:
        kern = np.zeros_like(tau)
        kern[np.argmin(np.abs(tau - half))] += 0.5 / dt
        kern[np.argmin(np.abs(tau + half))] += 0.5 / dt
    kern = kern / kern.sum()

    idx = np.clip(np.arange(phases.size)[:, None] - np.arange(-n_k, n_k + 1)[None, :],
                  0, phases.size - 1)
    return np.einsum("pk,pkt->pt", kern[None, :] * np.ones((phases.size, 1)),
                     ber[idx])


@dataclass
class JitterFit:
    """Dual-Dirac decomposition extracted from a bathtub curve."""

    rj_rms_ui: float
    dj_pp_ui: float
    mu_left: float
    mu_right: float
    sigma_left: float
    sigma_right: float
    n_points: int

    def tj(self, ber: float = 1e-12) -> float:
        """Total jitter at a target BER: ``TJ = DJ + 2*Q(BER)*RJ``."""
        return self.dj_pp_ui + 2.0 * float(q_of_ber(ber)) * self.rj_rms_ui

    def eye_width(self, ber: float = 1e-12) -> float:
        """Predicted eye width, ``1 UI - TJ(BER)``."""
        return 1.0 - self.tj(ber)

    def summary(self) -> str:
        return (
            f"RJ = {self.rj_rms_ui * 1e3:.3f} mUI rms, "
            f"DJ = {self.dj_pp_ui * 1e3:.2f} mUI pp, "
            f"TJ(1e-12) = {self.tj() * 1e3:.1f} mUI, "
            f"predicted EW = {self.eye_width():.4f} UI"
        )


def fit_dual_dirac(
    phase_ui: np.ndarray,
    ber: np.ndarray,
    q_lo: float = 3.0,
    q_hi: float = 7.0,
) -> JitterFit:
    """Separate RJ and DJ by straight-line fitting on the Q scale.

    Converting BER to Q linearises each bathtub edge, because a Gaussian tail
    satisfies

        Q_left(t)  = (t - mu_L)/sigma_L        Q_right(t) = (mu_R - t)/sigma_R

    Fitting a line to each edge therefore gives ``sigma`` from the slope -- the
    random jitter -- and, extrapolating to ``Q = 0``, the dual-Dirac means
    ``mu_L`` and ``mu_R``.  The deterministic jitter is what separates those
    means from an ideal 1 UI eye:

        RJ = (sigma_L + sigma_R)/2
        DJ = 1 - (mu_R - mu_L)

    The fit region matters.  Below ``q_lo`` the tail is not yet Gaussian --
    bounded ISI still dominates and would inflate the apparent RJ; above
    ``q_hi`` the data is usually below the noise floor of whatever produced it.
    The conventional window is Q = 3..7, i.e. BER 1e-3 to 1e-12.

    Note that ISI is itself deterministic jitter: data-dependent jitter appears
    in ``DJ``, so a fitted DJ larger than what was injected is the ISI showing
    up, not an error.
    """
    phase_ui = np.asarray(phase_ui, dtype=float)
    ber = np.asarray(ber, dtype=float)
    mid = plateau_centre(ber)

    def edge(sl: slice, sign: float) -> tuple[float, float, int]:
        t, b = phase_ui[sl], ber[sl]
        with np.errstate(divide="ignore", invalid="ignore"):
            q = q_of_ber(np.clip(b, 1e-300, 0.5 - 1e-12))
        ok = np.isfinite(q) & (q >= q_lo) & (q <= q_hi)
        if ok.sum() < 3:
            return float("nan"), float("nan"), int(ok.sum())
        # q = sign*(t - mu)/sigma  ->  q = (sign/sigma)*t - sign*mu/sigma
        slope, intercept = np.polyfit(t[ok], q[ok], 1)
        sigma = sign / slope
        mu = -intercept * sigma / sign
        return float(sigma), float(mu), int(ok.sum())

    s_l, mu_l, n_l = edge(slice(0, mid + 1), +1.0)
    s_r, mu_r, n_r = edge(slice(mid, None), -1.0)

    rj = float(np.nanmean([abs(s_l), abs(s_r)]))
    dj = float(1.0 - (mu_r - mu_l))
    return JitterFit(rj, dj, mu_l, mu_r, s_l, s_r, n_l + n_r)


# ==========================================================================
# Time-domain eye
# ==========================================================================

@dataclass
class TimeEye:
    """Eye folded from a simulated waveform."""

    phase_ui: np.ndarray
    traces: np.ndarray              # (n_symbols, samples_per_ui+1)
    upper_min: np.ndarray
    lower_max: np.ndarray
    n_symbols: int

    def eye_height(self) -> tuple[float, float]:
        """Worst-case opening over the observed patterns, and its phase."""
        opening = self.upper_min - self.lower_max
        j = int(np.argmax(opening))
        return float(opening[j]), float(self.phase_ui[j])

    def eye_width(self) -> float:
        """Phase span over which every observed trace stays on its own side."""
        ok = (self.upper_min > 0) & (self.lower_max < 0)
        if not ok.any():
            return 0.0
        idx = np.nonzero(ok)[0]
        return float(self.phase_ui[idx[-1]] - self.phase_ui[idx[0]])


def fold_eye(
    waveform: np.ndarray, cfg: LinkConfig, k_peak: int, decisions: np.ndarray
) -> TimeEye:
    """Fold a waveform into a one-UI eye, aligned on the pulse peak.

    Traces are split by the transmitted symbol so the worst-case opening is the
    gap between the lowest 'one' trace and the highest 'zero' trace.  That gap
    is the eye height *for the patterns that actually occurred* -- a PRBS emits
    a vanishing fraction of the 2^200 possible ISI combinations, which is
    precisely why this is optimistic relative to the statistical result.
    """
    m = cfg.samples_per_ui
    start = k_peak - m // 2
    n = min(decisions.size, (waveform.size - start - m - 1) // m)
    if n < 2:
        raise ValueError("waveform too short to fold an eye")

    idx = start + np.arange(n)[:, None] * m + np.arange(m + 1)[None, :]
    traces = waveform[idx]
    d = np.asarray(decisions[:n])

    ones, zeros = traces[d > 0], traces[d < 0]
    upper = ones.min(axis=0) if ones.size else np.full(m + 1, np.nan)
    lower = zeros.max(axis=0) if zeros.size else np.full(m + 1, np.nan)
    phase = (np.arange(m + 1) - m // 2) / m
    return TimeEye(phase, traces, upper, lower, n)
