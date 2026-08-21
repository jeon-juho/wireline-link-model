"""Analog chain: frequency-response composition and time-domain extraction.

The channel and the CTLE are both LTI, so they compose by multiplication in the
frequency domain.  The single shared artifact this module produces -- the
impulse response, and the pulse response derived from it -- is what both the
time-domain and the statistical engine consume.  Computing it once here is what
makes the two methods comparable (spec.md criterion 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LinkConfig


def compose(*responses: np.ndarray) -> np.ndarray:
    """Cascade LTI blocks by multiplying their frequency responses.

        H(f) = H_1(f) * H_2(f) * ... * H_n(f)

    Valid only if the blocks do not load one another; see assumption 1 in
    docs/architecture.md.  All inputs must already be on a common grid.
    """
    if not responses:
        raise ValueError("compose() needs at least one response")
    out = np.ones_like(responses[0], dtype=complex)
    for h in responses:
        if h.shape != out.shape:
            raise ValueError("all responses must share the same frequency grid")
        out = out * h
    return out


def analysis_grid(cfg: LinkConfig) -> np.ndarray:
    """One-sided frequency grid, in Hz, matching ``cfg``'s time record.

        f[k] = k * fs / n_time,   k = 0 .. n_time/2

    This is the grid ``np.fft.irfft`` inverts, so DC and fs/2 are both present.
    """
    return np.arange(cfg.n_time // 2 + 1) * cfg.df


def resample(
    f: np.ndarray, h: np.ndarray, cfg: LinkConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a measured response onto the analysis grid.

    Magnitude and unwrapped phase are interpolated separately,

        |H(f)|  by linear interpolation of |H|
        arg H(f) by linear interpolation of unwrap(arg H)

    rather than interpolating the real and imaginary parts.  A channel with
    propagation delay tau has phase rotating at 2*pi*f*tau; linear
    interpolation of a rotating phasor's Cartesian components chords across the
    arc and biases the magnitude low, while interpolating magnitude and phase
    separately does not.

    Above the highest measured frequency the response is set to zero.  For a
    lossy backplane this is benign because the measured response has already
    fallen far below the signal band of interest, but it is an extrapolation
    choice, not a measurement -- see the assumptions in docs/architecture.md.

    Returns
    -------
    f_grid : np.ndarray
        The analysis grid in Hz.
    h_grid : np.ndarray
        Complex response on that grid, zero above ``f.max()``.
    """
    f_grid = analysis_grid(cfg)
    h_grid = np.zeros(f_grid.size, dtype=complex)

    inband = f_grid <= f[-1]
    mag = np.interp(f_grid[inband], f, np.abs(h))
    phase = np.interp(f_grid[inband], f, np.unwrap(np.angle(h)))
    h_grid[inband] = mag * np.exp(1j * phase)

    # irfft requires a real DC bin and a real Nyquist bin for a real h(t).
    h_grid[0] = h_grid[0].real
    h_grid[-1] = h_grid[-1].real
    return f_grid, h_grid


def impulse_response(
    f: np.ndarray, h: np.ndarray, cfg: LinkConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Impulse response h(t) from a measured frequency response.

    The one-sided response is resampled onto the analysis grid and inverted
    with a real inverse DFT, which imposes Hermitian symmetry H(-f) = H*(f):

        h[n] = (1/N) * sum_k H[k] * exp(j*2*pi*k*n/N),   N = n_time

    numpy's ``irfft`` carries the 1/N, so ``h`` is the discrete impulse
    response that convolves directly with a signal sampled at ``cfg.dt`` -- no
    further dt scaling is needed.  As a consequence ``h.sum() == H(0)``, the DC
    gain, which :func:`check_dc_gain` uses as a self-test.

    Returns
    -------
    t : np.ndarray
        Time in s, starting at 0, spacing ``cfg.dt``.
    h_t : np.ndarray
        Real impulse response, unitless per sample.
    """
    _, h_grid = resample(f, h, cfg)
    h_t = np.fft.irfft(h_grid, n=cfg.n_time)
    t = np.arange(cfg.n_time) * cfg.dt
    return t, h_t


def pulse_response(h_t: np.ndarray, cfg: LinkConfig) -> np.ndarray:
    """Response to a single rectangular pulse one UI wide and 1 V tall.

        p(t) = integral_0^UI h(t - tau) d_tau

    Discretely this is the convolution of h with a boxcar of ``samples_per_ui``
    ones.  The result is truncated back to the original record length.

    The cursors of p -- its values sampled once per UI -- are the ISI terms the
    DFE cancels and the statistical engine sums over.
    """
    box = np.ones(cfg.samples_per_ui)
    return np.convolve(h_t, box)[: h_t.size]


@dataclass(frozen=True)
class SplitPulse:
    """The pulse response cut into per-symbol portions.

    ``portions[i, j]`` is the pulse response in cursor slot ``m = i - n_pre`` at
    sub-UI sampling phase ``j``:

        portions[i, j] = p[k_peak + (j - M/2) + (i - n_pre)*M]

    Sampling the pulse once per UI collapses this to a cursor vector, but the
    phase must be chosen first -- and the correct phase is not generally the
    peak.  A Mueller-Muller CDR locks where ``h1 == h_-1``
    (docs/dfe_notes.md section 6), and eye height must in any case be evaluated
    across phase rather than at a single point.  Keeping the full sub-UI
    waveform per slot is what makes both possible; it is StatOpt's
    ``splitPulse`` structure (docs/statopt_review.md section 5).

    Phase index ``j`` runs 0..M-1 and covers one full UI centred on the peak, so
    ``j = M//2`` is the peak phase and ``phase_ui(j)`` is the offset in UI.
    """

    portions: np.ndarray
    n_pre: int
    n_post: int
    k_peak: int
    samples_per_ui: int

    @property
    def n_cursor(self) -> int:
        return self.portions.shape[0]

    @property
    def peak_phase(self) -> int:
        """Phase index corresponding to the pulse-response peak."""
        return self.samples_per_ui // 2

    def phase_ui(self, j: int | np.ndarray) -> float | np.ndarray:
        """Sampling-phase offset from the peak, in UI."""
        return (np.asarray(j) - self.peak_phase) / self.samples_per_ui

    def at(self, j: int) -> np.ndarray:
        """Cursor vector at phase ``j``, ordered from ``-n_pre`` to ``+n_post``."""
        return self.portions[:, j]

    def h(self, m: int, j: int) -> float:
        """Single cursor ``h_m`` at phase ``j``."""
        return float(self.portions[m + self.n_pre, j])

    def split(self, j: int) -> tuple[np.ndarray, float, np.ndarray]:
        """``(precursors, main, postcursors)`` at phase ``j``."""
        c = self.at(j)
        return c[: self.n_pre], float(c[self.n_pre]), c[self.n_pre + 1 :]


def split_pulse(
    p: np.ndarray,
    cfg: LinkConfig,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> SplitPulse:
    """Cut the pulse response into per-symbol portions across one UI of phase.

    Slots that fall outside the record are filled with zeros rather than
    wrapping, so a truncated tail reads as no ISI rather than as ISI from the
    wrong part of the waveform.

    Depths default to ``cfg.n_pre`` and ``cfg.isi_depth``.
    """
    n_pre = cfg.n_pre if n_pre is None else n_pre
    n_post = cfg.isi_depth if n_post is None else n_post
    m = cfg.samples_per_ui
    k_peak = int(np.argmax(p))

    offsets = np.arange(m) - m // 2                       # sub-UI phase
    slots = np.arange(-n_pre, n_post + 1) * m             # cursor slot
    idx = k_peak + slots[:, None] + offsets[None, :]      # (n_cursor, M)

    portions = np.zeros(idx.shape)
    valid = (idx >= 0) & (idx < p.size)
    portions[valid] = p[idx[valid]]
    return SplitPulse(portions, n_pre, n_post, k_peak, m)


def cursors(
    p: np.ndarray, cfg: LinkConfig, n_pre: int | None = None, n_post: int | None = None
) -> tuple[int, np.ndarray]:
    """Cursor vector at the peak sampling phase.

    Convenience wrapper over :func:`split_pulse` for the phase-independent case.
    Prefer the ``SplitPulse`` API where the sampling phase matters -- which is
    everywhere the CDR is involved.

    Returns ``(k_peak, c)`` with ``c`` ordered from ``-n_pre`` to ``+n_post``.
    """
    sp = split_pulse(p, cfg, n_pre, n_post)
    return sp.k_peak, sp.at(sp.peak_phase)


def mm_lock_phase(sp: SplitPulse) -> tuple[int, float]:
    """Sampling phase a Mueller-Muller CDR locks to.

    The MM phase detector has mean output

        E[e_MM] = h1 - h_-1

    so the loop settles where the first pre- and postcursor are equal
    (docs/dfe_notes.md section 6).  Sampling later moves both cursors along the
    tail -- h1 down, h_-1 up -- so the discriminant is monotonically decreasing
    and has at most one crossing within a UI.

    Returns
    -------
    j : int
        Nearest phase index to the lock point.
    phase_ui : float
        Lock phase in UI relative to the peak, linearly interpolated between
        the two bracketing samples.  Falls back to the closest available phase
        if no sign change occurs within the UI.
    """
    d = sp.portions[sp.n_pre + 1, :] - sp.portions[sp.n_pre - 1, :]  # h1 - h_-1
    sign_change = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    if sign_change.size == 0:
        j = int(np.argmin(np.abs(d)))
        return j, float(sp.phase_ui(j))

    a = int(sign_change[0])
    frac = d[a] / (d[a] - d[a + 1])  # linear interpolation of the zero crossing
    j = a if frac < 0.5 else a + 1
    return int(j), float(sp.phase_ui(a) + frac / sp.samples_per_ui)


@dataclass(frozen=True)
class Distortion:
    """Peak-distortion analysis at one sampling phase."""

    phase_ui: float
    main: float
    isi: float
    eye: float
    taps: np.ndarray
    residual_isi: float
    eye_with_dfe: float

    @property
    def open(self) -> bool:
        return self.eye_with_dfe > 0.0


def peak_distortion(sp: SplitPulse, j: int, n_taps: int = 0) -> Distortion:
    """Worst-case eye at phase ``j``, with and without an ideal DFE.

    Peak distortion assumes every interfering symbol takes its worst sign:

        eye = h0 - sum_{m != 0} |h_m|

    A zero-forcing DFE with N taps cancels postcursors 1..N exactly, since for
    feedback taps the ZF and MMSE solutions coincide
    (docs/dfe_notes.md section 5.2):

        w_m = h_m / h0,   m = 1..N
        eye_dfe = h0 - sum_{pre} |h_m| - sum_{m > N} |h_m|

    Tap weights are returned unclipped; a weight beyond
    ``DFEConfig.max_tap_weight`` signals an error-propagation problem that
    should be surfaced, not silently limited.
    """
    pre, main, post = sp.split(j)
    isi = float(np.abs(pre).sum() + np.abs(post).sum())

    n_taps = max(0, min(n_taps, post.size))
    taps = (post[:n_taps] / main) if (n_taps and main != 0.0) else np.zeros(n_taps)
    residual = float(np.abs(pre).sum() + np.abs(post[n_taps:]).sum())

    return Distortion(
        phase_ui=float(sp.phase_ui(j)),
        main=main,
        isi=isi,
        eye=main - isi,
        taps=taps,
        residual_isi=residual,
        eye_with_dfe=main - residual,
    )


def eye_vs_phase(sp: SplitPulse, n_taps: int = 0) -> np.ndarray:
    """Worst-case eye height at every sampling phase, in V.

    The maximum of this curve is the best achievable sampling point; the MM
    lock phase is where the CDR will actually sit.  The two need not coincide,
    and the gap between them is a real margin loss.
    """
    return np.array(
        [peak_distortion(sp, j, n_taps).eye_with_dfe for j in range(sp.samples_per_ui)]
    )


def check_dc_gain(h_t: np.ndarray, h_dc: complex) -> tuple[float, float]:
    """Self-test: the impulse response must sum to the DC gain.

        sum_n h[n] = H(0)

    Returns ``(sum_h, |sum_h - H(0)|)``.  A large residual means the frequency
    grid was resampled or truncated badly.
    """
    total = float(np.sum(h_t))
    return total, abs(total - float(np.real(h_dc)))
