"""Channel: Touchstone .s4p -> Sdd21 -> impulse response -> pulse response -> cursors.

The full path from a measured 4-port single-ended S-parameter file to the
sampled channel coefficients ``h_m`` that the DFE cancels and the statistical
engine sums over.  Cascading the channel with the CTLE lives in
:mod:`src.analog`; everything here concerns the channel alone.

All quantities are SI: frequency in Hz, time in s, voltage in V.

Port ordering
-------------
A 4-port differential file can use either of two common conventions, and the
file itself does not record which:

    THROUGH_PAIRED  ports (1,2) are the two ends of one line, (3,4) the two
                    ends of the other.  Differential pairs are {1,3} at the
                    near end and {2,4} at the far end.  Strong |S21|, |S43|.

    END_PAIRED      ports (1,2) are the (+,-) of the near-end pair and (3,4)
                    the (+,-) of the far end.  Strong |S31|, |S42|.
                    This is the ordering scikit-rf's se2gmm expects.

**Symptoms of getting it wrong.**  The failure is not an exception -- it is a
plausible-looking result computed from the wrong transmission term, usually the
near-end coupling.  On the bundled 802.3ck backplane, assuming END_PAIRED when
the file is actually THROUGH_PAIRED produces:

    - insertion loss that is *non-monotonic* in frequency
      (24.3 dB at 1 GHz, 18.5 dB at 4 GHz, 35.8 dB at 8 GHz)
    - return loss *better* than insertion loss, which is physically absurd
      for a through channel
    - DC gain 0.048 instead of 0.955
    - propagation delay 0.10 ns instead of 4.91 ns -- roughly 50x too short
      for 32 inches of stripline

If insertion loss is not smooth and monotonically increasing, suspect the port
ordering before suspecting the data.  :func:`detect_port_order` infers it from
the coupling structure rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skrf as rf

from .config import LinkConfig

THROUGH_PAIRED = "through_paired"
END_PAIRED = "end_paired"


# ==========================================================================
# 1. Loading and port ordering
# ==========================================================================

def detect_port_order(ntwk: rf.Network, f_max: float = 1e9) -> str:
    """Infer the port-ordering convention from the coupling structure.

    The dominant off-diagonal transmission terms of a through channel are the
    two single-ended through paths, so comparing the candidate pairings

        through_paired  if  mean(|S21| + |S43|)  >  mean(|S31| + |S42|)
        end_paired      otherwise

    identifies the convention.

    The comparison is restricted to ``f <= f_max`` (1 GHz by default), and this
    restriction is essential rather than cosmetic.  A lossy backplane's through
    path falls off steeply with frequency while residual coupling does not, so
    over a full 0-50 GHz band the coupling terms outweigh the through terms and
    invert the test.  On the bundled channel the full-band sums are through 576
    vs. end 749 -- the wrong answer -- while below 1 GHz they are 1.55 vs. 0.19,
    an unambiguous 8:1.  Near DC the channel is nearly lossless, which is what
    makes the test sound.

    Parameters
    ----------
    ntwk : skrf.Network
        4-port network as read from file, before any renumbering.
    f_max : float
        Upper frequency in Hz for the comparison.  At least the three lowest
        points are always used, in case the file starts above ``f_max``.

    Returns
    -------
    str
        ``THROUGH_PAIRED`` or ``END_PAIRED``.
    """
    s = np.abs(ntwk.s)
    mask = ntwk.f <= f_max
    if mask.sum() < 3:
        mask = np.zeros(ntwk.f.size, dtype=bool)
        mask[: min(3, ntwk.f.size)] = True
    through = (s[mask, 1, 0] + s[mask, 3, 2]).mean()
    end = (s[mask, 2, 0] + s[mask, 3, 1]).mean()
    return THROUGH_PAIRED if through > end else END_PAIRED


def load(path: str | Path, port_order: str | None = None) -> rf.Network:
    """Load a 4-port Touchstone file and renumber it to END_PAIRED ordering.

    The returned network has ports ordered (near+, near-, far+, far-), which is
    what :func:`differential_response` and scikit-rf's ``se2gmm`` require.

    Assumptions
    -----------
    - The file describes a through channel, so the ordering test of
      :func:`detect_port_order` applies.  A NEXT/FEXT aggressor file has no
      dominant through path; pass ``port_order`` explicitly for those.
    - Reference impedance is whatever the file declares (50 ohm here);
      no renormalization is done before the mixed-mode transform.

    Parameters
    ----------
    path : str or Path
        Path to the .s4p file.
    port_order : str, optional
        Force a convention.  Defaults to :func:`detect_port_order`.

    Returns
    -------
    skrf.Network
        4-port network in END_PAIRED single-ended ordering.
    """
    ntwk = rf.Network(str(path))
    if ntwk.nports != 4:
        raise ValueError(f"expected a 4-port network, got {ntwk.nports} ports")

    order = port_order or detect_port_order(ntwk)
    if order == THROUGH_PAIRED:
        # {1,3} are the near-end pair and {2,4} the far-end pair, so move
        # old port 1 -> new index 2 and old port 2 -> new index 1.
        ntwk.renumber([0, 1, 2, 3], [0, 2, 1, 3])
    elif order != END_PAIRED:
        raise ValueError(f"unknown port_order {order!r}")
    return ntwk


def check_passivity(ntwk: rf.Network, tol: float = 1e-3) -> float:
    """Largest |S| in the file, as a data-sanity check.

    A passive network satisfies |S_ij| <= 1 for every entry.  Values above 1
    indicate a measurement or de-embedding artifact, and will show up later as
    an impulse response that gains energy.

    Returns the maximum magnitude; raises if it exceeds ``1 + tol``.
    """
    peak = float(np.abs(ntwk.s).max())
    if peak > 1.0 + tol:
        raise ValueError(f"network is not passive: max|S| = {peak:.4f}")
    return peak


# ==========================================================================
# 2. Mixed-mode extraction
# ==========================================================================

def differential_response(ntwk: rf.Network) -> tuple[np.ndarray, np.ndarray]:
    """Extract the differential insertion response Sdd21(f).

    For a pair with terminals (a+, a-) driving a pair (b+, b-), the
    differential-to-differential transmission term is

        Sdd_ba = 1/2 * ( S_{b+,a+} - S_{b+,a-} - S_{b-,a+} + S_{b-,a-} )

    which is the ba element of ``M S M^-1`` under the standard mixed-mode
    transform M.  This function applies scikit-rf's generalized mixed-mode
    conversion, which additionally renormalises the differential reference
    impedance to 2*z0 (100 ohm for a 50 ohm file).

    Assumptions
    -----------
    - Ports are in END_PAIRED ordering; see :func:`load`.
    - The pair is nominally symmetric.  Asymmetry appears as mode conversion
      (Scd21), which this function discards -- use :func:`mixed_mode` to see it.

    Returns
    -------
    f : np.ndarray
        Frequency in Hz, as recorded in the file.
    sdd21 : np.ndarray
        Complex differential insertion response, unitless.
    """
    mm = ntwk.copy()
    mm.se2gmm(p=2)
    return mm.f.copy(), mm.s[:, 1, 0].copy()


def mixed_mode(ntwk: rf.Network) -> dict[str, np.ndarray]:
    """Return the full set of mixed-mode terms of interest.

    Indices after ``se2gmm(p=2)`` are 0,1 = differential ports 1,2 and
    2,3 = common-mode ports 1,2.

    Returns
    -------
    dict
        ``f``, ``sdd21`` (differential insertion), ``sdd11`` (differential
        return), ``scd21`` (differential-to-common mode conversion),
        ``scc21`` (common-mode insertion).
    """
    mm = ntwk.copy()
    mm.se2gmm(p=2)
    return {
        "f": mm.f.copy(),
        "sdd21": mm.s[:, 1, 0].copy(),
        "sdd11": mm.s[:, 0, 0].copy(),
        "scd21": mm.s[:, 3, 0].copy(),
        "scc21": mm.s[:, 3, 2].copy(),
    }


def insertion_loss_db(sdd21: np.ndarray) -> np.ndarray:
    """Insertion loss in dB, positive for a lossy channel.

        IL(f) = -20 * log10( |Sdd21(f)| )
    """
    return -20.0 * np.log10(np.abs(sdd21) + 1e-30)


def loss_at(f: np.ndarray, sdd21: np.ndarray, f_target: float) -> float:
    """Insertion loss in dB at the frequency point nearest ``f_target`` (Hz)."""
    i = int(np.argmin(np.abs(f - f_target)))
    return float(insertion_loss_db(sdd21)[i])


def insertion_loss_report(
    f: np.ndarray,
    sdd21: np.ndarray,
    cfg: LinkConfig,
    extra: tuple[float, ...] = (1e9, 2e9, 4e9, 12e9, 16e9),
) -> dict[str, float]:
    """Print insertion loss at the Nyquist frequency and a few reference points.

    The Nyquist value is the headline number: for NRZ at symbol rate Rs the
    strongest spectral content of a 1010 pattern sits at Rs/2, so IL(Rs/2) is
    the single figure that most nearly summarises how hard the channel is.

    The figure itself is drawn by ``plotting.plot_channel_response`` -- by
    project convention only ``plotting.py`` writes to disk.

    Returns
    -------
    dict
        Loss in dB keyed by a frequency label, plus ``"nyquist"``.
    """
    nyq = loss_at(f, sdd21, cfg.nyquist)
    print(f"insertion loss (Nyquist = {cfg.nyquist / 1e9:.0f} GHz for "
          f"{cfg.bit_rate / 1e9:.0f} Gbps NRZ)")
    out: dict[str, float] = {"nyquist": nyq}
    for ftgt in sorted({*extra, cfg.nyquist}):
        il = loss_at(f, sdd21, ftgt)
        mark = "  <-- Nyquist" if ftgt == cfg.nyquist else ""
        print(f"  {ftgt / 1e9:5.1f} GHz : {il:6.2f} dB{mark}")
        out[f"{ftgt / 1e9:g}GHz"] = il
    return out


# ==========================================================================
# 3. Frequency grid preparation
# ==========================================================================

def analysis_grid(cfg: LinkConfig) -> np.ndarray:
    """One-sided frequency grid, in Hz, matching ``cfg``'s time record.

        f[k] = k * fs / n_time,   k = 0 .. n_time/2

    This is the grid ``np.fft.irfft`` inverts, so DC and fs/2 are both present.

    Note that ``fs = M * Rs``, not ``2 * f_max``.  Deriving the sample rate from
    the file's upper frequency would give 100 GHz here, i.e. 6.25 samples per
    UI -- non-integer, so symbol-rate sampling could not land on a sample.
    """
    return np.arange(cfg.n_time // 2 + 1) * cfg.df


@dataclass(frozen=True)
class GridReport:
    """What :func:`prepare_response` had to do to the measured data."""

    dc_extrapolated: bool
    f_min_measured: float
    f_max_measured: float
    uniform_input: bool
    truncation_db: float
    tapered: bool
    band_limited: bool

    def summary(self) -> str:
        bits = [
            f"input {self.f_min_measured / 1e9:.3f}-{self.f_max_measured / 1e9:.1f} GHz",
            "uniform" if self.uniform_input else "NON-UNIFORM (resampled)",
            "DC extrapolated" if self.dc_extrapolated else "DC measured",
            f"truncation at {self.truncation_db:.1f} dB",
        ]
        if self.tapered:
            bits.append("tapered")
        if self.band_limited:
            bits.append("band-limited above grid Nyquist")
        return ", ".join(bits)


def prepare_response(
    f: np.ndarray,
    h: np.ndarray,
    cfg: LinkConfig,
    taper: bool = False,
    taper_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, GridReport]:
    """Put a measured response on the uniform DC-inclusive grid an IFFT needs.

    Four things are handled explicitly, because each is a silent failure if
    skipped:

    **1. Resampling to a uniform grid.**  ``irfft`` interprets bin ``k`` as
    ``k*fs/N``; a non-uniform or offset input grid is silently reinterpreted and
    the impulse response comes out with the wrong time axis.  Magnitude and
    unwrapped phase are interpolated separately,

        |H(f)|   by linear interpolation of |H|
        arg H(f) by linear interpolation of unwrap(arg H)

    rather than the real and imaginary parts.  A channel with delay tau has
    phase rotating at 2*pi*f*tau -- 38 degrees per 20 MHz step here -- and
    linear interpolation of a rotating phasor's Cartesian components chords
    across the arc, biasing the magnitude low by an amount that grows with
    frequency.  Interpolating magnitude and phase separately does not.

    **2. DC extrapolation.**  If the file starts above DC the response is
    extended down with the magnitude held at its lowest measured value and the
    phase forced to zero, since a physical channel has real H(0).  A complex or
    zero DC term shows up as an imaginary component in h(t) or as baseline
    droop in the pulse response.

    **3. Band limiting.**  Above the highest measured frequency the response is
    set to zero.  Truncating a response that has *not* yet rolled off produces
    Gibbs ringing and energy before the pulse arrives -- an acausal impulse
    response.  ``GridReport.truncation_db`` records the level at the cut so this
    is checkable rather than assumed; below roughly -40 dB it is harmless.  For
    channels where it is not, ``taper=True`` applies a raised-cosine roll-off
    over the top ``taper_frac`` of the measured band:

        w(f) = 0.5 * (1 + cos(pi * (f - f_a) / (f_max - f_a)))

    **4. Conjugate symmetry.**  A real h(t) requires H(-f) = H*(f).  Rather than
    mirroring the spectrum by hand -- where the classic bug is duplicating or
    dropping the Nyquist bin, leaving an imaginary residue that gets quietly
    discarded by ``.real`` -- this uses ``np.fft.irfft``, which imposes the
    symmetry structurally.  It requires bins 0 and N/2 to be real, which is
    enforced here.

    Returns
    -------
    f_grid, h_grid : np.ndarray
        The analysis grid in Hz and the complex response on it.
    report : GridReport
        What was extrapolated, resampled, truncated, or tapered.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h)
    if f.size < 2:
        raise ValueError("need at least two frequency points")
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequency axis must be strictly increasing")

    uniform = bool(np.allclose(np.diff(f), f[1] - f[0]))
    mag = np.abs(h)
    phase = np.unwrap(np.angle(h))

    # -- 2. DC extrapolation ------------------------------------------------
    dc_extrapolated = f[0] > 0.0
    if dc_extrapolated:
        f = np.concatenate(([0.0], f))
        mag = np.concatenate(([mag[0]], mag))
        phase = np.concatenate(([0.0], phase))

    # -- 3. optional taper before truncation --------------------------------
    if taper:
        f_a = f[-1] - taper_frac * (f[-1] - f[0])
        band = f >= f_a
        w = 0.5 * (1.0 + np.cos(np.pi * (f[band] - f_a) / (f[-1] - f_a)))
        mag = mag.copy()
        mag[band] *= w

    # -- 1. resample onto the uniform analysis grid -------------------------
    f_grid = analysis_grid(cfg)
    h_grid = np.zeros(f_grid.size, dtype=complex)
    inband = f_grid <= f[-1]
    h_grid[inband] = np.interp(f_grid[inband], f, mag) * np.exp(
        1j * np.interp(f_grid[inband], f, phase)
    )

    # -- 4. Hermitian requirements for a real inverse transform -------------
    h_grid[0] = h_grid[0].real
    h_grid[-1] = h_grid[-1].real

    trunc_db = float(-20.0 * np.log10(np.abs(h[-1]) + 1e-30))
    return (
        f_grid,
        h_grid,
        GridReport(
            dc_extrapolated=dc_extrapolated,
            f_min_measured=float(f[1] if dc_extrapolated else f[0]),
            f_max_measured=float(f[-1]),
            uniform_input=uniform,
            truncation_db=trunc_db,
            tapered=bool(taper),
            band_limited=bool(f[-1] < f_grid[-1]),
        ),
    )


# ==========================================================================
# 4. Impulse response
# ==========================================================================

def impulse_response(
    f: np.ndarray, h: np.ndarray, cfg: LinkConfig, taper: bool = False
) -> tuple[np.ndarray, np.ndarray, GridReport]:
    """Impulse response h(t) from a measured frequency response.

    After :func:`prepare_response` has put the response on a uniform
    DC-inclusive grid, the real inverse DFT gives

        h[n] = (1/N) * sum_k H[k] * exp(j*2*pi*k*n/N),   N = n_time

    with H(-f) = H*(f) imposed by ``irfft``.

    **Normalization.**  numpy's ``irfft`` carries the 1/N, so ``h`` is the
    discrete impulse response that convolves directly with a signal sampled at
    ``cfg.dt`` -- there is no additional ``dt`` factor.  Guessing this wrong
    scales every downstream voltage by N or by dt.  The consequence
    ``sum(h) == H(0)`` is the self-test in :func:`check_dc_gain`.

    **Time window.**  The record spans ``T = n_time/fs`` and the grid spacing is
    ``df = 1/T``.  These are the same statement: too coarse a df is too short a
    window, and a window shorter than the impulse response tail wraps that tail
    around onto the start of the record, where it reads as spurious pre-cursor
    ISI.  :func:`check_wraparound` tests for it.

    Assumptions
    -----------
    - The channel is LTI and passive.
    - Behaviour above the measured band contributes negligibly; see
      ``GridReport.truncation_db``.

    Returns
    -------
    t : np.ndarray
        Time in s, starting at 0, spacing ``cfg.dt``.
    h_t : np.ndarray
        Real impulse response, unitless per sample.
    report : GridReport
        Grid preparation diagnostics.
    """
    _, h_grid, report = prepare_response(f, h, cfg, taper=taper)
    h_t = np.fft.irfft(h_grid, n=cfg.n_time)
    t = np.arange(cfg.n_time) * cfg.dt
    return t, h_t, report


def check_dc_gain(h_t: np.ndarray, h_dc: complex) -> tuple[float, float]:
    """Self-test: the impulse response must sum to the DC gain.

        sum_n h[n] = H(0)

    Returns ``(sum_h, |sum_h - H(0)|)``.  A large residual means the frequency
    grid was resampled, extrapolated, or normalized badly.
    """
    total = float(np.sum(h_t))
    return total, abs(total - float(np.real(h_dc)))


def check_causality(
    h_t: np.ndarray, onset_frac: float = 0.01, tol: float = 1e-3
) -> tuple[float, float]:
    """Fraction of impulse-response energy arriving before the causal onset.

    A causal channel has essentially zero response before its propagation
    delay.  Energy that appears there comes from band truncation (Gibbs
    ringing) or from phase inconsistency in the measured data, and it
    contaminates the pre-cursors -- which is where it does the most damage,
    since a DFE cannot cancel them.

    The onset is the first sample exceeding ``onset_frac`` of the peak, and the
    test is the energy fraction strictly before it.

    **Not "energy before the peak."**  That is the obvious formulation and it
    is wrong: the rising edge is legitimate causal content, and for a sharp
    arrival it carries a large share of the total.  On this channel the rise
    from 1 % to peak spans only 16 samples (half a UI) yet holds 27 % of the
    energy -- a number that looks alarming and means nothing.  Measured from
    the onset instead, the same response gives 1.4e-5, i.e. genuinely causal.

    Returns ``(pre_fraction, tol)``; compare the two.  For reference, this
    channel measures 1.4e-5 untapered and 1.0e-5 with ``taper=True``, so the
    50 GHz truncation at -60 dB is doing no harm and the taper is unnecessary
    here -- it changes the main cursor by 0.001 %.
    """
    energy = float(np.sum(h_t**2))
    if energy <= 0:
        return 0.0, tol
    mag = np.abs(h_t)
    onset = int(np.argmax(mag > onset_frac * mag.max()))
    return float(np.sum(h_t[:onset] ** 2)) / energy, tol


def check_wraparound(h_t: np.ndarray, tail_frac: float = 0.1) -> float:
    """Fraction of energy in the last ``tail_frac`` of the record.

    If the impulse response has not decayed by the end of the window, the
    circular nature of the DFT wraps the remainder onto the beginning, where it
    masquerades as pre-cursor ISI.  A value near zero means the window is long
    enough; raise ``cfg.n_time`` if it is not.
    """
    n = int(h_t.size * (1.0 - tail_frac))
    energy = float(np.sum(h_t**2))
    return float(np.sum(h_t[n:] ** 2)) / energy if energy > 0 else 0.0


# ==========================================================================
# 5. Pulse response
# ==========================================================================

def pulse_response(h_t: np.ndarray, cfg: LinkConfig) -> np.ndarray:
    """Response to a single rectangular pulse one UI wide and 1 V tall.

        p(t) = integral_0^UI h(t - tau) d_tau

    Discretely this is the convolution of h with a boxcar of exactly
    ``samples_per_ui`` ones -- not a single sample.  A one-sample excitation is
    an impulse, not a symbol, and yields cursors roughly M times too small.  The
    result is truncated back to the original record length.

    Assumptions
    -----------
    - Ideal rectangular TX pulse: zero rise time, no TX equalization.  A real
      driver's finite edge rate would low-pass the pulse slightly further.
    - Linearity, so any data pattern follows by superposition:
      ``y(t) = sum_k a_k * p(t - k*UI)``.

    Returns
    -------
    np.ndarray
        Pulse response in V, same length as ``h_t``.
    """
    box = np.ones(cfg.samples_per_ui)
    return np.convolve(h_t, box)[: h_t.size]


# ==========================================================================
# 6. Cursors
# ==========================================================================

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
    across phase rather than at a single point.

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

    The main cursor is located as ``argmax(p)``.  This identifies the *slot*,
    which is unambiguous; it does not commit to a sampling phase, which is why
    the full sub-UI waveform is retained per slot.

    Slots falling outside the record are filled with zeros rather than wrapping,
    so a truncated tail reads as absent ISI rather than as ISI taken from the
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
    Prefer the ``SplitPulse`` API wherever the sampling phase matters -- which
    is everywhere the CDR is involved.

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
    tail -- h1 down, h_-1 up -- so the discriminant decreases monotonically and
    has at most one crossing within a UI.

    Returns
    -------
    j : int
        Nearest phase index to the lock point.
    phase_ui : float
        Lock phase in UI relative to the peak, linearly interpolated between
        the bracketing samples.  Falls back to the closest available phase if
        no sign change occurs within the UI.
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
class CursorReport:
    """Automatically identified cursor structure at one sampling phase."""

    phase_ui: float
    main: float
    main_slot: int
    pre: np.ndarray
    post: np.ndarray
    significant_pre: list[tuple[int, float]]
    significant_post: list[tuple[int, float]]
    isi_total: float
    eye_worst_case: float
    threshold: float

    @property
    def n_significant_post(self) -> int:
        return len(self.significant_post)

    def format(self, max_rows: int = 12) -> str:
        """Human-readable summary."""
        lines = [
            f"cursors at phase {self.phase_ui:+.4f} UI "
            f"(threshold {self.threshold * 1e3:.2f} mV = "
            f"{self.threshold / self.main * 100:.1f}% of main)",
            f"  main   h0  = {self.main * 1e3:8.3f} mV   "
            f"(pulse slot {self.main_slot})",
        ]
        shown = self.significant_pre[-max_rows:]
        for m, v in shown:
            lines.append(f"  pre    h{m:+d} = {v * 1e3:8.3f} mV   "
                         f"({v / self.main:+.4f} of main)")
        if not self.significant_pre:
            lines.append("  pre    none above threshold")
        for m, v in self.significant_post[:max_rows]:
            lines.append(f"  post   h{m:+d} = {v * 1e3:8.3f} mV   "
                         f"({v / self.main:+.4f} of main)")
        extra = self.n_significant_post - min(max_rows, self.n_significant_post)
        if extra > 0:
            lines.append(f"  post   ... {extra} more above threshold")
        lines += [
            f"  significant : {len(self.significant_pre)} pre, "
            f"{self.n_significant_post} post",
            f"  total |ISI| = {self.isi_total * 1e3:8.3f} mV",
            f"  worst-case eye = {self.eye_worst_case * 1e3:8.3f} mV "
            f"({'OPEN' if self.eye_worst_case > 0 else 'CLOSED'})",
        ]
        return "\n".join(lines)


def identify_cursors(
    sp: SplitPulse, j: int | None = None, threshold_frac: float = 0.01
) -> CursorReport:
    """Identify main, pre- and post-cursors automatically at one phase.

    The main cursor is the slot the pulse peak falls in; pre-cursors are the
    slots before it and post-cursors those after.  A cursor counts as
    *significant* when

        |h_m| >= threshold_frac * |h0|

    which is a reporting threshold only -- the ISI and eye totals below use
    every cursor in the record, since truncating the tail overstates the eye
    (docs/spec.md section 4).

    Worst-case eye follows peak-distortion analysis, where every interfering
    symbol takes its worst sign:

        eye = h0 - sum_{m != 0} |h_m|

    Parameters
    ----------
    sp : SplitPulse
        Split pulse response.
    j : int, optional
        Sampling phase index; defaults to the peak phase.
    threshold_frac : float
        Significance threshold as a fraction of the main cursor.

    Returns
    -------
    CursorReport
    """
    j = sp.peak_phase if j is None else j
    pre, main, post = sp.split(j)
    thr = threshold_frac * abs(main)

    sig_pre = [(-(sp.n_pre - i), float(v)) for i, v in enumerate(pre)
               if abs(v) >= thr]
    sig_post = [(i + 1, float(v)) for i, v in enumerate(post) if abs(v) >= thr]
    isi = float(np.abs(pre).sum() + np.abs(post).sum())

    return CursorReport(
        phase_ui=float(sp.phase_ui(j)),
        main=main,
        main_slot=sp.k_peak,
        pre=pre,
        post=post,
        significant_pre=sig_pre,
        significant_post=sig_post,
        isi_total=isi,
        eye_worst_case=main - isi,
        threshold=thr,
    )


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

    Assumptions
    -----------
    - Decisions are correct, so no error propagation.
    - Taps are ideal and unquantized, with zero feedback delay.

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
