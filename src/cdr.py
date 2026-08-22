"""CDR: phase detector, digital loop filter, phase interpolator.

Produces the recovered clock phase over time.  That phase is what the slicer
and DFE sample on, which is why this module matters beyond the loop itself: the
cursors ``h_m = p(t_s + m*T)`` are functions of where the CDR settles, and on
this channel the difference between the pulse peak and the Mueller-Muller lock
point is the difference between an open and a closed eye (docs/spec.md
section 5).

Two detectors are provided because the specification is ambiguous.
``spec.md`` calls for a "baud-rate, Alexander-type" CDR, but those are
different architectures:

    Alexander (2x oversampled)   Needs a data sample *and* an edge sample per
                                 UI.  Locks near the actual zero crossing, so
                                 its lock point barely depends on the channel.
                                 Costs a second full-bandwidth sampler.

    Mueller-Muller (baud-rate)   One sample per UI, reusing the data slicer.
                                 Locks where h1 == h_-1, so the lock point is a
                                 property of the equalized pulse response and
                                 moves whenever the equalization moves.

"Bang-bang" describes the binary output and applies to both.  With a DFE in the
loop there is a real argument for baud-rate beyond power: DFE feedback cancels
ISI at the data sampling instant only, so the half-UI edge sample an Alexander
detector needs sees a signal whose ISI correction is simply wrong.

Nonlinearity
------------
The detector emits ``sign(dt)``, whose derivative is zero everywhere except at
the origin -- there is strictly no small-signal gain and no transfer function.
Analysis is possible only because jitter dithers the quantizer, making the
*average* output smooth:

    E[b] = erf(dt / (sigma_tau*sqrt(2)))   ->   K_pd = 2 / (sigma_tau*sqrt(2*pi))

This is a describing function, not an identity, and it carries three caveats
that the code exposes rather than hides:

- ``K_pd`` varies as ``1/sigma_tau``, so loop bandwidth is an operating-point
  property.  :func:`loop_parameters` therefore takes ``sigma_tau`` explicitly
  instead of pretending the loop has one bandwidth.
- The loop moves at most ``K_p`` per update, so phase slew is bounded by
  ``K_p/(M*T)`` regardless of small-signal bandwidth.  :attr:`CDRResult.slew_limited`
  reports whether a run hit that bound.
- ``erf`` saturates, so restoring force *weakens* for large errors and
  acquisition behaves differently from tracking.

All quantities are SI, except phase, which is in UI throughout -- it is the
natural unit here and avoids a factor of 2*pi in every expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CDRConfig, LinkConfig

# ==========================================================================
# Phase detectors
# ==========================================================================
#
# Sign convention, used everywhere in this module:
#
#     pd = +1  the clock is EARLY  -> increase phase, sample later
#     pd = -1  the clock is LATE   -> decrease phase, sample earlier
#     pd =  0  no phase information this symbol
#
# so the loop filter can add ``K_p * pd`` directly and still be negative
# feedback.


def alexander_pd(d_prev: int, d_curr: int, e: int) -> int:
    """Alexander (bang-bang) phase detector, from two data and one edge sample.

    ``e`` is sampled half a UI before ``d_curr``, i.e. at the nominal crossing
    between ``d_prev`` and ``d_curr``.  All three are +/-1 decisions.

        no transition (d_prev == d_curr)  ->  0, the edge carries no information
        e == d_prev                       -> +1, the crossing has not happened
                                                 yet, so the clock is early
        e == d_curr                       -> -1, the crossing already happened,
                                                 so the clock is late

    Equivalent to the usual ``XOR(d_prev, e) - XOR(e, d_curr)`` formulation.

    Assumptions
    -----------
    - The edge sampler has the same bandwidth and offset as the data sampler.
      In silicon it does not, and the mismatch appears as a static phase offset.
    - Transition density is data dependent: a long run emits zeros, so the loop
      coasts.  A PRBS supplies roughly one transition every two UI.
    """
    if d_prev == d_curr:
        return 0
    return 1 if e == d_prev else -1


def mueller_muller_pd(
    y_prev: float, y_curr: float, d_prev: int, d_curr: int
) -> float:
    """Mueller-Muller phase detector, baud-rate.

        e_MM[n] = d[n-1]*y[n] - d[n]*y[n-1]

    whose expectation over random data is (docs/dfe_notes.md section 6)

        E[e_MM] = h1 - h_-1

    so the loop settles where the first post- and pre-cursor are equal.
    Sampling earlier makes ``h1`` larger and ``h_-1`` smaller, so a positive
    output means "early", matching this module's sign convention.

    Returns the raw (unsliced) error; :func:`sign_sign` converts it to the
    bang-bang decision an actual implementation makes.

    Assumptions
    -----------
    - Decisions are correct.  A wrong decision inverts that term's contribution.
    - The lock point depends on the *equalized* pulse response, so it moves
      whenever the CTLE or DFE changes.
    """
    return d_prev * y_curr - d_curr * y_prev


def sign_sign(x: float) -> int:
    """Quantize a detector output to +/-1, with 0 mapping to 0."""
    return int(np.sign(x))


# ==========================================================================
# Phase interpolator
# ==========================================================================

@dataclass(frozen=True)
class PhaseInterpolator:
    """Finite-resolution phase interpolator.

    A real PI can only place the clock on a discrete grid, typically 32-128
    steps per UI.  The quantization has two consequences the model must keep:

    - it bounds the static phase error to +/- half a step, and
    - it sets the minimum phase movement, so a proportional gain finer than one
      step cannot actually be realized.  The default ``K_p = 1/64`` is exactly
      one step at ``n_steps = 64``, which is not a coincidence
      (docs/cdr_notes.md section 9).

    Attributes
    ----------
    n_steps : int
        Steps per UI.  64 is typical.
    """

    n_steps: int = 64

    @property
    def step_ui(self) -> float:
        """Phase resolution in UI."""
        return 1.0 / self.n_steps

    def quantize(self, phase_ui: float | np.ndarray) -> float | np.ndarray:
        """Snap a phase in UI to the interpolator grid."""
        return np.round(np.asarray(phase_ui) * self.n_steps) / self.n_steps


def sample_at(x: np.ndarray, index: float) -> float:
    """Sample a waveform at a fractional index by linear interpolation.

    The PI resolution (64 steps/UI) is finer than the waveform grid
    (``samples_per_ui``, typically 32), so the sampling instant generally falls
    between samples and interpolation is unavoidable.

    Linear interpolation is adequate because the waveform is heavily
    oversampled -- at 32 samples/UI the error is well below the PI's own
    quantization -- but it is a low-pass operation, so it slightly rounds the
    very edges an Alexander detector keys on.
    """
    if index <= 0:
        return float(x[0])
    if index >= x.size - 1:
        return float(x[-1])
    i = int(index)
    frac = index - i
    return float(x[i] * (1.0 - frac) + x[i + 1] * frac)


# ==========================================================================
# Linearized loop analysis (the reference the simulation is checked against)
# ==========================================================================

def linearized_pd_gain(sigma_tau_ui: float) -> float:
    """Describing-function gain of a bang-bang detector, per UI of phase error.

        K_pd = 2 / (sigma_tau * sqrt(2*pi))

    Obtained by differentiating ``E[b] = erf(dt/(sigma*sqrt(2)))`` at the lock
    point.  Valid only near lock and only for the jitter it was evaluated at --
    the inverse dependence on ``sigma_tau`` is the defining awkwardness of a
    bang-bang loop, not a modelling artifact.
    """
    if sigma_tau_ui <= 0:
        raise ValueError("sigma_tau must be positive: a noiseless BBPD has no gain")
    return float(2.0 / (sigma_tau_ui * np.sqrt(2.0 * np.pi)))


@dataclass(frozen=True)
class LoopParameters:
    """Second-order loop parameters at one operating point."""

    k_pd: float
    k_p: float
    k_i: float
    decimation: int
    update_rate: float
    omega_n: float
    zeta: float
    f_3db: float
    peaking_db: float
    f_zero: float
    dither_pp_ui: float
    max_slew_ui_per_s: float

    def summary(self) -> str:
        return (
            f"K_pd={self.k_pd:.2f}/UI, w_n/2pi={self.omega_n / 2 / np.pi / 1e6:.2f} MHz, "
            f"zeta={self.zeta:.2f}, f_3dB={self.f_3db / 1e6:.2f} MHz, "
            f"peaking={self.peaking_db:.2f} dB, dither={self.dither_pp_ui * 1e3:.1f} mUI"
        )


def loop_parameters(
    cfg: LinkConfig,
    k_p: float,
    k_i: float,
    decimation: int,
    sigma_tau_ui: float,
) -> LoopParameters:
    """Linearized second-order parameters (docs/cdr_notes.md sections 4-6, 8).

    With the phase updated once per ``M`` UI, the update period is
    ``T_u = M/Rs`` and

        w_n   = sqrt(K_pd*K_i) / T_u
        zeta  = (K_p/2) * sqrt(K_pd/K_i)
        w_3dB ~= 2*zeta*w_n = K_pd*K_p/T_u        for zeta >~ 1.5
        w_zero = w_n/(2*zeta) = K_i/(K_p*T_u)

    A type-II loop always peaks; the peak is

        (w_peak/w_n)^2 = (sqrt(1 + 8*zeta^2) - 1) / (4*zeta^2)

    and the dither the loop cannot avoid is ``K_p`` peak-to-peak in UI.

    Every number here is conditional on ``sigma_tau_ui`` through ``K_pd``.
    """
    k_pd = linearized_pd_gain(sigma_tau_ui)
    t_u = decimation / cfg.symbol_rate

    omega_n = np.sqrt(k_pd * k_i) / t_u
    zeta = 0.5 * k_p * np.sqrt(k_pd / k_i)

    # Exact -3 dB point: x^4 - (2 + 4*zeta^2)*x^2 - 1 = 0 with x = w/w_n.
    a = 2.0 + 4.0 * zeta**2
    x2 = 0.5 * (a + np.sqrt(a**2 + 4.0))
    f_3db = float(np.sqrt(x2) * omega_n / (2.0 * np.pi))

    r2 = (np.sqrt(1.0 + 8.0 * zeta**2) - 1.0) / (4.0 * zeta**2)
    w_pk = np.sqrt(r2) * omega_n
    peaking = float(20.0 * np.log10(abs(_jitter_transfer(w_pk, omega_n, zeta))))

    return LoopParameters(
        k_pd=k_pd, k_p=k_p, k_i=k_i, decimation=decimation,
        update_rate=1.0 / t_u,
        omega_n=float(omega_n), zeta=float(zeta), f_3db=f_3db,
        peaking_db=peaking,
        f_zero=float(omega_n / (2.0 * zeta) / (2.0 * np.pi)),
        dither_pp_ui=float(k_p),
        max_slew_ui_per_s=float(k_p / t_u),
    )


def _jitter_transfer(w: np.ndarray, omega_n: float, zeta: float) -> np.ndarray:
    s = 1j * np.asarray(w, dtype=float)
    return (2 * zeta * omega_n * s + omega_n**2) / (
        s**2 + 2 * zeta * omega_n * s + omega_n**2
    )


def jitter_transfer(f: np.ndarray, lp: LoopParameters) -> np.ndarray:
    """Recovered-clock phase over input phase.

        H(s) = (2*zeta*w_n*s + w_n^2) / (s^2 + 2*zeta*w_n*s + w_n^2)

    ``H(0) = 1``: a CDR is a tracking loop, and jitter common to data and clock
    causes no sampling error.  The numerator zero is the type-II signature and
    is why ``|H|`` exceeds unity and rolls off at -20 dB/dec rather than -40.
    """
    return _jitter_transfer(2 * np.pi * np.asarray(f), lp.omega_n, lp.zeta)


def error_transfer(f: np.ndarray, lp: LoopParameters) -> np.ndarray:
    """Sampling phase error over input phase, ``E(s) = 1 - H(s)``.

        E(s) = s^2 / (s^2 + 2*zeta*w_n*s + w_n^2)

    High-pass, so ``|E| ~ (f/f_n)^2`` below ``f_n`` -- the +40 dB/dec slope of
    the jitter-tolerance mask.
    """
    return 1.0 - jitter_transfer(f, lp)


# ==========================================================================
# The CDR
# ==========================================================================

@dataclass
class CDRResult:
    """Recovered clock phase and everything needed to interpret it."""

    phase_ui: np.ndarray            # per symbol, quantized, unwrapped
    phase_raw_ui: np.ndarray        # before PI quantization
    freq_reg: np.ndarray            # integral path state, UI per update
    pd_out: np.ndarray              # per symbol detector output
    samples: np.ndarray             # data samples actually taken
    decisions: np.ndarray           # sliced decisions, +/-1
    sample_index: np.ndarray        # fractional waveform index per symbol
    updates: int = 0
    slew_limited: bool = False
    params: LoopParameters | None = None

    @property
    def n_symbols(self) -> int:
        return self.phase_ui.size

    def steady_state(self, frac: float = 0.25) -> tuple[float, float]:
        """Mean and peak-to-peak phase over the last ``frac`` of the run, in UI."""
        n = max(int(self.n_symbols * (1.0 - frac)), 0)
        tail = self.phase_ui[n:]
        return float(tail.mean()), float(tail.max() - tail.min())

    def lock_time(self, tol_ui: float = 0.02, cfg: LinkConfig | None = None) -> float:
        """Symbols (or seconds, if ``cfg`` given) until the phase stays within
        ``tol_ui`` of its final mean."""
        final, _ = self.steady_state()
        outside = np.nonzero(np.abs(self.phase_ui - final) > tol_ui)[0]
        n = int(outside[-1] + 1) if outside.size else 0
        return n / cfg.symbol_rate if cfg is not None else float(n)

    def summary(self) -> str:
        mean, pp = self.steady_state()
        return (
            f"{self.n_symbols} symbols, {self.updates} loop updates | "
            f"lock phase {mean:+.4f} UI, dither {pp * 1e3:.1f} mUI pp"
            + ("  [SLEW LIMITED]" if self.slew_limited else "")
        )


@dataclass
class CDR:
    """Bang-bang CDR: detector, decimating loop filter, phase interpolator.

    Parameters
    ----------
    cfg : LinkConfig
        Supplies the symbol rate and the oversampling of the input waveform.
    detector : str
        ``"mm"`` (baud-rate Mueller-Muller) or ``"alexander"`` (2x oversampled).
    k_p, k_i : float
        Proportional and integral gains, in UI per update.
    decimation : int
        Detector outputs accumulated per phase update.  Without decimation the
        loop lands roughly 100x above a 16 MHz target and at the ``f_baud/10``
        stability ceiling (docs/cdr_notes.md section 9).
    pi_steps : int
        Phase interpolator resolution, steps per UI.
    vote : str
        ``"majority"`` (sign of the accumulated detector output, what silicon
        usually does) or ``"sum"`` (use the accumulation directly, more
        information per update but no longer strictly bang-bang).
    pd_tap : str
        ``"pre_dfe"`` -- the detector sees the raw sampled waveform.
        ``"post_dfe"`` -- it sees the DFE-corrected value.  This is not a
        detail: a pre-DFE MM detector settles at ``h1 == h_-1`` while a post-DFE
        one drives toward ``h_-1 == 0``, and on this channel those are far apart
        (docs/spec.md section 6, open item).
    dfe_taps : np.ndarray, optional
        Absolute postcursor weights in volts, ``c_m = h_m``.  Applied as
        ``z[n] = y[n] - sum_m c_m*d[n-m]``.
    """

    cfg: LinkConfig
    detector: str = "mm"
    k_p: float = 1.0 / 64
    k_i: float = 6.8e-6
    decimation: int = 100
    pi_steps: int = 64
    vote: str = "majority"
    pd_tap: str = "pre_dfe"
    dfe_taps: np.ndarray | None = None
    pi: PhaseInterpolator = field(init=False)

    def __post_init__(self) -> None:
        if self.detector not in ("mm", "alexander"):
            raise ValueError(f"unknown detector {self.detector!r}")
        if self.vote not in ("majority", "sum"):
            raise ValueError(f"unknown vote {self.vote!r}")
        if self.pd_tap not in ("pre_dfe", "post_dfe"):
            raise ValueError(f"unknown pd_tap {self.pd_tap!r}")
        object.__setattr__(self, "pi", PhaseInterpolator(self.pi_steps))

    @classmethod
    def from_config(cls, cfg: LinkConfig, cdr_cfg: CDRConfig | None = None, **kw):
        """Build from the bounded parameters in :class:`~src.config.CDRConfig`."""
        c = cfg.cdr if cdr_cfg is None else cdr_cfg
        return cls(
            cfg=cfg,
            k_p=float(c.k_p.value),
            k_i=float(c.k_i.value),
            decimation=int(c.decimation.value),
            **kw,
        )

    def parameters(self, sigma_tau_ui: float = 0.02) -> LoopParameters:
        """Linearized loop parameters at a given rms input jitter."""
        return loop_parameters(self.cfg, self.k_p, self.k_i,
                               self.decimation, sigma_tau_ui)

    # ----------------------------------------------------------------------

    def run(
        self,
        waveform: np.ndarray,
        n_symbols: int | None = None,
        initial_phase_ui: float = 0.0,
        disturbance_ui: np.ndarray | None = None,
        sigma_tau_ui: float = 0.02,
    ) -> CDRResult:
        """Track the clock phase across a waveform.

        The loop runs symbol by symbol:

            1. sample the waveform at ``n*UI + phase`` (fractional, interpolated)
            2. slice, optionally after DFE subtraction
            3. run the phase detector
            4. accumulate; every ``decimation`` symbols update

                   freq  += K_i * b
                   phase += K_p * b + freq

            5. quantize the phase to the interpolator grid

        Step 4 is the type-II structure: two accumulators, hence ``s^2`` in the
        open-loop denominator and zero steady-state error to a frequency offset.

        Parameters
        ----------
        waveform : np.ndarray
            Received waveform on the ``cfg.samples_per_ui`` grid, already
            equalized if a CTLE is in use.
        n_symbols : int, optional
            Defaults to as many whole symbols as the waveform holds, less a
            one-UI guard for the edge sample.
        initial_phase_ui : float
            Starting phase offset, for measuring acquisition.
        disturbance_ui : np.ndarray, optional
            Per-symbol phase disturbance added to the ideal sampling instant,
            i.e. input jitter.  Drive it with a sinusoid to measure jitter
            transfer.
        sigma_tau_ui : float
            Only used to attach :class:`LoopParameters` to the result; it does
            not affect the simulation, which is genuinely nonlinear.

        Returns
        -------
        CDRResult
        """
        x = np.asarray(waveform, dtype=float)
        m = self.cfg.samples_per_ui
        max_sym = x.size // m - 2
        n = max_sym if n_symbols is None else min(n_symbols, max_sym)
        if n < 4:
            raise ValueError("waveform too short for a CDR run")

        dist = (np.zeros(n) if disturbance_ui is None
                else np.asarray(disturbance_ui, dtype=float)[:n])
        taps = None if self.dfe_taps is None else np.asarray(self.dfe_taps, float)

        phase = float(initial_phase_ui)
        freq = 0.0
        accum = 0.0
        slew_hit = False

        ph_q = np.zeros(n)
        ph_raw = np.zeros(n)
        fr = np.zeros(n)
        pd = np.zeros(n)
        samp = np.zeros(n)
        corr = np.zeros(n)
        idx = np.zeros(n)
        dec = np.zeros(n, dtype=int)
        updates = 0

        for k in range(n):
            phase_q = float(self.pi.quantize(phase))
            index = (k + phase_q + dist[k]) * m

            y = sample_at(x, index)
            z = y
            if taps is not None and k > 0:
                # dec[k-back:k][::-1] is [d[k-1], d[k-2], ...], so taps[0] pairs
                # with the first postcursor.  A stride slice of the form
                # dec[k-1 : k-1-back : -1] silently yields an empty array when
                # k-1-back reaches -1, dropping the correction near the start.
                back = min(taps.size, k)
                z = y - float(np.dot(taps[:back], dec[k - back : k][::-1]))
            d = 1 if z >= 0.0 else -1

            samp[k], idx[k], dec[k], corr[k] = y, index, d, z
            ph_q[k], ph_raw[k], fr[k] = phase_q, phase, freq

            # -- detector ---------------------------------------------------
            # Both samples the MM detector uses must come from the same tap
            # point; mixing a post-DFE current sample with a pre-DFE previous
            # one would fabricate a phase error out of the tap difference.
            series = corr if self.pd_tap == "post_dfe" else samp
            if k == 0:
                b = 0.0
            elif self.detector == "mm":
                b = float(sign_sign(
                    mueller_muller_pd(series[k - 1], series[k], dec[k - 1], d)
                ))
            else:
                e_index = (k + phase_q + dist[k] - 0.5) * m
                e = 1 if sample_at(x, e_index) >= 0.0 else -1
                b = float(alexander_pd(dec[k - 1], d, e))
            pd[k] = b
            accum += b

            # -- decimated loop filter --------------------------------------
            if (k + 1) % self.decimation == 0:
                vote = float(np.sign(accum)) if self.vote == "majority" else accum
                freq += self.k_i * vote
                step = self.k_p * vote + freq
                if abs(self.k_p * vote) > self.k_p + 1e-15:
                    slew_hit = True
                phase += step
                accum = 0.0
                updates += 1

        return CDRResult(
            phase_ui=ph_q, phase_raw_ui=ph_raw, freq_reg=fr, pd_out=pd,
            samples=samp, decisions=dec, sample_index=idx,
            updates=updates, slew_limited=slew_hit,
            params=self.parameters(sigma_tau_ui),
        )


# ==========================================================================
# Self-test
# ==========================================================================

def self_test(verbose: bool = True) -> bool:
    """Verify the detectors, the interpolator, and the loop's closed-loop behaviour."""
    from .channel import mixed_mode, load, impulse_response, pulse_response, split_pulse
    from . import configs

    cfg = LinkConfig()
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<42s} {detail}")

    if verbose:
        print("Alexander PD truth table")
    report("no transition -> 0",
           alexander_pd(1, 1, 1) == 0 and alexander_pd(-1, -1, -1) == 0)
    report("edge still at old value -> early (+1)",
           alexander_pd(-1, 1, -1) == +1 and alexander_pd(1, -1, 1) == +1)
    report("edge already at new value -> late (-1)",
           alexander_pd(-1, 1, 1) == -1 and alexander_pd(1, -1, -1) == -1)

    if verbose:
        print("\nphase interpolator")
    pi = PhaseInterpolator(64)
    report("step is 1/64 UI", abs(pi.step_ui - 1 / 64) < 1e-15,
           f"{pi.step_ui * 1e3:.4f} mUI")
    report("quantization error <= half a step",
           float(np.abs(pi.quantize(np.linspace(-1, 1, 9999))
                        - np.linspace(-1, 1, 9999)).max()) <= 0.5 * pi.step_ui + 1e-12)
    report("K_p = 1/64 is exactly one PI step", abs(1 / 64 - pi.step_ui) < 1e-15)

    if verbose:
        print("\nlinearized loop (docs/cdr_notes.md sections 2, 6, 9)")
    report("K_pd = 2/(sigma*sqrt(2pi))",
           abs(linearized_pd_gain(0.02) - 39.894) < 0.01,
           f"{linearized_pd_gain(0.02):.3f} /UI at sigma=0.02 UI")
    report("K_pd doubles when sigma halves",
           abs(linearized_pd_gain(0.01) / linearized_pd_gain(0.02) - 2.0) < 1e-12)
    lp = loop_parameters(cfg, 1 / 64, 6.8e-6, 100, 0.02)
    report("overdamped by design", lp.zeta > 1.5, f"zeta = {lp.zeta:.2f}")
    report("dither == K_p", abs(lp.dither_pp_ui - 1 / 64) < 1e-15,
           f"{lp.dither_pp_ui * 1e3:.2f} mUI pp")
    report("H(0) == 1 (tracking loop)",
           abs(jitter_transfer(np.array([1.0]), lp)[0] - 1.0) < 1e-6)
    report("type-II loop always peaks", lp.peaking_db > 0,
           f"{lp.peaking_db:.3f} dB")
    f_hi = np.array([100 * lp.f_3db])
    report("E(f) -> 1 well above the loop bandwidth",
           abs(abs(error_transfer(f_hi, lp)[0]) - 1.0) < 0.01)

    if verbose:
        print("\nclosed loop on the real channel")
    ccfg = configs.get("baseline")
    mm = mixed_mode(load(ccfg.channel_path))
    _, h, _ = impulse_response(mm["f"], mm["sdd21"], ccfg)
    p = pulse_response(h, ccfg)
    sp = split_pulse(p, ccfg)
    from .channel import mm_lock_phase
    _, expect = mm_lock_phase(sp)

    from . import tx
    wave = tx.generate(ccfg, 4000, prbs_order=15, rise_time=0.0,
                       edge_method="ideal", swing=1.0)
    rx = np.convolve(h, wave.waveform)[: wave.waveform.size]

    # The CDR's phase is absolute and modulo 1 UI, so it absorbs the channel's
    # 78.59 UI of propagation delay; split_pulse measures relative to the pulse
    # peak.  Comparing them requires putting both in the same frame.
    def wrap(v):
        return (v + 0.5) % 1.0 - 0.5

    peak_phase = wrap(sp.k_peak / ccfg.samples_per_ui)
    expect_mm = wrap(peak_phase + expect)

    cdr = CDR(ccfg, detector="mm", decimation=20)
    res = cdr.run(rx, initial_phase_ui=0.0)
    got, dither = res.steady_state()
    err = abs(wrap(got - expect_mm))
    report("MM settles at the predicted lock phase",
           err < 2 * cdr.pi.step_ui,
           f"{got:+.4f} UI vs {expect_mm:+.4f} predicted "
           f"({err * 64:.2f} PI steps)")
    report("dither is bounded",
           dither <= 8 * cdr.pi.step_ui, f"{dither * 1e3:.1f} mUI pp")
    report("loop updated as expected",
           abs(res.updates - 4000 // 20) <= 1, f"{res.updates} updates")

    a = CDR(ccfg, detector="alexander", decimation=20).run(rx)
    ga, _ = a.steady_state()
    err_a = abs(wrap(ga - peak_phase))
    report("Alexander samples at the pulse centre",
           err_a < 2 * cdr.pi.step_ui,
           f"{ga:+.4f} UI vs {peak_phase:+.4f} predicted "
           f"({err_a * 64:.2f} PI steps)")
    report("MM lock sits later than Alexander's",
           wrap(got - ga) > 0.1,
           f"MM is {wrap(got - ga):+.4f} UI later -- the eye-closing offset")

    if verbose:
        print(f"\n{lp.summary()}")
        print(f"{'all checks passed' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if self_test() else 1)
