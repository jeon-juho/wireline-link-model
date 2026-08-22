"""CTLE: continuous-time linear equalizer, one zero and two poles.

    H(s) = A_dc * (1 + s/w_z) / ((1 + s/w_p1) * (1 + s/w_p2))

The block is specified the way a designer thinks about it -- DC gain, boost,
where the peak sits, and where the output node rolls off -- and the pole/zero
positions are solved for.  Background derivations are in docs/ctle_notes.md;
this module implements them and inverts them.

All quantities are SI: frequency in Hz, angular frequency in rad/s, time in s.

Frequency warping
-----------------
The bilinear transform compresses the frequency axis,

    w_a = (2/T)*tan(w_d*T/2)   ->   f_apparent/f_design ~= 1 - (1/3)*(pi*f/fs)^2

so a singularity lands *below* where it was placed, and the error grows as f^2.
At 32 samples/UI (fs = 512 GHz) that is -0.24 % for a 13.9 GHz second pole and
-0.03 % for a 4.6 GHz first pole; at 8 samples/UI it is -3.7 % and -0.4 %.

Two derived quantities inherit this differently, which is why the effect is
worth correcting rather than absorbing:

- the peak sits near ``sqrt(f_p1*f_p2)`` and so takes the *average* pole error;
- the boost is set by the *ratio* ``f_p1/f_z``, and because compression is
  unequal the ratio itself changes -- warping is not a pure frequency shift.

:func:`discretize` therefore prewarps each singularity by default.  Note that
``scipy.signal.bilinear`` performs no prewarping of its own.  Prewarping pins
the singularities exactly but not the shape between them, so the peak still
lands slightly off; :func:`verify_discretization` measures that residual rather
than assuming it is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal

from .config import CTLEConfig, LinkConfig

# ==========================================================================
# Forward relations: pole/zero -> peak location and gain
# ==========================================================================

def peak_frequency(f_z: float, f_p1: float, f_p2: float) -> float:
    """Frequency of maximum gain, in Hz.

    Maximizing |H(jw)|^2 over w gives (docs/ctle_notes.md section 5)

        w_peak = sqrt( sqrt((w_p1^2 - w_z^2)*(w_p2^2 - w_z^2)) - w_z^2 )

    The expression is homogeneous in frequency, so it may be evaluated on Hz
    directly without converting to rad/s.

    The common approximation ``f_peak ~= sqrt(f_p1*f_p2)`` -- the geometric mean
    of the poles -- holds when ``f_z << f_p1``.  The zero sets the *height* of
    the peak, not its location.
    """
    z2, p12, p22 = f_z**2, f_p1**2, f_p2**2
    inner = (p12 - z2) * (p22 - z2)
    if inner <= 0:
        raise ValueError("no peak exists: the zero is not below both poles")
    val = np.sqrt(inner) - z2
    if val <= 0:
        raise ValueError("no peak exists: response is monotonic")
    return float(np.sqrt(val))


def peak_gain_db(f_z: float, f_p1: float, f_p2: float) -> float:
    """Gain at the peak relative to DC, in dB -- the achieved boost.

        |H(w_peak)|        sqrt(1 + w_peak^2/w_z^2)
        -----------  =  ------------------------------------------------
            A_dc        sqrt(1 + w_peak^2/w_p1^2) * sqrt(1 + w_peak^2/w_p2^2)

    This is the *exact* boost.  The asymptotic estimate
    ``20*log10(f_p1/f_z)`` assumes the mid-band plateau at ``gm*R_D`` is
    actually reached, which needs ``f_p2 >> f_p1``.  With a realistic
    ``f_p2 = 3*f_p1`` the roll-off eats about 2.5 dB of the peak, so designing
    to the asymptotic formula lands the measured peaking short of target
    (docs/ctle_notes.md section 6).
    """
    u = peak_frequency(f_z, f_p1, f_p2) ** 2
    num = np.sqrt(1.0 + u / f_z**2)
    den = np.sqrt(1.0 + u / f_p1**2) * np.sqrt(1.0 + u / f_p2**2)
    return float(20.0 * np.log10(num / den))


def asymptotic_boost_db(f_z: float, f_p1: float) -> float:
    """Asymptotic boost ``20*log10(f_p1/f_z)``, for comparison only.

    Equals ``20*log10(1 + gm*R_S/2)``, i.e. exactly the DC gain that
    degeneration throws away.  Always optimistic relative to
    :func:`peak_gain_db`.
    """
    return float(20.0 * np.log10(f_p1 / f_z))


# ==========================================================================
# Inversion: peak location and boost -> pole/zero
# ==========================================================================

def _f_p1_from(f_z: float, f_peak: float, f_p2: float) -> float:
    """First pole placing the peak at ``f_peak``, given ``f_z`` and ``f_p2``.

    Rearranging the peak-location expression of :func:`peak_frequency`:

        f_peak^2 + f_z^2 = sqrt((f_p1^2 - f_z^2)*(f_p2^2 - f_z^2))

    square both sides and solve for ``f_p1``:

        f_p1^2 = f_z^2 + (f_peak^2 + f_z^2)^2 / (f_p2^2 - f_z^2)

    Closed form, which is what reduces the two-unknown design problem to a
    one-dimensional root find.
    """
    if f_z >= f_p2:
        raise ValueError("zero must lie below the second pole")
    return float(np.sqrt(f_z**2 + (f_peak**2 + f_z**2) ** 2 / (f_p2**2 - f_z**2)))


def solve_zero_pole(
    boost_db: float, peak_freq_hz: float, second_pole_hz: float
) -> tuple[float, float]:
    """Solve for ``(f_z, f_p1)`` giving the requested boost and peak location.

    Two unknowns and two conditions -- peak location and peak gain -- but the
    peak-location condition is invertible in closed form
    (:func:`_f_p1_from`), so this reduces to a single root find over ``f_z``:

        g(f_z) = peak_gain_db(f_z, f_p1(f_z), f_p2) - boost_db = 0

    ``g`` is monotonically decreasing: pushing the zero up reduces the boost.
    The limits bracket the achievable range --

        f_z -> 0      boost -> infinity  (unbounded on paper; in silicon it
                                          costs DC gain, which is the real cap)
        f_z -> f_p2   boost -> 0 dB      (zero cancels the pole, no peaking)

    -- so a solution exists for any positive ``boost_db``, and Brent's method
    finds it robustly.

    Assumptions
    -----------
    - ``f_p2`` is given, not solved for.  It is set by the output node,
      ``f_p2 = 1/(2*pi*R_D*C_L)``, and is a consequence of the load rather than
      a free design variable (docs/ctle_notes.md section 3).
    - The result is a small-signal, single-stage response.  Cascading N stages
      multiplies the dB boost but also multiplies the roll-off.

    Parameters
    ----------
    boost_db : float
        Exact peak gain relative to DC, in dB.  Positive.
    peak_freq_hz : float
        Where the peak should sit.  Usually the Nyquist frequency.
    second_pole_hz : float
        Output-node pole.

    Returns
    -------
    (f_z, f_p1) : tuple of float
        In Hz.
    """
    if boost_db <= 0:
        raise ValueError(f"boost_db must be positive, got {boost_db}")
    if not 0 < peak_freq_hz:
        raise ValueError("peak_freq_hz must be positive")

    def g(f_z: float) -> float:
        return peak_gain_db(f_z, _f_p1_from(f_z, peak_freq_hz, second_pole_hz),
                            second_pole_hz) - boost_db

    lo, hi = 1e-6 * peak_freq_hz, 0.999 * second_pole_hz
    g_lo, g_hi = g(lo), g(hi)
    if g_lo < 0 or g_hi > 0:
        raise ValueError(
            f"boost {boost_db:.2f} dB unreachable with f_peak="
            f"{peak_freq_hz / 1e9:.2f} GHz, f_p2={second_pole_hz / 1e9:.2f} GHz; "
            f"achievable range is ({g_hi + boost_db:.2f}, {g_lo + boost_db:.2f}) dB"
        )

    f_z = float(optimize.brentq(g, lo, hi, xtol=1e-6, rtol=1e-14))
    return f_z, _f_p1_from(f_z, peak_freq_hz, second_pole_hz)


# ==========================================================================
# The CTLE
# ==========================================================================

@dataclass(frozen=True)
class CTLE:
    """A one-zero, two-pole CTLE stage.

    Construct either from singularities directly, or from a design intent with
    :meth:`from_spec`.
    """

    f_z: float
    f_p1: float
    f_p2: float
    dc_gain_db: float = 0.0
    n_stages: int = 1

    @classmethod
    def from_spec(
        cls,
        boost_db: float,
        peak_freq_hz: float,
        second_pole_hz: float,
        dc_gain_db: float = 0.0,
        n_stages: int = 1,
    ) -> "CTLE":
        """Build from design intent, solving for the singularities.

        ``dc_gain_db`` defaults to 0 dB, i.e. the stage is treated as pure
        shaping with unity DC gain.  A real degenerated pair has
        ``A_dc = gm*R_D/(1 + gm*R_S/2)``, and the boost is exactly the DC gain
        that degeneration discarded -- so a physical stage has negative
        ``dc_gain_db``.  Separating the two lets the model answer "what does
        the shape do" independently of "how much amplitude did it cost".
        """
        f_z, f_p1 = solve_zero_pole(boost_db, peak_freq_hz, second_pole_hz)
        return cls(f_z, f_p1, second_pole_hz, dc_gain_db, n_stages)

    @classmethod
    def from_config(cls, cfg: CTLEConfig) -> "CTLE":
        """Build from the bounded parameters in :class:`~src.config.CTLEConfig`."""
        return cls(
            f_z=float(cfg.f_z.value),
            f_p1=float(cfg.f_p1.value),
            f_p2=float(cfg.f_p2.value),
            n_stages=int(cfg.n_stages.value),
        )

    # -- derived quantities ------------------------------------------------

    @property
    def a_dc(self) -> float:
        """Linear DC gain of one stage."""
        return float(10.0 ** (self.dc_gain_db / 20.0))

    @property
    def peak_freq(self) -> float:
        """Peak frequency in Hz (identical for one stage or a cascade)."""
        return peak_frequency(self.f_z, self.f_p1, self.f_p2)

    @property
    def boost_db(self) -> float:
        """Achieved boost in dB, for the full cascade of ``n_stages``."""
        return self.n_stages * peak_gain_db(self.f_z, self.f_p1, self.f_p2)

    @property
    def asymptotic_boost_db(self) -> float:
        """Asymptotic boost of the cascade, always optimistic."""
        return self.n_stages * asymptotic_boost_db(self.f_z, self.f_p1)

    def summary(self) -> str:
        return (
            f"CTLE {self.n_stages}x: f_z={self.f_z / 1e9:.3f} GHz, "
            f"f_p1={self.f_p1 / 1e9:.3f} GHz, f_p2={self.f_p2 / 1e9:.3f} GHz, "
            f"peak {self.peak_freq / 1e9:.3f} GHz, boost {self.boost_db:.2f} dB "
            f"(asymptotic {self.asymptotic_boost_db:.2f} dB), "
            f"A_dc {self.dc_gain_db:.1f} dB"
        )

    # -- continuous time ---------------------------------------------------

    def analog_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerator and denominator polynomials in s, one stage.

            num = A_dc * [1/w_z, 1]
            den = [1/w_p1, 1] * [1/w_p2, 1]

        Coefficients are in rad/s, highest power of s first, as
        ``scipy.signal.freqs`` and ``bilinear`` expect.
        """
        w_z, w_p1, w_p2 = (2 * np.pi * f for f in (self.f_z, self.f_p1, self.f_p2))
        num = self.a_dc * np.array([1.0 / w_z, 1.0])
        den = np.polymul([1.0 / w_p1, 1.0], [1.0 / w_p2, 1.0])
        return num, np.asarray(den)

    def response(self, f: np.ndarray) -> np.ndarray:
        """Continuous-time frequency response H(j*2*pi*f), cascade included."""
        num, den = self.analog_coeffs()
        _, h = signal.freqs(num, den, worN=2 * np.pi * np.asarray(f, dtype=float))
        return h**self.n_stages

    def bode(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(magnitude_db, phase_deg)`` of the continuous-time response."""
        h = self.response(f)
        return 20.0 * np.log10(np.abs(h) + 1e-30), np.degrees(np.unwrap(np.angle(h)))

    # -- discrete time -----------------------------------------------------

    def discretize(
        self, cfg: LinkConfig, prewarp: bool = True, match_peak: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear-transform one stage to a digital filter at ``cfg.fs``.

        With ``prewarp=True`` each singularity is pre-distorted,

            w_analog = 2*fs * tan(w_target / (2*fs))

        so the transform's compression lands it exactly on ``w_target``.
        ``scipy.signal.bilinear`` does not do this itself.

        With ``match_peak=True`` the singularities are additionally scaled by a
        common factor (:func:`_peak_match_scale`) so the *peak* lands exactly on
        ``self.peak_freq``.  Prewarping alone cannot achieve this: it fixes the
        singularities but not the shape between them, leaving the peak ~0.06 %
        off at 32 samples/UI.  Because a uniform scale preserves every
        pole/zero ratio, it moves the peak without touching the boost.

        Use ``match_peak`` when the peak location is the specification -- which
        it is, given this class is parameterized by ``peak_freq_hz``.  It is off
        by default so the singularities stay exactly where a designer placed
        them.

        Returns ``(b, a)`` for a single stage; :meth:`apply` handles cascading.
        """
        trio = (self.f_z, self.f_p1, self.f_p2)
        if match_peak:
            if not prewarp:
                raise ValueError("match_peak requires prewarp=True")
            k = _peak_match_scale(self, cfg.fs)
            trio = tuple(k * f for f in trio)
        if prewarp:
            trio = tuple(prewarp_frequency(f, cfg.fs) for f in trio)

        stage = CTLE(*trio, dc_gain_db=self.dc_gain_db, n_stages=1)
        num, den = stage.analog_coeffs()
        return signal.bilinear(num, den, fs=cfg.fs)

    def digital_response(
        self, f: np.ndarray, cfg: LinkConfig, prewarp: bool = True,
        match_peak: bool = False,
    ) -> np.ndarray:
        """Frequency response of the discretized filter, cascade included."""
        b, a = self.discretize(cfg, prewarp=prewarp, match_peak=match_peak)
        _, h = signal.freqz(b, a, worN=2 * np.pi * np.asarray(f) / cfg.fs)
        return h**self.n_stages

    def apply(
        self, x: np.ndarray, cfg: LinkConfig, prewarp: bool = True,
        match_peak: bool = False,
    ) -> np.ndarray:
        """Filter a time-domain waveform through the CTLE.

        Applied ``n_stages`` times in sequence, which is equivalent to one
        filter of the cascaded response but numerically better conditioned than
        expanding the polynomial to order 2N.
        """
        b, a = self.discretize(cfg, prewarp=prewarp, match_peak=match_peak)
        y = np.asarray(x, dtype=float)
        for _ in range(self.n_stages):
            y = signal.lfilter(b, a, y)
        return y


# ==========================================================================
# Warping helpers and verification
# ==========================================================================

def prewarp_frequency(f: float, fs: float) -> float:
    """Analog frequency that lands at ``f`` after the bilinear transform.

        f_analog = (fs/pi) * tan(pi*f/fs)

    which is ``w_a = (2/T)*tan(w_d*T/2)`` written in Hz.  Diverges as ``f``
    approaches ``fs/2``, so a singularity near Nyquist cannot be represented.
    """
    if not 0 < f < 0.5 * fs:
        raise ValueError(f"f={f:.3e} must lie strictly inside (0, fs/2={0.5 * fs:.3e})")
    return float(fs / np.pi * np.tan(np.pi * f / fs))


def unwarp_frequency(f: float, fs: float) -> float:
    """Digital frequency an analog singularity at ``f`` lands on.

    Inverse of :func:`prewarp_frequency`:  ``f_digital = (fs/pi)*arctan(pi*f/fs)``.
    """
    return float(fs / np.pi * np.arctan(np.pi * f / fs))


def predicted_digital_peak(
    f_z: float, f_p1: float, f_p2: float, fs: float, prewarp: bool = True
) -> float:
    """Where the discretized filter peaks, computed rather than searched.

    The bilinear transform satisfies ``H_d(f) = H_a(prewarp(f))`` exactly, so
    the digital peak is the unwarped image of the analog prototype's peak:

        f_peak_digital = unwarp( peak_frequency(prototype singularities) )

    With ``prewarp=True`` the prototype singularities are themselves prewarped.
    That pins each singularity on target but leaves the *shape* between them
    distorted, and the peak is a property of the shape -- which is why
    prewarping reduces the peak error without eliminating it, and flips its
    sign (-0.080 % becomes +0.056 % for the default design at 512 GHz).
    """
    trio = (f_z, f_p1, f_p2)
    if prewarp:
        trio = tuple(prewarp_frequency(f, fs) for f in trio)
    return unwarp_frequency(peak_frequency(*trio), fs)


def _peak_match_scale(ctle: "CTLE", fs: float) -> float:
    """Uniform scale on the singularities that lands the digital peak on target.

    Scaling ``f_z``, ``f_p1`` and ``f_p2`` by a common factor leaves every
    ratio -- and therefore the boost -- unchanged, while moving the peak
    proportionally.  That makes the residual peak error correctable by a single
    scalar without disturbing the gain the stage was designed for.

    Solves ``predicted_digital_peak(k*f_z, k*f_p1, k*f_p2) == ctle.peak_freq``.
    """
    target = ctle.peak_freq

    def err(k: float) -> float:
        return predicted_digital_peak(
            k * ctle.f_z, k * ctle.f_p1, k * ctle.f_p2, fs, prewarp=True
        ) - target

    return float(optimize.brentq(err, 0.9, 1.1, xtol=1e-14, rtol=1e-15))


def refine_peak(f: np.ndarray, mag: np.ndarray) -> tuple[float, float]:
    """Locate a spectral peak to sub-bin accuracy.

    Fits a parabola to ``log|H|`` through the three samples around the grid
    maximum, which is exact for a Gaussian-shaped peak and accurate to second
    order for any smooth one:

        delta = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)     in bins
        f_peak = f[i] + delta*df
        log|H_peak| = y1 - 0.25*(y0 - y2)*delta

    Refinement is not optional here.  A CTLE peak is broad, so the grid maximum
    can sit a full bin away from the true one; at 8 GHz an FFT bin of 1.95 MHz
    is 0.024 %, which is the same size as the discretization residual being
    measured.  Without refinement a correct filter reads as a wrong one.

    Requires ``f`` to be uniformly spaced.

    Returns
    -------
    (f_peak, mag_peak)
    """
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    i = int(np.argmax(mag))
    if not 0 < i < mag.size - 1:
        return float(f[i]), float(mag[i])

    y0, y1, y2 = np.log(mag[i - 1 : i + 2])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(f[i]), float(mag[i])

    delta = 0.5 * (y0 - y2) / denom
    return (
        float(f[i] + delta * (f[1] - f[0])),
        float(np.exp(y1 - 0.25 * (y0 - y2) * delta)),
    )


def warping_error(f: float, fs: float) -> float:
    """Relative shift a singularity at ``f`` suffers with no prewarping.

        f_apparent/f_design = arctan(pi*f/fs) / (pi*f/fs)  ~=  1 - (pi*f/fs)^2/3

    Returned as a signed fraction, so -0.0024 means the feature lands 0.24 %
    low.  The f^2 scaling is why the second pole moves most.
    """
    x = np.pi * f / fs
    return float(np.arctan(x) / x - 1.0)


def verify_discretization(
    ctle: CTLE, cfg: LinkConfig, prewarp: bool = True, match_peak: bool = False
) -> dict[str, float]:
    """Measure where the discretized filter's features actually landed.

    Prewarping pins the singularities but not the shape between them, so the
    peak -- a property of the shape -- can still drift.  This measures the
    drift instead of assuming it away.

    Returns a dict of the analog design targets, the measured digital peak
    frequency and boost, and the resulting errors.
    """
    f = np.linspace(1e6, 0.45 * cfg.fs, 400_000)
    kw = {"prewarp": prewarp, "match_peak": match_peak}
    h = np.abs(ctle.digital_response(f, cfg, **kw))
    dc = np.abs(ctle.digital_response(np.array([1e3]), cfg, **kw))[0]

    f_peak_d, peak_mag = refine_peak(f, h)
    boost_d = float(20.0 * np.log10(peak_mag / dc))

    return {
        "f_peak_analog": ctle.peak_freq,
        "f_peak_digital": f_peak_d,
        "f_peak_error": f_peak_d / ctle.peak_freq - 1.0,
        "boost_analog_db": ctle.boost_db,
        "boost_digital_db": boost_d,
        "boost_error_db": boost_d - ctle.boost_db,
        "dc_gain_error": dc / (ctle.a_dc**ctle.n_stages) - 1.0,
    }


# ==========================================================================
# Self-test
# ==========================================================================

def self_test(cfg: LinkConfig | None = None, verbose: bool = True) -> bool:
    """Verify the forward relations, the inversion, and the discretization."""
    cfg = LinkConfig() if cfg is None else cfg
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<40s} {detail}")

    if verbose:
        print("forward relations vs. docs/ctle_notes.md section 7.1")
    f_z, f_p1, f_p2 = 0.77e9, 4.62e9, 13.9e9
    fp = peak_frequency(f_z, f_p1, f_p2) / 1e9
    pg = peak_gain_db(f_z, f_p1, f_p2)
    report("peak frequency == 7.90 GHz", abs(fp - 7.90) < 0.03, f"{fp:.3f} GHz")
    report("peak gain == 13.1 dB", abs(pg - 13.1) < 0.05, f"{pg:.3f} dB")
    report("asymptotic overstates the boost",
           asymptotic_boost_db(f_z, f_p1) > pg,
           f"{asymptotic_boost_db(f_z, f_p1):.2f} dB vs {pg:.2f} dB exact "
           f"(deficit {asymptotic_boost_db(f_z, f_p1) - pg:.2f} dB)")

    if verbose:
        print("\ninversion round-trip")
    for boost, f_peak in ((13.19, 8e9), (6.0, 8e9), (20.0, 8e9), (13.19, 6e9)):
        c = CTLE.from_spec(boost, f_peak, 13.9e9)
        report(f"boost {boost:.2f} dB @ {f_peak / 1e9:.0f} GHz recovered",
               abs(c.boost_db - boost) < 1e-6
               and abs(c.peak_freq / f_peak - 1) < 1e-9,
               f"f_z={c.f_z / 1e9:.3f} f_p1={c.f_p1 / 1e9:.3f} GHz")

    try:
        CTLE.from_spec(-1.0, 8e9, 13.9e9)
        report("negative boost rejected", False)
    except ValueError:
        report("negative boost rejected", True)

    if verbose:
        print("\ndiscretization (fs = %.0f GHz, %d samples/UI)"
              % (cfg.fs / 1e9, cfg.samples_per_ui))
    c = CTLE.from_spec(13.19, 8e9, 13.9e9)
    modes = ((False, False, 5e-3, "no prewarp"),
             (True, False, 1e-3, "prewarp"),
             (True, True, 1e-5, "prewarp + match_peak"))
    for pw, mp, tol, tag in modes:
        v = verify_discretization(c, cfg, prewarp=pw, match_peak=mp)
        report(f"{tag}: peak error < {tol * 100:g}%",
               abs(v["f_peak_error"]) < tol,
               f"{v['f_peak_digital'] / 1e9:.5f} GHz "
               f"({v['f_peak_error'] * 100:+.4f}%)")
        report(f"{tag}: boost within 0.01 dB", abs(v["boost_error_db"]) < 0.01,
               f"{v['boost_digital_db']:.4f} dB "
               f"({v['boost_error_db']:+.4f} dB)")
        # The analytic prediction must agree with the measured peak.
        pred = predicted_digital_peak(c.f_z, c.f_p1, c.f_p2, cfg.fs, prewarp=pw)
        if not mp:
            report(f"{tag}: analytic prediction matches",
                   abs(pred / v["f_peak_digital"] - 1) < 2e-4,
                   f"predicted {pred / 1e9:.5f} GHz")
    report("digital DC gain matches A_dc",
           abs(verify_discretization(c, cfg)["dc_gain_error"]) < 1e-9)

    # Warping estimate should agree with the exact relation.
    for f in (4.62e9, 13.9e9):
        exact = warping_error(f, cfg.fs)
        approx = -(np.pi * f / cfg.fs) ** 2 / 3.0
        report(f"warping estimate at {f / 1e9:.2f} GHz",
               abs(exact - approx) < 0.05 * abs(exact),
               f"exact {exact * 100:+.4f}%, 1-x^2/3 gives {approx * 100:+.4f}%")

    if verbose:
        print("\ntime domain")
    dc_step = np.ones(4096)
    y = c.apply(dc_step, cfg)
    report("step settles to A_dc", abs(y[-1] - c.a_dc) < 1e-6, f"{y[-1]:.9f}")
    report("no NaN or overflow", np.all(np.isfinite(y)))

    if verbose:
        print(f"\n{c.summary()}")
        print(f"\n{'all checks passed' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if self_test() else 1)
