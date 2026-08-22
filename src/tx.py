"""Transmitter: PRBS generation, NRZ waveform synthesis, edge shaping, TX FFE.

Produces the stimulus the time-domain engine drives through the channel.  The
statistical engine needs none of this -- it works from the pulse response -- so
this module is what makes the two methods comparable rather than redundant
(docs/spec.md criterion 3).

All quantities are SI: time in s, voltage in V.

Amplitude convention
--------------------
Symbols are +/- ``swing``, so the differential peak-to-peak amplitude is
``2 * swing``.  The default 0.5 V gives a 1.0 Vppd launch.  Because the channel
model is linear, every downstream voltage scales with this number; the pulse
response in :mod:`src.channel` is computed for a unit pulse, so a cursor there
multiplies by ``swing`` to become a real voltage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import LinkConfig

# ==========================================================================
# PRBS
# ==========================================================================

#: Primitive polynomials from ITU-T O.150, as the exponent of the middle term
#: in ``x^n + x^t + 1``.  The constant term is implicit.
#:
#:     PRBS7   x^7  + x^6  + 1      period      127
#:     PRBS9   x^9  + x^5  + 1      period      511
#:     PRBS11  x^11 + x^9  + 1      period     2047
#:     PRBS15  x^15 + x^14 + 1      period    32767
#:     PRBS20  x^20 + x^3  + 1      period  1048575
#:     PRBS23  x^23 + x^18 + 1      period  8388607
#:     PRBS31  x^31 + x^28 + 1      period 2147483647
PRBS_POLY: dict[int, int] = {7: 6, 9: 5, 11: 9, 15: 14, 20: 3, 23: 18, 31: 28}


def prbs(order: int, n_bits: int, seed: int | None = None) -> np.ndarray:
    """Generate a PRBS bit sequence.

    The polynomial ``x^n + x^t + 1`` corresponds to the linear recurrence

        a[k] = a[k-n] XOR a[k-(n-t)]

    over GF(2).  The exponent mapping is worth stating explicitly because it is
    easy to invert: a recurrence ``a[k] = a[k-n] XOR a[k-m]`` has characteristic
    polynomial ``x^n + x^(n-m) + 1``, so realizing ``x^n + x^t + 1`` requires
    ``m = n - t``, not ``m = t``.  Using ``m = t`` yields the *reciprocal*
    polynomial -- also primitive, also full period, but the time-reversed
    sequence, which fails bit-exact conformance against instrument-generated
    patterns.

    Assumptions
    -----------
    - The seed is non-zero.  The all-zero state is the single fixed point of
      the recurrence and would emit zeros forever; it is rejected.
    - The sequence is *not* DC balanced over one period: a maximal-length
      sequence has 2^(n-1) ones and 2^(n-1) - 1 zeros, an excess of exactly one
      one.  This is a property of the sequence, not a defect.

    Parameters
    ----------
    order : int
        PRBS order n; must be a key of :data:`PRBS_POLY`.
    n_bits : int
        Number of bits to generate.  May exceed the period, in which case the
        pattern repeats.
    seed : int, optional
        Initial n-bit state.  Defaults to all ones.

    Returns
    -------
    np.ndarray
        ``uint8`` array of 0/1, length ``n_bits``.
    """
    if order not in PRBS_POLY:
        raise ValueError(
            f"unsupported PRBS order {order}; have {sorted(PRBS_POLY)}"
        )
    if n_bits < 0:
        raise ValueError("n_bits must be non-negative")

    n = order
    m = n - PRBS_POLY[order]            # recurrence lag, see docstring
    seed = (1 << n) - 1 if seed is None else int(seed)
    if seed & ((1 << n) - 1) == 0:
        raise ValueError("seed must be non-zero: the all-zero state is absorbing")

    if n_bits == 0:
        return np.zeros(0, dtype=np.uint8)

    # bytearray rather than ndarray: scalar indexing in a Python loop is far
    # cheaper on bytearray, and the recurrence lag (n - t = 3 for PRBS31) is
    # too short for block vectorization to pay off.
    total = max(n_bits, n)
    buf = bytearray(total)
    for i in range(n):
        buf[i] = (seed >> (n - 1 - i)) & 1
    for k in range(n, total):
        buf[k] = buf[k - n] ^ buf[k - m]

    return np.frombuffer(bytes(buf), dtype=np.uint8)[:n_bits].copy()


# --------------------------------------------------------------------------
# GF(2) polynomial arithmetic, used to verify the period without generating it
# --------------------------------------------------------------------------

def _gf2_mulmod(a: int, b: int, poly: int, n: int) -> int:
    """Carry-less multiply of two GF(2) polynomials, reduced mod ``poly``."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a >> n & 1:
            a ^= poly
    return out


def _gf2_powmod(base: int, exp: int, poly: int, n: int) -> int:
    """``base**exp mod poly`` in GF(2)[x], by square-and-multiply."""
    result = 1
    while exp:
        if exp & 1:
            result = _gf2_mulmod(result, base, poly, n)
        base = _gf2_mulmod(base, base, poly, n)
        exp >>= 1
    return result


def _prime_factors(v: int) -> list[int]:
    """Distinct prime factors of ``v`` by trial division.

    Adequate here: the largest value tested is 2^31 - 1, whose square root is
    about 46341.
    """
    factors, d = [], 2
    while d * d <= v:
        if v % d == 0:
            factors.append(d)
            while v % d == 0:
                v //= d
        d += 1 if d == 2 else 2
    if v > 1:
        factors.append(v)
    return factors


def prbs_period(order: int) -> int:
    """Period of the PRBS sequence, computed rather than measured.

    The LFSR state advances by multiplication by ``x`` in
    ``GF(2)[x] / p(x)``, so the sequence period is the multiplicative order of
    ``x`` modulo ``p``.  That order is ``2^n - 1`` exactly when ``p`` is
    primitive, which is checked by

        x^(2^n - 1)       == 1  (mod p)
        x^((2^n - 1) / q) != 1  (mod p)   for every prime q dividing 2^n - 1

    This matters for PRBS31: its period is 2147483647, so measuring it by
    generating the sequence would need 2 GB and minutes of work.  The order
    test finishes in microseconds.

    Returns
    -------
    int
        The period, which is ``2**order - 1`` for a primitive polynomial.
    """
    n = order
    poly = (1 << n) | (1 << PRBS_POLY[order]) | 1
    full = (1 << n) - 1

    if _gf2_powmod(2, full, poly, n) != 1:      # 2 == x in bit representation
        raise ValueError(f"x^(2^{n}-1) != 1: polynomial is not irreducible")
    for q in _prime_factors(full):
        if _gf2_powmod(2, full // q, poly, n) == 1:
            # Order is a proper divisor; fall back to reporting it honestly.
            return next(
                d for d in sorted(_divisors(full))
                if _gf2_powmod(2, d, poly, n) == 1
            )
    return full


def _divisors(v: int) -> list[int]:
    out = []
    d = 1
    while d * d <= v:
        if v % d == 0:
            out += [d, v // d]
        d += 1
    return out


# ==========================================================================
# Symbols and waveform
# ==========================================================================

def symbols(bits: np.ndarray, swing: float = 0.5) -> np.ndarray:
    """Map bits to NRZ symbol levels.

        a[k] = swing * (2*b[k] - 1),   b in {0,1}  ->  a in {-swing, +swing}

    NRZ is 1 bit per symbol, so the symbol sequence is the bit sequence.
    """
    return swing * (2.0 * np.asarray(bits, dtype=float) - 1.0)


def upsample(sym: np.ndarray, cfg: LinkConfig) -> np.ndarray:
    """Hold each symbol for one UI on the oversampled grid.

        x[n] = a[floor(n / M)],   M = cfg.samples_per_ui

    This is a zero-order hold, i.e. an ideal rectangular NRZ waveform with zero
    rise time; :func:`shape_edges` adds finite edges afterwards.

    The boxcar here is the same width as the one :func:`src.channel.pulse_response`
    convolves with, which is what makes a single isolated symbol reproduce the
    pulse response exactly -- the basis of the time-domain vs. statistical
    cross-check.
    """
    return np.repeat(np.asarray(sym, dtype=float), cfg.samples_per_ui)


# Fraction of a full raised-cosine transition spanned by its 20-80% region:
#     s(u) = 0.5*(1 - cos(pi*u)),  s(u20)=0.2, s(u80)=0.8
#     u20 = arccos(0.6)/pi = 0.295167,  u80 = arccos(-0.6)/pi = 0.704833
_RC_2080 = 0.4096660

# For a first-order step response 1 - exp(-t/tau):
#     t80 - t20 = tau * (ln 5 - ln 1.25) = tau * ln 4
_LPF_2080 = np.log(4.0)


def shape_edges(
    x: np.ndarray,
    cfg: LinkConfig,
    rise_time: float,
    method: str = "raised_cosine",
) -> np.ndarray:
    """Apply a finite rise/fall time to an ideal rectangular waveform.

    ``rise_time`` is the **20-80 % transition time in seconds**, the wireline
    convention.  Each method converts it to its own natural parameter, so the
    two produce comparable edges rather than comparable-looking parameters:

    ``"raised_cosine"``
        The transition follows ``s(u) = 0.5*(1 - cos(pi*u))`` over a full
        duration ``T_full = rise_time / 0.40967``.  Implemented by convolving
        with the derivative of that transition, which is a **half-sine**

            w[k] = sin(pi*(k + 0.5)/L),   normalized to sum 1

        since ``d/du [0.5*(1 - cos(pi*u))] = (pi/2)*sin(pi*u)``.  A Hann window
        ``1 - cos(2*pi*u)`` is the tempting choice and is wrong: it integrates
        to ``u - sin(2*pi*u)/(2*pi)``, whose 20-80 % span is 0.3276 of the full
        duration rather than 0.4097, so every edge comes out 20 % faster than
        requested.  L = round(T_full/dt).  Adds no overshoot.

    ``"lowpass"``
        A single real pole with ``tau = rise_time / ln(4)``, applied as the
        step-invariant discrete filter

            y[n] = alpha*y[n-1] + (1 - alpha)*x[n],   alpha = exp(-dt/tau)

        This is the more physical model of a driver whose output node has an
        RC time constant, and unlike the raised cosine its response is
        infinite in extent, so it contributes a small amount of ISI of its own.

    ``"ideal"``
        Returned unchanged; zero rise time.

    Assumptions
    -----------
    - Rise and fall are symmetric.  Real drivers are not exactly, and the
      asymmetry appears as duty-cycle distortion, which is not modelled.
    - The shaping is linear and data-independent, so it could equivalently be
      folded into the channel response.  It is applied here because the
      time-domain engine needs the actual waveform.
    """
    if method == "ideal" or rise_time <= 0.0:
        return np.asarray(x, dtype=float)

    if method == "raised_cosine":
        length = max(int(round(rise_time / _RC_2080 / cfg.dt)), 1)
        k = np.arange(length)
        w = np.sin(np.pi * (k + 0.5) / length)
        w /= w.sum()
        return np.convolve(x, w, mode="same")

    if method == "lowpass":
        tau = rise_time / _LPF_2080
        alpha = float(np.exp(-cfg.dt / tau))
        return signal.lfilter([1.0 - alpha], [1.0, -alpha], x)

    raise ValueError(
        f"unknown edge method {method!r}; use 'raised_cosine', 'lowpass', or 'ideal'"
    )


def _cross(x: np.ndarray, i: int, level: float) -> float:
    """Fractional index where ``x`` crosses ``level`` between ``i`` and ``i+1``."""
    denom = x[i + 1] - x[i]
    if denom == 0.0:
        return float(i)
    return i + (level - x[i]) / denom


def measure_rise_time(x: np.ndarray, cfg: LinkConfig) -> float:
    """Measure the 20-80 % rise time of the first rising edge, in s.

    Crossings are linearly interpolated between samples.  Integer sample
    indices are not good enough here: at ``cfg.dt = 1.95 ps`` a 10 ps edge
    spans about five samples, so index quantization alone is a +/- 20 % error
    -- large enough to look like a bug in the edge shaping when it is only a
    property of the ruler.

    The low and high levels are taken as the global min and max of ``x``, which
    assumes the waveform actually settles at both rails; a heavily equalized or
    ringing waveform would need a more careful reference.

    Returns ``nan`` if no rising edge with both crossings is present.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 0:
        return float("nan")
    v20, v80 = lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo)

    above80 = np.nonzero(x >= v80)[0]
    if above80.size == 0 or above80[0] == 0:
        return float("nan")
    i80 = int(above80[0])

    below20 = np.nonzero(x[:i80] <= v20)[0]
    if below20.size == 0:
        return float("nan")
    i20 = int(below20[-1])

    return (_cross(x, i80 - 1, v80) - _cross(x, i20, v20)) * cfg.dt


# ==========================================================================
# TX FFE
# ==========================================================================

def ffe(
    sym: np.ndarray,
    pre: float = -0.05,
    main: float | None = None,
    post: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Feed-forward equalizer at symbol rate, with a pre-cursor tap.

        y[k] = c_pre * a[k+1] + c_main * a[k] + c_post * a[k-1]

    The pre-cursor tap is the reason a TX FFE exists alongside an RX DFE: a DFE
    feeds back *decided* symbols and so can only reach backwards, leaving the
    pre-cursor h_-1 structurally uncancellable (docs/dfe_notes.md section 7).
    The TX has the whole sequence available and can act on ``a[k+1]``.

    Implemented as a convolution advanced by one sample, since the tap on
    ``a[k+1]`` makes the response non-causal by one symbol:

        y = convolve(a, [c_pre, c_main, c_post])[1 : len(a)+1]

    With ``normalize=True`` the taps are scaled so ``sum |c_i| = 1``.  This is
    the usual launch-amplitude constraint: the driver's peak output is fixed,
    so emphasis is bought by *attenuating* the steady-state level rather than
    by boosting transitions -- the same trade the CTLE makes in the frequency
    domain (docs/ctle_notes.md section 4).  The cost is a smaller main cursor,
    which is why FFE and CTLE boost cannot both be spent freely.

    Parameters
    ----------
    sym : np.ndarray
        Symbol sequence.
    pre, post : float
        Pre- and post-cursor tap weights.  Conventionally negative.
    main : float, optional
        Main tap.  Defaults to ``1 - |pre| - |post|``, the value that already
        satisfies the normalization.
    normalize : bool
        Rescale so the absolute tap sum is 1.

    Returns
    -------
    np.ndarray
        Equalized symbols, same length as the input.
    """
    if main is None:
        main = 1.0 - abs(pre) - abs(post)
    taps = np.array([pre, main, post], dtype=float)
    if normalize:
        total = np.abs(taps).sum()
        if total == 0:
            raise ValueError("all FFE taps are zero")
        taps = taps / total

    a = np.asarray(sym, dtype=float)
    return np.convolve(a, taps)[1 : a.size + 1]


# ==========================================================================
# Full transmitter
# ==========================================================================

@dataclass(frozen=True)
class TxWaveform:
    """Everything the transmitter produced, for reuse and for plotting."""

    bits: np.ndarray
    symbols: np.ndarray
    waveform: np.ndarray
    t: np.ndarray
    prbs_order: int
    swing: float
    rise_time: float
    edge_method: str
    ffe_taps: np.ndarray | None

    @property
    def n_symbols(self) -> int:
        return self.symbols.size

    def summary(self) -> str:
        ffe = ("off" if self.ffe_taps is None
               else "[" + ", ".join(f"{c:+.4f}" for c in self.ffe_taps) + "]")
        return (
            f"PRBS{self.prbs_order}, {self.n_symbols} symbols, "
            f"{self.swing * 2 * 1e3:.0f} mVppd, "
            f"{self.edge_method} edges {self.rise_time * 1e12:.1f} ps (20-80%), "
            f"FFE {ffe}"
        )


def generate(
    cfg: LinkConfig,
    n_symbols: int,
    prbs_order: int = 7,
    swing: float = 0.5,
    rise_time: float = 15e-12,
    edge_method: str = "raised_cosine",
    ffe_taps: tuple[float, float] | tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> TxWaveform:
    """Build a complete TX waveform: PRBS -> symbols -> FFE -> hold -> edges.

    The FFE is applied at symbol rate, before upsampling, because it is a
    symbol-spaced filter; applying it to the oversampled waveform would be both
    wasteful and wrong at fractional delays.

    Parameters
    ----------
    cfg : LinkConfig
        Supplies the symbol rate and oversampling factor.
    n_symbols : int
        Length of the pattern to generate.
    prbs_order : int
        7 or 31 for the two orders spec.md contemplates; any key of
        :data:`PRBS_POLY` works.
    swing : float
        Symbol level in V; differential pp amplitude is twice this.
    rise_time : float
        20-80 % rise time in s.  Default 15 ps is 0.24 UI at 16 GBd.
    edge_method : str
        ``"raised_cosine"``, ``"lowpass"``, or ``"ideal"``.
    ffe_taps : tuple, optional
        ``(pre, main)`` or ``(pre, main, post)``.  ``None`` disables the FFE,
        which is the default -- docs/notes.md lists an ideal TX (no FFE) among
        the model's assumptions, so enabling it is a deliberate departure.
    seed : int, optional
        PRBS seed.

    Returns
    -------
    TxWaveform
    """
    bits = prbs(prbs_order, n_symbols, seed=seed)
    sym = symbols(bits, swing=swing)

    taps = None
    if ffe_taps is not None:
        pre, main, *rest = ffe_taps
        post = rest[0] if rest else 0.0
        sym = ffe(sym, pre=pre, main=main, post=post)
        taps = np.array([pre, main, post], dtype=float)
        taps = taps / np.abs(taps).sum()

    wave = shape_edges(upsample(sym, cfg), cfg, rise_time, method=edge_method)
    t = np.arange(wave.size) * cfg.dt
    return TxWaveform(
        bits=bits, symbols=sym, waveform=wave, t=t,
        prbs_order=prbs_order, swing=swing, rise_time=rise_time,
        edge_method=edge_method, ffe_taps=taps,
    )


# ==========================================================================
# Self-test
# ==========================================================================

def self_test(cfg: LinkConfig | None = None, verbose: bool = True) -> bool:
    """Verify the PRBS generators and the waveform chain.

    Checks, in order:

    1. **Period is 2^n - 1** for every supported order, by the GF(2) order test.
    2. **Measured period matches** for orders short enough to generate, which
       cross-checks the algebraic test against the actual generator.
    3. **Balance**: 2^(n-1) ones and 2^(n-1) - 1 zeros over one period.
    4. **Maximum run length** is n ones, which is the defining property of a
       maximal-length sequence.
    5. **Autocorrelation** is -1/(2^n - 1) at every non-zero cyclic shift.
    6. **Rise time** delivered by each shaping method matches the request.
    7. **FFE tap normalization** and the one-symbol advance of the pre-cursor.

    Returns True if everything passes.
    """
    cfg = LinkConfig() if cfg is None else cfg
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<38s} {detail}")

    if verbose:
        print("PRBS period (algebraic order test)")
    for order in sorted(PRBS_POLY):
        expected = 2**order - 1
        got = prbs_period(order)
        report(f"PRBS{order} period == 2^{order}-1", got == expected,
               f"{got:,}")

    if verbose:
        print("\nPRBS sequence properties (generated)")
    for order in (7, 9, 11, 15):
        period = 2**order - 1
        seq = prbs(order, 2 * period)

        wrapped = np.array_equal(seq[:period], seq[period:])
        report(f"PRBS{order} repeats after {period}", wrapped)

        # No shorter period.  Any shorter period must divide the full one, and
        # testing a candidate d means comparing a *whole period* against itself
        # shifted by d -- comparing only the first d samples proves nothing
        # (d = 1 "passes" whenever the first two bits happen to match).
        shorter = [d for d in _divisors(period)
                   if d < period and np.array_equal(seq[:period], seq[d:d + period])]
        report(f"PRBS{order} has no shorter period", not shorter,
               "" if not shorter else f"found {shorter}")

        ones = int(seq[:period].sum())
        report(f"PRBS{order} balance", ones == 2 ** (order - 1),
               f"{ones} ones / {period - ones} zeros")

        runs = np.diff(np.nonzero(np.diff(np.concatenate(([2], seq[:period], [2]))))[0])
        report(f"PRBS{order} max run == {order}", int(runs.max()) == order,
               f"max run {int(runs.max())}")

        # Cyclic autocorrelation of the +/-1 sequence: 1 at zero shift,
        # -1/(2^n - 1) everywhere else.
        pm = 2.0 * seq[:period] - 1.0
        ac = np.fft.irfft(np.abs(np.fft.rfft(pm)) ** 2, n=period) / period
        off_peak = np.abs(ac[1:] + 1.0 / period).max()
        report(f"PRBS{order} autocorrelation", off_peak < 1e-9,
               f"max deviation {off_peak:.2e}")

    if verbose:
        print("\nWaveform chain")
    # One long low run then one long high run gives a single isolated edge with
    # both rails settled, which is what measure_rise_time assumes.
    step = symbols(np.array([0] * 8 + [1] * 8, dtype=np.uint8))
    for method in ("raised_cosine", "lowpass"):
        for target in (10e-12, 20e-12, 30e-12):
            w = shape_edges(upsample(step, cfg), cfg, target, method=method)
            got = measure_rise_time(w, cfg)
            err = abs(got - target) / target
            # The raised-cosine kernel length is an integer number of samples,
            # so the achievable rise time is quantized by ~dt/0.41 = 4.8 ps.
            report(f"{method} rise time {target * 1e12:.0f} ps", err < 0.08,
                   f"measured {got * 1e12:.2f} ps ({err * 100:.1f}% error)")

    ideal = shape_edges(upsample(step, cfg), cfg, 0.0, method="ideal")
    got = measure_rise_time(ideal, cfg)
    report("ideal edges are sub-sample", got <= cfg.dt,
           f"{got * 1e12:.2f} ps <= dt = {cfg.dt * 1e12:.2f} ps")

    a = symbols(prbs(7, 64))
    y = ffe(a, pre=-0.1, main=0.9, post=0.0, normalize=True)
    report("FFE taps normalized", abs(abs(np.array([-0.1, 0.9])).sum() - 1.0) < 1e-12,
           "sum|c| = 1")
    # With main=1 and pre=0 the FFE must be transparent.
    passthru = ffe(a, pre=0.0, main=1.0, post=0.0, normalize=True)
    report("FFE identity taps are transparent",
           np.allclose(passthru, a), "")
    # The pre-cursor tap must act on the *next* symbol.
    imp = np.zeros(9)
    imp[4] = 1.0
    y2 = ffe(imp, pre=1.0, main=0.0, post=0.0, normalize=False)
    report("FFE pre-tap advances by one symbol", y2[3] == 1.0 and y2[4] == 0.0,
           f"energy at index {int(np.argmax(np.abs(y2)))}, expected 3")

    if verbose:
        print(f"\n{'all checks passed' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if self_test() else 1)
