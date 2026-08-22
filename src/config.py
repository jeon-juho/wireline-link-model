"""Link configuration.

A single ``LinkConfig`` instance is threaded through every module; no module
holds mutable state of its own.  All fields are SI units.

Tunable parameters are :class:`~src.params.Bounded` rather than bare floats, so
a sweep or optimizer can read its search space off the config directly --
``params.free_parameters(cfg)`` returns every knob with its admissible range.
See docs/statopt_review.md for where the pattern comes from.

Nested blocks use ``field(default_factory=...)``.  A plain ``ctle: CTLEConfig =
CTLEConfig()`` default would be evaluated once at class-definition time and
shared by every ``LinkConfig`` ever constructed -- the aliasing bug documented
in docs/statopt_review.md, "Patterns to avoid" section 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .params import Bounded

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHANNEL = (
    REPO_ROOT
    / "channel"
    / "palkert_3ck_02_0120"
    / "THRU_VL5_OD-BP-Channel_16inch_16inch.s4p"
)


@dataclass(frozen=True)
class CTLEConfig:
    """Continuous-time linear equalizer: one zero, two poles.

        H(s) = A_dc * (1 + s/w_z) / [(1 + s/w_p1)(1 + s/w_p2)]

    Defaults are the design derived in docs/ctle_notes.md section 7.1 for this
    channel's 13.2 dB Nyquist loss.
    """

    enabled: bool = True
    f_z: Bounded = field(
        default_factory=lambda: Bounded(
            0.77e9, 0.1e9, 5e9, unit="Hz",
            note="zero; sets boost height, must sit below f_p1 (ctle_notes 3)",
        )
    )
    f_p1: Bounded = field(
        default_factory=lambda: Bounded(
            4.62e9, 1e9, 12e9, unit="Hz",
            note="first pole; f_p1/f_z is the asymptotic boost (ctle_notes 4)",
        )
    )
    f_p2: Bounded = field(
        default_factory=lambda: Bounded(
            13.9e9, 5e9, 40e9, unit="Hz",
            note="output-node pole 1/(R_D*C_L); finite value caps peaking (ctle_notes 6)",
        )
    )
    n_stages: Bounded = field(
        default_factory=lambda: Bounded(
            1, 1, 2, step=1,
            note="one slope per stage; 2 needed only if one cannot match a*sqrt(f)+b*f",
        )
    )

    #: Linear boost beyond this enhances noise and crosstalk more than it helps
    #: the eye (docs/ctle_notes.md section 4).  Not a knob -- a ceiling.
    max_boost_db: float = 25.0


@dataclass(frozen=True)
class DFEConfig:
    """Decision feedback equalizer.

    Taps are not free parameters in the usual sense: for feedback taps the
    zero-forcing and MMSE solutions coincide, so the optimum is always the
    postcursor of the combined channel + CTLE response (docs/dfe_notes.md
    section 5.2).  What *is* a design choice is how many taps to spend.
    """

    enabled: bool = True
    n_taps: Bounded = field(
        default_factory=lambda: Bounded(
            2, 1, 4, step=1,
            note="spec section 2 calls for 1-2; bound allows testing whether that suffices",
        )
    )

    #: Error propagation stays benign for roughly |w| <= 0.5
    #: (docs/dfe_notes.md section 7).  A tap solution above this is a red flag,
    #: not something to clip silently.
    max_tap_weight: float = 0.5


@dataclass(frozen=True)
class CDRConfig:
    """Baud-rate bang-bang CDR, type-II digital loop.

        w_n = sqrt(K_pd*K_i)/T ,   zeta = (K_p/2)*sqrt(K_pd/K_i)

    Defaults follow the consistency check in docs/cdr_notes.md section 9, which
    shows that without decimation the loop lands ~100x above the target
    bandwidth and at the f_baud/10 stability ceiling.
    """

    enabled: bool = True
    k_p: Bounded = field(
        default_factory=lambda: Bounded(
            1.0 / 64, 1.0 / 1024, 1.0 / 16, unit="UI",
            note="proportional gain; sets both loop BW and dither (cdr_notes 8)",
        )
    )
    k_i: Bounded = field(
        default_factory=lambda: Bounded(
            2.7e-4, 1e-8, 1e-2,
            note="integral gain; sets w_n and frequency tracking, not bandwidth. "
                 "Raw loop gain, NOT normalized by K_pd: zeta = (k_p/2)*sqrt(K_pd/k_i), "
                 "so this value targets zeta = 3 at sigma_tau = 0.02 UI. Folding K_pd "
                 "in instead (cdr_notes section 4's normalized form) would read 6.8e-6 "
                 "and give zeta = 18.9 -- outside the intended 2..5.",
        )
    )
    decimation: Bounded = field(
        default_factory=lambda: Bounded(
            100, 8, 256, step=1,
            note="PD updates per phase update; without it the loop is unusable (cdr_notes 9)",
        )
    )
    zeta_target: Bounded = field(
        default_factory=lambda: Bounded(
            3.0, 1.0, 5.0,
            note="overdamped by design; type-II peaking never reaches zero (cdr_notes 6)",
        )
    )
    sigma_tau: Bounded = field(
        default_factory=lambda: Bounded(
            0.02, 0.005, 0.1, unit="UI",
            note="rms jitter that dithers the quantizer; K_pd = 2/(sigma*sqrt(2pi))",
        )
    )


@dataclass(frozen=True)
class LinkConfig:
    """All link and simulation parameters, in SI units.

    Attributes
    ----------
    symbol_rate : float
        Symbol rate Rs in Bd.  For NRZ (1 bit/symbol) this equals the bit rate,
        so 16 Gbps -> 16e9 Bd.
    samples_per_ui : int
        Oversampling factor M of the analog-domain time grid.  The analog
        sample rate is ``fs = M * Rs``; M must be an integer so that one UI
        spans a whole number of samples and symbol-rate sampling is exact.  It
        also sets the sampling-phase resolution, since a phase sweep steps
        through the M sub-UI positions.
    n_time : int
        Number of samples in the extracted impulse response.  Sets the time
        window ``T = n_time / fs`` and the frequency resolution ``df = fs /
        n_time`` of the grid the channel is resampled onto.
    n_pre, isi_depth : int
        Precursor and postcursor depth for peak-distortion and statistical
        analysis.  ``isi_depth`` is not a free choice: on this channel 20 UI
        captures only 86 % of the postcursor tail and overstates the eye by
        ~70 mV.  See docs/spec.md section 4.
    channel_path : Path
        Touchstone .s4p file for the channel.
    """

    symbol_rate: float = 16e9
    samples_per_ui: int = 32
    n_time: int = 32768
    n_pre: int = 4
    isi_depth: int = 200
    channel_path: Path = DEFAULT_CHANNEL

    ctle: CTLEConfig = field(default_factory=CTLEConfig)
    dfe: DFEConfig = field(default_factory=DFEConfig)
    cdr: CDRConfig = field(default_factory=CDRConfig)

    @property
    def bit_rate(self) -> float:
        """Bit rate in b/s.  NRZ is 1 bit/symbol, so bit_rate == symbol_rate."""
        return self.symbol_rate

    @property
    def ui(self) -> float:
        """Unit interval in s.

        UI = 1 / Rs
        """
        return 1.0 / self.symbol_rate

    @property
    def nyquist(self) -> float:
        """Signal Nyquist frequency in Hz.

        f_nyq = Rs / 2
        """
        return 0.5 * self.symbol_rate

    @property
    def fs(self) -> float:
        """Analog-domain sample rate in Hz.

        fs = M * Rs
        """
        return self.samples_per_ui * self.symbol_rate

    @property
    def dt(self) -> float:
        """Analog-domain sample period in s.

        dt = 1 / fs = UI / M
        """
        return 1.0 / self.fs

    @property
    def time_window(self) -> float:
        """Duration of the impulse-response record in s.

        T = n_time / fs
        """
        return self.n_time / self.fs

    @property
    def df(self) -> float:
        """Frequency spacing of the resampled analysis grid in Hz.

        df = fs / n_time = 1 / T
        """
        return self.fs / self.n_time

    def knobs(self) -> dict[str, Bounded]:
        """Every tunable parameter with its bounds, as a flat dotted mapping."""
        from .params import free_parameters

        return free_parameters(self)
