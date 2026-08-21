"""Baseline: 16 Gbps NRZ over the 802.3ck 16+16 inch backplane, full RX.

CTLE values are the design worked through in docs/ctle_notes.md section 7.1 for
this channel's measured 13.19 dB Nyquist loss:

    f_peak ~= f_N = 8 GHz          ->  f_p1 * f_p2 ~= 64 GHz^2
    f_p2 = 3 * f_p1                ->  f_p1 = 4.62 GHz, f_p2 = 13.9 GHz
    boost 13.2 dB, +2.4 dB for the finite-f_p2 deficit
                                   ->  f_p1/f_z = 6.03, f_z = 0.77 GHz

The deficit correction in step 3 is the step that is easy to skip; the
asymptotic formula alone lands 2.4 dB short of target.
"""

from __future__ import annotations

from ..config import CDRConfig, CTLEConfig, DFEConfig, LinkConfig
from ..params import Bounded

F_P1 = 4.62e9
POLE_RATIO = 3.0        # f_p2 / f_p1, set by the achievable output-node bandwidth
BOOST_RATIO = 6.03      # f_p1 / f_z, including the finite-f_p2 deficit


def config() -> LinkConfig:
    """The default design point."""
    ctle = CTLEConfig(
        f_z=Bounded(
            F_P1 / BOOST_RATIO, 0.1e9, 5e9, unit="Hz",
            note="13.2 dB boost target incl. 2.4 dB finite-f_p2 deficit",
        ),
        f_p1=Bounded(F_P1, 1e9, 12e9, unit="Hz", note="f_peak ~= sqrt(f_p1*f_p2) = 8 GHz"),
        f_p2=Bounded(
            POLE_RATIO * F_P1, 5e9, 40e9, unit="Hz",
            note="output-node pole; ratio 3 assumed achievable",
        ),
        n_stages=Bounded(1, 1, 2, step=1, note="one stage suffices at 13.2 dB"),
    )
    return LinkConfig(ctle=ctle, dfe=DFEConfig(), cdr=CDRConfig())
