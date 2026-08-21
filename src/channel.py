"""Channel: Touchstone .s4p loading and mixed-mode extraction.

Turns a 4-port single-ended S-parameter file into the differential insertion
response Sdd21(f) that the rest of the analog chain consumes.

Port ordering
-------------
A 4-port differential file can use either of two common conventions, and the
file itself does not record which.  Getting it wrong silently produces a
plausible-looking but meaningless result, so the convention is detected from
the data rather than assumed -- see :func:`detect_port_order`.

    THROUGH_PAIRED  ports (1,2) are the two ends of one line, (3,4) the two
                    ends of the other.  Differential pairs are {1,3} at the
                    near end and {2,4} at the far end.  Strong |S21|, |S43|.

    END_PAIRED      ports (1,2) are the (+,-) of the near-end pair and (3,4)
                    the (+,-) of the far end.  Strong |S31|, |S42|.
                    This is the ordering scikit-rf's se2gmm expects.

The bundled IEEE 802.3ck backplane file is THROUGH_PAIRED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import skrf as rf

THROUGH_PAIRED = "through_paired"
END_PAIRED = "end_paired"


def detect_port_order(ntwk: rf.Network, f_max: float = 1e9) -> str:
    """Infer the port-ordering convention from the coupling structure.

    In a through channel the dominant off-diagonal transmission terms are the
    two single-ended through paths, so comparing the two candidate pairings

        through_paired  if  mean(|S21| + |S43|)  >  mean(|S31| + |S42|)
        end_paired      otherwise

    identifies the convention.

    The comparison is restricted to ``f <= f_max`` (1 GHz by default).  This
    matters: a lossy backplane's through path falls off steeply with frequency
    while residual coupling does not, so over a full 0-50 GHz band the coupling
    terms can outweigh the through terms and invert the test.  On the bundled
    802.3ck channel the full-band sums are through 576 vs. end 749 -- the wrong
    answer -- while below 1 GHz they are 1.55 vs. 0.19, an unambiguous 8:1.
    Near DC the channel is nearly lossless, which is what makes the test sound.

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


def differential_response(ntwk: rf.Network) -> tuple[np.ndarray, np.ndarray]:
    """Extract the differential insertion response Sdd21(f).

    For a pair with terminals (a+, a-) driving a pair (b+, b-), the
    differential-to-differential transmission term is

        Sdd_ba = 1/2 * ( S_{b+,a+} - S_{b+,a-} - S_{b-,a+} + S_{b-,a-} )

    which is the ba element of ``M S M^-1`` under the standard mixed-mode
    transform M.  This function applies scikit-rf's generalized mixed-mode
    conversion, which additionally renormalises the differential reference
    impedance to 2*z0 (100 ohm for a 50 ohm file).

    Parameters
    ----------
    ntwk : skrf.Network
        4-port network in END_PAIRED ordering, as returned by :func:`load`.

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
