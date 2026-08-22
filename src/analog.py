"""Analog chain: cascading LTI blocks in the frequency domain.

The channel and the CTLE are both LTI, so they compose by multiplication.  This
module owns that composition; the path from a measured .s4p through to cursors
lives in :mod:`src.channel`, and applies equally to a cascaded response.

The single shared artifact the pipeline produces -- the impulse response, and
the pulse response derived from it -- is what both the time-domain and the
statistical engine consume.  Computing it once from one composed response is
what makes the two methods comparable (docs/spec.md criterion 3).
"""

from __future__ import annotations

import numpy as np

from .channel import analysis_grid, impulse_response, prepare_response, pulse_response
from .config import LinkConfig

__all__ = [
    "compose",
    "cascade_to_pulse",
    # re-exported so callers can treat the analog chain as one namespace
    "analysis_grid",
    "impulse_response",
    "prepare_response",
    "pulse_response",
]


def compose(*responses: np.ndarray) -> np.ndarray:
    """Cascade LTI blocks by multiplying their frequency responses.

        H(f) = H_1(f) * H_2(f) * ... * H_n(f)

    Assumptions
    -----------
    - The blocks do not load one another: each is driven from an ideal source
      and terminated ideally, so no inter-stage reflection occurs.  This is
      assumption 1 in docs/architecture.md and is the main idealization in the
      analog path -- a real CTLE input presents a finite impedance to the
      channel termination.
    - All inputs are already on a common frequency grid.  Composing responses
      sampled on different grids is a silent error, so shape is checked.
    """
    if not responses:
        raise ValueError("compose() needs at least one response")
    out = np.ones_like(responses[0], dtype=complex)
    for h in responses:
        if h.shape != out.shape:
            raise ValueError("all responses must share the same frequency grid")
        out = out * h
    return out


def cascade_to_pulse(
    f: np.ndarray, *responses: np.ndarray, cfg: LinkConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose responses and take the cascade through to a pulse response.

        H(f) = prod_i H_i(f)  ->  h(t)  ->  p(t)

    Convenience wrapper for the common channel-times-CTLE case; identical to
    calling :func:`compose`, then ``channel.impulse_response``, then
    ``channel.pulse_response``.

    Returns ``(t, h, p)``.
    """
    h_f = compose(*responses)
    t, h_t, _ = impulse_response(f, h_f, cfg)
    return t, h_t, pulse_response(h_t, cfg)
