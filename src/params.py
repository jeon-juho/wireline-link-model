"""Bounded parameters.

Adapted from StatOpt's ``valueWithLimits`` (see docs/statopt_review.md §1): a
tunable parameter is not a bare float but a value carrying its own admissible
range.  The payoff is that a sweep or optimizer reads its search space directly
off the configuration object, so there is never a second, separately maintained
description of which parameters are free and how far they may move.

Bounds here are physical, not arbitrary.  A DFE tap is limited to |w| <= 0.5
because error propagation grows with tap weight (docs/dfe_notes.md section 7);
a CTLE boost is limited because a linear equalizer amplifies noise and crosstalk
along with the signal (docs/ctle_notes.md section 4).  Recording the reason
alongside the number is the point of the ``note`` field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Bounded:
    """A scalar with an admissible range, and optionally a sweep step.

    Parameters
    ----------
    value : float
        Current value, in ``unit``.
    lo, hi : float
        Inclusive bounds.  Default to -inf / +inf, i.e. unconstrained.
    step : float, optional
        Natural increment for sweeps.  If omitted, :meth:`sweep` divides the
        range evenly instead.
    unit : str
        SI unit string for display only -- ``"Hz"``, ``"s"``, ``"V"``, or ``""``
        for a dimensionless quantity.  Values are always stored in base SI.
    note : str
        Why the bounds are what they are.  Cite the note or spec section.
    """

    value: float
    lo: float = -math.inf
    hi: float = math.inf
    step: float | None = None
    unit: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) exceeds hi ({self.hi})")
        if not (self.lo <= self.value <= self.hi):
            raise ValueError(
                f"value {self.value} outside bounds [{self.lo}, {self.hi}]"
                + (f" -- {self.note}" if self.note else "")
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")

    @property
    def free(self) -> bool:
        """True if the parameter has a finite range to search over."""
        return math.isfinite(self.lo) and math.isfinite(self.hi) and self.hi > self.lo

    def clip(self, v: float) -> float:
        """Clamp ``v`` into the admissible range."""
        return float(min(max(v, self.lo), self.hi))

    def at(self, v: float) -> "Bounded":
        """Return a copy at a new value, clipped into range."""
        return replace(self, value=self.clip(v))

    def sweep(self, n: int = 11) -> np.ndarray:
        """Candidate values spanning the admissible range.

        Uses ``step`` when one is defined, otherwise ``n`` evenly spaced points.
        This is the search space an optimizer or parameter sweep consumes.
        """
        if not self.free:
            raise ValueError(f"cannot sweep an unbounded parameter: {self!r}")
        if self.step is not None:
            return np.arange(self.lo, self.hi + 0.5 * self.step, self.step)
        return np.linspace(self.lo, self.hi, n)

    def __float__(self) -> float:
        return float(self.value)

    def describe(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        rng = f"[{self.lo:g}, {self.hi:g}]" if self.free else "unbounded"
        return f"{self.value:g}{u}  in {rng}" + (f"  -- {self.note}" if self.note else "")


def free_parameters(obj, prefix: str = "") -> dict[str, Bounded]:
    """Collect every :class:`Bounded` field in a (possibly nested) config.

    Returns a flat mapping of dotted name -> parameter, e.g.
    ``{"ctle.f_z": Bounded(...), "cdr.k_p": Bounded(...)}``.  This is the
    equivalent of StatOpt's ``adaption.knobs`` list, except it is derived from
    the configuration rather than maintained separately, so it cannot fall out
    of sync with it.
    """
    found: dict[str, Bounded] = {}
    for name in getattr(obj, "__dataclass_fields__", {}):
        val = getattr(obj, name)
        dotted = f"{prefix}{name}"
        if isinstance(val, Bounded):
            found[dotted] = val
        elif hasattr(val, "__dataclass_fields__"):
            found.update(free_parameters(val, prefix=f"{dotted}."))
    return found
