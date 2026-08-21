"""No equalization -- the "before" case for spec.md criterion 1.

Same channel and analysis grid as :mod:`~src.configs.baseline`, with CTLE, DFE,
and CDR disabled.  This is the closed-eye reference every equalization result is
measured against, not a design anyone would build.
"""

from __future__ import annotations

from dataclasses import replace

from ..config import LinkConfig
from . import baseline


def config() -> LinkConfig:
    """Baseline with every equalizer switched off."""
    cfg = baseline.config()
    return replace(
        cfg,
        ctle=replace(cfg.ctle, enabled=False),
        dfe=replace(cfg.dfe, enabled=False),
        cdr=replace(cfg.cdr, enabled=False),
    )
