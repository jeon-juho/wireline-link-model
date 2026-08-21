"""Experiment configurations, one module per scenario.

Adapted from StatOpt's ``generateUserSettingsExample*.py`` convention
(docs/statopt_review.md section 3): each experiment is an executable Python
module returning a fully populated :class:`~src.config.LinkConfig`, rather than
a YAML or JSON file.  Python buys computed values -- ``f_p1 = 3 * f_z`` -- and
makes every past experiment a version-controlled, self-documenting artifact.

Unlike StatOpt, the active configuration is selected by name at run time rather
than by editing an import statement:

    python scripts\\plot_channel.py --config no_eq
"""

from __future__ import annotations

from importlib import import_module

from ..config import LinkConfig

#: Registered experiment names, mapped to the module providing ``config()``.
REGISTRY: dict[str, str] = {
    "baseline": "baseline",
    "no_eq": "no_eq",
}


def available() -> list[str]:
    """Names accepted by :func:`get`."""
    return sorted(REGISTRY)


def get(name: str = "baseline") -> LinkConfig:
    """Build the named experiment configuration.

    Raises
    ------
    KeyError
        If ``name`` is not registered, listing what is.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown config {name!r}; available: {', '.join(available())}")
    module = import_module(f"{__name__}.{REGISTRY[name]}")
    return module.config()
