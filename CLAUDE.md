# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

Before doing anything else in this repo, read:

- `docs/spec.md` — link specification and target performance
- `docs/architecture.md` — model structure and block decomposition

If either file does not exist yet, say so and ask before assuming requirements.

## Project

A 16 Gbps NRZ wireline receiver link model implemented in Python. The receive chain
under study is **CTLE → DFE → CDR**, driven through a measured channel.

## Environment

Windows, Python 3.11, virtual environment at `.venv`.
Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-rf`, `control`.

```powershell
# activate (PowerShell)
.venv\Scripts\Activate.ps1

# run a simulation
python scripts\<script_name>.py
```

`scikit-rf` handles S-parameter (Touchstone) channel data; `control` handles
continuous/discrete-time transfer functions for the CTLE and CDR loop.

## Layout

| Directory   | Contents                                                     |
|-------------|--------------------------------------------------------------|
| `src/`      | Library modules — the model itself                            |
| `scripts/`  | Executable entry points that drive the modules                |
| `channel/`  | Channel S-parameter files (`.s4p`)                            |
| `results/`  | Simulation output (plots, data)                               |
| `docs/`     | Specification and design documents                            |

## Working rules

**Modules vs. scripts.** New code goes in `src/` as importable modules. Anything
that runs goes in `scripts/` as a script. Never mix the two — no simulation
driver code inside `src/`, no reusable model logic inside `scripts/`.

**Document the math.** Every numerical function carries the underlying theoretical
expression in its docstring — the equation it implements, not just a prose
description of what it returns.

**SI units throughout.** Frequency in Hz, time in s, and all other physical
quantities in base SI units. No GHz, ps, or mV in function signatures or return
values; convert only at plot/display boundaries, and state the unit in every
docstring.

**Surface your assumptions.** Whenever you build or extend a model, present an
explicit list of *unverified assumptions* alongside it — simplifications,
values taken on faith, effects deliberately omitted. This list is part of the
deliverable, not an optional footnote.
