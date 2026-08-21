# StatOpt Review — Design Patterns Worth Borrowing

Analysis of [savob/statopt-python](https://github.com/savob/statopt-python), a
Python port of the MATLAB StatOpt statistical link modelling tool. It solves
roughly the problem our statistical engine ([`spec.md`](spec.md) §4) solves, at
larger scope: PAM-4, crosstalk, jitter, noise, distortion, and genetic-algorithm
equalizer optimization.

> Based on reading the repository structure and the key modules
> (`statopt.py`, `userSettingsObjects.py`, `generateISI.py`, `generatePDF.py`,
> `generateBER.py`, and a settings example), not a line-by-line audit of all 20
> modules.

## Adoption status

| Pattern | Status |
|---|---|
| `valueWithLimits` → `Bounded` | **adopted** — `src/params.py`, all knobs in `src/config.py` |
| Config blocks via `field(default_factory=...)` | **adopted** — aliasing bug avoided by construction |
| Experiment configs as Python modules | **adopted** — `src/configs/`, selected with `--config` |
| Per-symbol pulse split → phase-resolved cursors | **adopted** — `SplitPulse` in `src/analog.py`; this was a correction, and it changed a headline result (`spec.md` §5) |
| Pipeline unaware of the optimizer | **held** — engines stay pure functions of `LinkConfig` |
| `generate*` / `display*` split by domain | **deferred** — `plotting.py` is still one module; seam noted |
| Statistical eye pipeline (§6 below) | **pending** — blueprint for `sim_stat.py`, not yet written |

## How it is organized

Flat namespace at repo root, ~20 modules, split by verb:

| Prefix | Modules | Role |
|---|---|---|
| `generate*` | `PulseResponse`, `ISI`, `PDF`, `BER`, `Results`, `FixedInfluence`, `VariableInfluence` | all computation |
| `display*` | `Responses`, `Distributions`, `Interferences`, `Results` | all plotting |
| config | `userSettingsObjects`, `generateUserSettingsExample0..4`, `generateSettingsLimits`, `checkSettings` | parameters |
| control | `statopt.py`, `initializeSimulation`, `adaption` | pipeline and optimization |

The whole simulation is one loop in `statopt.py`:

```python
simSettings = generateUserSettings()
generateSettingsLimits(simSettings)
simResults = initializeSimulation(simSettings)
checkSettings(simSettings)
generateFixedInfluence(simSettings, simResults)

while not simResults.finished:
    generateVariableInfluence(simSettings, simResults)
    generatePulseResponse(simSettings, simResults)
    generateISI(simSettings, simResults)
    generatePDF(simSettings, simResults)
    generateBER(simSettings, simResults)
    generateResults(simSettings, simResults)
    adaptLink(simSettings, simResults)      # mutates simSettings, may set finished
```

Every stage has the identical signature `f(simSettings, simResults)` and mutates
`simResults` in place.

---

## Patterns worth adopting

### 1. Parameters carry their own bounds — `valueWithLimits`

The single best idea in the codebase. No tunable parameter is a bare float:

```python
@dataclass(init=False)
class valueWithLimits:
    value: float = float("nan")
    maxValue: float = float("nan")
    minValue: float = float("nan")
    increment: float = float("nan")
    minIncrement: float = float("nan")
    maxIncrement: float = float("nan")
```

The optimizer's search space is therefore declared inline with the values, so
`adaptLink()` needs no separate parameter-space description — it reads bounds off
the settings object it is already given.

**For us:** every open item in `spec.md` §6 is a parameter with a physically
meaningful range — CTLE `f_z`/`f_p1`/`f_p2`, DFE tap weights (bounded by the
`|w₁| ≲ 0.5` error-propagation limit from `dfe_notes.md` §7), CDR `K_p`/`K_i`.
If `LinkConfig` holds these as bounded values rather than floats, any later sweep
or optimization gets its search space for free, and `checkSettings`-style
validation becomes possible.

### 2. Separate the computation namespace from the plotting namespace

`generate*` vs `display*` is our `src/` vs `plotting.py` rule, enforced by naming
across the whole codebase. Their version scales better than ours: four display
modules split by domain rather than one monolith.

**For us:** `architecture.md` §7 already says nothing in `src/` writes to disk
except `plotting.py`. When that file outgrows itself, split it their way —
`plotting/responses.py`, `plotting/distributions.py`, `plotting/results.py` —
rather than letting one module accumulate every figure.

### 3. Experiment configs as executable Python, one file per scenario

`generateUserSettingsExample0.py` … `Example4.py` — each a function returning a
fully populated settings object, with units in comments:

```python
simSettings.general.symbolRate.value  = 32e9    # symbol rate [S/s]
simSettings.general.samplesPerSymb.value = 100
simSettings.receiver.CTLE.zeroFreq.value = 20.5e9   # frequency of first zero [Hz]
simSettings.receiver.FFE.taps.pre1 = valueWithLimits(-0.05)
```

Python rather than YAML buys computed values (`f_p1 = 3 * f_z`) and makes each
past experiment a version-controlled, self-documenting artifact.

**For us:** a `configs/` package with one module per experiment, each returning a
`LinkConfig`. Fits the `src/`-modules / `scripts/`-drivers rule without straining
it.

### 4. Keep the physics pipeline unaware that it is being optimized

The entire chain is a pure-ish function of settings, wrapped in a loop that an
optimizer drives. `generatePulseResponse` has no idea `adaptLink` exists.

**For us:** the CTLE/DFE/CDR values are all still open. If `sim_stat.run(cfg)` and
`sim_time.run(cfg)` stay clean functions of config, sweeping or optimizing them
later is a wrapper, not a refactor. This is worth protecting now, while the
engines are still unwritten.

### 5. Split the pulse response into per-symbol portions, not just cursors

`generateISI` keeps `splitPulse` as an explicit structure — `pre1`, `main`,
`post0`, … — retaining the full sub-UI waveform for each symbol slot, then
multiplies cursor combinations against those portions to build trajectories.

**For us this is a correction, not an enhancement.** Our `analog.cursors()`
samples only at `argmax(p)`. That is wrong for two reasons already documented:
an MM CDR locks where `h₁ = h₋₁`, not at the peak (`dfe_notes.md` §6), and eye
height must be evaluated across sampling phase, not at one point. Cursors need to
become a function of phase, which is exactly what keeping the per-symbol
waveform gives us.

### 6. The statistical eye pipeline

The canonical algorithm, and it maps directly onto our `sim_stat.py`:

1. Enumerate all cursor combinations (base-M over the modulation levels) as polar
   ±1 vectors.
2. Group combinations by their main/post-cursor transitions.
3. Multiply each combination against the split pulse portions → trajectories.
4. Bin trajectories into a **fixed pre-allocated 2D grid**, voltage × time-within-symbol.
5. Convolve **jitter horizontally** (across time) and **noise vertically** (across voltage).
6. Normalize each time column to sum to 1.
7. Integrate the wrong side of each threshold → BER contour → bathtub curves.

Two implementation details worth taking on trust rather than rediscovering:

- Before the horizontal jitter convolution they **concatenate adjacent symbol
  PDFs, convolve, then trim back** to one symbol — otherwise the symbol
  boundary produces a discontinuity artifact.
- Bin edges vs. bin centers differ between MATLAB's `hist` and numpy's
  `histogram`; they call this out explicitly as a porting hazard. A half-bin
  offset here is a silent voltage-axis error.

---

## Patterns to avoid

### 1. `nothing()` dynamic attribute bags

Results are stored on empty class instances with attributes attached at runtime,
keyed by strings like `'c000'`, `'trans01'`, `'pre1'`. This is MATLAB struct
emulation, and it costs type checking, editor support, and any ability to know
what a stage produced without executing it.

**Use dataclasses.** Our `architecture.md` §4 already specifies result dataclasses
per engine; hold that line.

### 2. A real aliasing bug in the settings defaults

```python
@dataclass
class simulationSettings:
    general: generalSettings = generalSettings()      # evaluated ONCE at class definition
```

Python's dataclass machinery rejects mutable `list`/`dict`/`set` defaults but
permits arbitrary class instances — so every `simulationSettings()` shares one
`generalSettings` object. Since `adaptLink()` mutates settings in the
optimization loop, two settings objects in one process would alias. The same file
uses `field(default_factory=...)` correctly for a list, so the mechanism was
known but not applied here.

**For us:** `LinkConfig` is `frozen=True` with scalar fields, so we are safe
today. The moment it gains a nested block (`CTLEConfig`, `DFEConfig`), it must use
`field(default_factory=...)`.

### 3. Experiment selection by editing an import

`statopt.py` line 1 is `from generateUserSettingsExample0 import generateUserSettings`.
Switching experiments means editing source. Our `scripts/plot_channel.py` already
takes the target from `argv`; keep that.

### 4. Flat root namespace, and MATLAB porting residue

Twenty modules at repo root with no package, plus `.mat` files and
`loadMatlabFiles.py`. Our `src/` + `scripts/` split is better; no change needed.

---

## What it does *not* give us

- **Margin at a target BER is not implemented.** `generateBER` produces eye
  contours and bathtub curves, then stops and "delegates interpretation to
  downstream processing." Our criterion 2 requires eye height and width at
  BER 1e-12 specifically, so that extraction is ours to write regardless.
- **No time-domain engine.** StatOpt is statistical only. Our criterion 3
  cross-check between the two methods has no counterpart here.
- **No CDR.** Jitter enters as a distribution convolved into the PDF; there is no
  loop, no phase detector, no lock point. Everything in `cdr_notes.md` is outside
  its scope, including the `h₁ = h₋₁` lock condition that determines our sampling
  phase.

## Incidental find

`channels/` carries a full IEEE 802.3ck C2M set — THRU plus three FEXT and four
NEXT aggressors (`C2M__Z100_IL14_WC_BOR_H_L_H_*.s4p`), same family as our
backplane file. `spec.md` §7 flags ignoring crosstalk as the assumption that makes
our margins optimistic; this is a ready source of aggressor files if we ever
decide to quantify that.
