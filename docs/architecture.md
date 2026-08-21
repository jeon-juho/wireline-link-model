# Model Architecture

How the link model in [`spec.md`](spec.md) decomposes into Python modules.
Status: **draft** — module boundaries are proposed, no code written yet.

## 1. Organizing principle

The model is split at the **sampler**. Everything before it is continuous-time,
represented on an oversampled uniform grid; everything after it is discrete and
lives at the symbol rate.

```
        analog / oversampled  (fs = M x Rs)      |   discrete / baud-rate
                                                 |
  PRBS --> TX --> channel --> CTLE ------------->|--> DFE --> slicer --> bits
           |        (s4p)     (H(s))       ^     |     |         |
           |                               |     |     |         v
           |                          sample     |     +----- decisions
           |                            phase    |               |
           |                               |     |               v
           +-------------------------------+-----+----------- CDR (BBPD)
```

Two consequences that drive the module layout:

- **Analog blocks are LTI.** Channel and CTLE compose by multiplying frequency
  responses. Their combined impulse response is the single shared artifact that
  both simulation methods consume — computed once, reused by both.
- **DFE, slicer, and CDR are not LTI.** They involve decisions and feedback, so
  they are simulated sequentially and cannot be folded into the frequency-domain
  chain.

## 2. Modules (`src/`)

| Module           | Responsibility                                                            | Key dependencies      |
|------------------|---------------------------------------------------------------------------|-----------------------|
| `params.py`      | `Bounded` — a parameter carrying its own admissible range, so a sweep reads its search space off the config | numpy |
| `config.py`      | `LinkConfig` and the nested `CTLEConfig` / `DFEConfig` / `CDRConfig` blocks: all parameters in SI units, one object threaded through everything | `params` |
| `configs/`       | One module per experiment, each returning a `LinkConfig`; selected by name at run time | `config` |
| `tx.py`          | PRBS generation, NRZ symbol mapping, upsampling to the oversampled grid    | numpy                 |
| `channel.py`     | Load `.s4p`, extract mixed-mode Sdd21, enforce causality/passivity, produce frequency response | scikit-rf |
| `ctle.py`        | CTLE transfer function as poles/zeros; frequency response and impulse response | control, scipy    |
| `analog.py`      | Compose channel x CTLE into one frequency response; convert to impulse and pulse response | numpy, scipy |
| `dfe.py`         | Symbol-rate decision feedback: tap application, slicing, optional adaptation | numpy               |
| `cdr.py`         | Bang-bang phase detector, loop filter, phase accumulator; linearized reference model | control, numpy |
| `sim_time.py`    | Bit-by-bit transient engine — drives tx to analog to dfe/cdr over a PRBS sequence | all of the above |
| `sim_stat.py`    | Statistical engine — pulse response cursors to ISI distribution to BER/bathtub | numpy, scipy      |
| `metrics.py`     | Eye diagram construction, eye height/width extraction, bathtub curves, margin at a target BER | numpy |
| `plotting.py`    | All matplotlib output; writes to `results/`                                | matplotlib            |

`sim_time.py` and `sim_stat.py` are **engines**, not entry points — they expose
functions that take a `LinkConfig` and return results. The runnable drivers that
call them live in `scripts/`.

## 3. Data flow and shared artifacts

```
  LinkConfig
      |
      +--> channel.load() ---+
      |                      +--> analog.compose() --> H(f) --> h(t) --> pulse response p(t)
      +--> ctle.transfer() --+                                   |            |
      |                                                          |            |
      |                                       +------------------+            |
      |                                       v                               v
      +--> tx.prbs() ----------------> sim_time.run()                  sim_stat.run()
      |                                       |                               |
      |                                       v                               v
      +------------------------------> waveforms, eye                   BER, bathtub
                                              |                               |
                                              +----------> metrics <----------+
                                                              |
                                                              v
                                                     plotting --> results/
```

The **pulse response** `p(t)` — the analog chain's response to a single UI-wide
pulse — is the pivot. Its cursors, `p` sampled at multiples of the UI at the CDR
phase, give the DFE its tap targets and give the statistical engine its ISI
terms. Both engines must derive it from the same `analog.compose()` output, or
criterion 3 in the spec becomes meaningless.

## 4. Module interfaces (proposed)

Signatures are indicative, not final. All quantities SI: `f` in Hz, `t` in s,
voltages in V.

| Module      | Proposed entry points                                                  |
|-------------|------------------------------------------------------------------------|
| `channel`   | `load(path) -> skrf.Network`; `differential_response(ntwk) -> (f, sdd21)` |
| `ctle`      | `transfer_function(cfg) -> control.TransferFunction`; `response(tf, f) -> H` |
| `analog`    | `compose(*responses) -> H`; `impulse_response(f, H, cfg) -> (t, h)`; `pulse_response(h, cfg) -> p`; `split_pulse(p, cfg) -> SplitPulse`; `mm_lock_phase(sp) -> (j, phase_ui)`; `peak_distortion(sp, j, n_taps) -> Distortion`; `eye_vs_phase(sp, n_taps) -> np.ndarray` |
| `dfe`       | `run(x, taps) -> (decisions, y)` — tap values come from `SplitPulse` |
| `cdr`       | `bbpd(samples, decisions) -> early_late`; `loop(cfg) -> control.TransferFunction`; `run(...) -> phase_trajectory` |
| `sim_time`  | `run(cfg) -> TimeResult`                                               |
| `sim_stat`  | `run(cfg) -> StatResult`                                               |
| `metrics`   | `eye(waveform, cfg) -> EyeData`; `eye_height(...)`, `eye_width(...)`, `bathtub(...)`, `margin_at_ber(..., ber=1e-12)` |

Engines return result dataclasses rather than tuples, so a driver script can pass
one object to `metrics` and `plotting`.

## 5. Scripts (`scripts/`)

| Script                    | Purpose                                                       | Spec criterion |
|---------------------------|---------------------------------------------------------------|----------------|
| `plot_channel.py`         | Sanity-check the `.s4p`: insertion loss, impulse, pulse response | prerequisite |
| `run_eye_compare.py`      | Eye diagram with EQ off vs. on                                | 1              |
| `run_ber_margin.py`       | Statistical run; eye margin at BER 1e-12                      | 2              |
| `run_crosscheck.py`       | Per-block simulated vs. theoretical comparison                | 3              |

Script count and naming are TBD; this is the minimum set that covers the three
success criteria.

## 6. Cross-check strategy (criterion 3)

Each block has a closed form to check against. `run_crosscheck.py` exercises all
of them.

| Block   | Simulated                                  | Theoretical reference                                   |
|---------|--------------------------------------------|---------------------------------------------------------|
| Channel | FFT of the extracted impulse response      | Sdd21 read directly from the Touchstone file            |
| CTLE    | FFT of the CTLE impulse response           | Pole/zero transfer function evaluated at the same `f`   |
| DFE     | Residual post-cursor ISI after tap subtraction | Hand-computed pulse-response cursors minus tap weights |
| CDR     | Measured jitter transfer from a phase step or sinusoidal input | Linearized 2nd-order loop response from `control` |

The CDR check is the weakest of the four — a bang-bang PD has no fixed linear
gain, so the linearized reference is only valid in a limited regime. Tolerances
per block are TBD.

## 7. Conventions

- **Units.** SI everywhere internally, per CLAUDE.md. `ui` is a duration in
  seconds, not a count. Conversion to ps/GHz/mV happens only in `plotting.py`.
- **Sampling grid.** One oversampling factor `M` (samples per UI) defined in
  `LinkConfig`, used by every analog-domain module, with `fs = M x Rs`. `M` is
  TBD.
- **Frequency grid.** The `.s4p` frequency points determine the native grid;
  resampling to a uniform DC-inclusive grid happens once, in `channel.py`.
- **No global state.** `LinkConfig` is passed explicitly; modules hold no
  module-level mutable state.
- **Results.** Everything written to `results/`; nothing is written from inside
  `src/` except through `plotting.py`.

## 8. Unverified assumptions

1. Channel and CTLE can be composed by multiplication in the frequency domain —
   assumes the CTLE presents an ideal load and does not interact with channel
   termination.
2. ~~The `.s4p` extends low enough in frequency and far enough beyond Nyquist
   that the impulse response can be obtained without significant extrapolation
   error.~~ **Verified.** The file spans 0–50 GHz on a uniform 20 MHz grid with
   DC present, so no low-end extrapolation is needed, and Sdd21 is below −45 dB
   at the 50 GHz truncation. The DC self-test (`sum(h) == Sdd21(0)`) holds to
   2e-16.
3. DC extrapolation and enforced causality do not materially distort the pulse
   response cursors.
4. A single fixed oversampling factor `M` serves both the analog chain and CDR
   phase resolution adequately.
5. The DFE sees the CTLE output sampled at one phase per UI — baud-rate
   operation, with no fractional-UI processing.
6. Tap weights and CDR phase can be treated as converged when extracting margin;
   adaptation transients are excluded from the reported numbers.
7. The pulse response computed once from the LTI chain remains valid inside the
   time-domain engine, where DFE feedback makes the overall system nonlinear.
   True for the forward path only.
8. Splitting the model at the sampler assumes the slicer is ideal and
   instantaneous, so no analog behavior leaks past the sampling instant.
