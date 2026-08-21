# DFE Theory Notes — Post-Cursor Cancellation and Tap Weight Derivation

Background derivation for the DFE block specified in [`spec.md`](spec.md) §4.2, and the source of
the hand-calculation formulas used in the cross-check matrix (§8, rows X5 / X6). Companion to
[`ctle_notes.md`](ctle_notes.md).

> **Scope note.** Theory only. This note does **not** resolve any `TBD` in `spec.md`; the numbers
> in §2 and §8 are illustrative. The real `h_m` values come from testbench TB2 once the channel
> file (`TBD-01`) is chosen.

---

## 1. The pulse response is a complete description of the channel

Channel + CTLE is linear and time-invariant, so the single-bit response `p(t)` determines the
response to any data pattern. With data `a_k ∈ {+1, −1}` and symbol period `T = 1 UI = 62.5 ps`:

```
y(t) = Σ_k a_k · p(t − kT)                    (superposition)
```

Sampling once per UI at phase `t_s` defines the **sampled channel coefficients**:

```
h_m ≜ p(t_s + m·T),        m = …, −1, 0, +1, +2, …
```

so that

```
y[n] = a_n·h₀  +  Σ_{m≥1} a_{n−m}·h_m  +  Σ_{m≤−1} a_{n−m}·h_m  +  ν[n]
       └cursor┘   └── post-cursor ISI ──┘   └── pre-cursor ISI ──┘   └noise┘
```

Two facts carry the entire DFE argument:

1. `h_m` is a **deterministic property of the channel** — fixed, independent of the data.
2. ISI looks random only because the multiplying symbols `a_{n−m}` are random.

## 2. How ISI closes the eye — peak distortion analysis

For `a_n = +1` with all other symbols chosen adversarially:

```
y[n]|min = h₀ − Σ_{m≠0} |h_m|

worst-case eye height:   EH_wc = 2·( h₀ − Σ_{m≠0} |h_m| )
```

The eye is fully closed once `Σ_{m≠0}|h_m| ≥ h₀`. A representative post-CTLE set on a −20 dB
channel, and the effect of cancelling post-cursors:

| Case | `h₋₁` | `h₀` | `h₁` | `h₂` | `h₃` | residual `Σ\|ISI\|` | `EH_wc` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No DFE | 20 | 100 | 55 | 20 | 8 | 103 mV | closed (< 0) |
| **1-tap** (`h₁` cancelled) | 20 | 100 | — | 20 | 8 | 48 mV | **104 mV** |
| 2-tap (`h₁, h₂` cancelled) | 20 | 100 | — | — | 8 | 28 mV | 144 mV |

*(All values in mV; illustrative.)*

## 3. The 1-tap DFE principle — do not estimate what is already known

The bit at index `n−1` has **already been decided**. If that decision is correct then
`d[n−1] = a_{n−1}`, and since `h₁` is a known channel constant, the post-cursor contribution
`a_{n−1}·h₁` is **not random at all — it is fully known**. So subtract it before slicing:

```
z[n] = y[n] − c₁·d[n−1]

     = a_n·h₀ + (h₁ − c₁)·a_{n−1} + Σ_{m≥2} a_{n−m}h_m + Σ_{pre} + ν[n]
                └─────┬─────┘
                 exactly 0 when c₁ = h₁
```

This is the qualitative difference from the CTLE. A CTLE *reshapes* the frequency response to
*reduce* ISI; a DFE **algebraically cancels** the term. It is an exact subtraction, not an
approximation.

## 4. Why there is no noise enhancement

The feedback path carries a **decided ±1 digital value, not the noisy analog waveform**. Therefore

```
z[n] = a_n·h₀ + (residual ISI) + ν[n]        ← ν[n] passes through unchanged
```

A linear equalizer implementing `H_eq ≈ 1/H_ch` necessarily multiplies the noise by `|1/H_ch|` at
high frequency along with the signal:

| | numerator (eye) | denominator (σ) | net SNR |
| --- | --- | --- | --- |
| CTLE / linear EQ | ↑ | ↑ | limited improvement |
| **DFE** | ↑ | **unchanged** | direct improvement |

This is why the receiver does not simply keep increasing CTLE boost past ~20–25 dB (see
`ctle_notes.md` §4) and adds a DFE instead. The price is not noise — it is **error propagation**
(§7).

## 5. Determining the tap weight

### 5.1 Zero-forcing (ZF)

By definition, the value that nulls the residual post-cursor:

```
c₁ = h₁          [V]   the voltage actually subtracted
w₁ = h₁ / h₀     [–]   the cursor-normalized tap coefficient
```

> **Normalization convention.** `spec.md` §4.2 and cross-check rows X5/X6 use the normalized form
> `w₁ = h₁/h₀`. The summer then subtracts `w₁·h₀·d[n−1] = h₁·d[n−1]`, so the two statements agree.
> Getting this wrong in `models/dfe.va` misplaces the tap by a factor of `h₀`.

### 5.2 MMSE — and why it collapses onto the ZF solution

Minimize the mean-square error, assuming correct decisions (`d = a`) and i.i.d. symbols
(`E[a_i a_j] = δ_ij`):

```
J(c₁) = E[ (z[n] − a_n·h₀)² ]
      = (h₁ − c₁)² + Σ_{m≥2} h_m² + Σ_{pre} h_m² + σ²

dJ/dc₁ = −2(h₁ − c₁) = 0    →    c₁ = h₁
```

The important structural observation: **the `σ²` term does not depend on `c₁`.** There is no
ISI-versus-noise trade-off on a feedback tap, so

```
ZF solution  ≡  MMSE solution           (feedback taps only)
```

This is the opposite of the linear case. For a feedforward equalizer, ZF amplifies noise, so
MMSE ≠ ZF and MMSE always wins. In a DFE, **all the MMSE trade-off lives in the feedforward
section (CTLE/FFE); the feedback taps are always just the post-cursors of the combined
channel + FFE response.** That is the theoretical justification for "set the tap to `h₁/h₀` and
you are done".

### 5.3 Adaptive (sign-sign LMS)

When `h₁` is not known a priori, descend the error gradient:

```
e[n] = z[n] − h₀·d[n]                        ← requires an error slicer at threshold ±h₀
w₁[n+1] = w₁[n] + μ·sgn(e[n])·sgn(d[n−1])
```

At convergence `E[e[n]·d[n−1]] = 0`, i.e. `h₁ − c₁ = 0` — the same answer as §5.1. This is the
`TBD-19` stretch goal in `spec.md`.

## 6. The tap depends on sampling phase — coupling to the CDR

Since `h_m = p(t_s + mT)`, **the tap value is a function of where the CDR locks.** The dependence
runs both ways. Taking the expectation of the Mueller-Müller phase detector output:

```
E[ e_MM[n] ] = E[ d[n−1]·y[n] ] − E[ d[n]·y[n−1] ]

E[ a_{n−1}·Σ_k a_k h_{n−k} ]   = h₁
E[ a_n·Σ_k a_k h_{n−1−k} ]     = h₋₁

     →   E[ e_MM[n] ] = h₁ − h₋₁
```

So **an MM CDR locks at the phase where `h₁ = h₋₁`** — the point that balances pre- and
post-cursor.

This creates a real design interaction. If the PD observes the signal **before** DFE subtraction it
settles at `h₁ = h₋₁` as above. If it observes the **post-DFE** signal, where `h₁` has already been
cancelled, the loop instead drives toward `h₋₁ = 0` and the lock point shifts. The tap point is
therefore a design decision, not a detail — tracked as `TBD-36` in `spec.md` §4.3 and to be
confirmed in TB4.

## 7. Limits of a single tap

| Limitation | Detail | Spec reference |
| --- | --- | --- |
| Only `h₁` is cancelled | `h₂` and beyond survive; add a second tap when `\|h₂\|/h₀` exceeds the margin budget | `TBD-14` |
| Error propagation | A wrong `d[n−1]` subtracts `+h₁` instead of `−h₁`, so the disturbance doubles to `2h₁` and can burst. The penalty stays small for roughly `\|w₁\| ≲ 0.5` | §4.2 |
| Pre-cursor untouchable | Only past decisions are available, so `h₋₁` is structurally out of reach — it belongs to the CTLE or a TX FFE | §1.2 |
| Worst timing path in the RX | `y[n] → slice → feed back → subtract` must close within **1 UI = 62.5 ps**. Real silicon loop-unrolls (speculates) the first tap for this reason; v1 assumes ideal zero-delay feedback | `TBD-18` |

## 8. Extraction procedure (testbench TB2)

```
1) Disable the DFE, inject a single 1 UI pulse, capture v_sum(t)
2) Locate the peak → choose the sampling phase t_s
   (strictly the CDR lock phase; first pass t_s = t_peak)
3) h₀ = p(t_s), h₁ = p(t_s+T), h₂ = p(t_s+2T), h₋₁ = p(t_s−T)
4) w₁ = h₁ / h₀                                        → spec §8 X5
5) Enable the DFE, re-measure: h₁ − w₁·h₀ ≈ 0           → spec §8 X6
6) If |h₂|/h₀ < 0.05 one tap suffices, else n_tap = 2   → TBD-14
7) Predicted eye height = 2·( h₀ − Σ_{m≥2}|h_m| − |h₋₁| )
   compare against the measured TB3 eye
```

Steps 3–6 are what `scripts/pulse_to_dfe_taps.py` (spec §6.1) automates.

## 9. Formula summary

| Quantity | Expression |
| --- | --- |
| Sampled channel coefficient | `h_m = p(t_s + m·T)` |
| Received sample | `y[n] = a_n h₀ + Σ_{m≠0} a_{n−m} h_m + ν[n]` |
| Worst-case eye height (no DFE) | `EH_wc = 2·(h₀ − Σ_{m≠0}\|h_m\|)` |
| DFE output | `z[n] = y[n] − Σ_{m=1..N} c_m·d[n−m]` |
| ZF tap (absolute) | `c_m = h_m` |
| ZF tap (normalized) | `w_m = h_m/h₀` |
| MMSE cost | `J = Σ_m (h_m − c_m)² + Σ_{pre} h_m² + σ²` → minimized at `c_m = h_m` |
| Sign-sign LMS update | `w₁[n+1] = w₁[n] + μ·sgn(e[n])·sgn(d[n−1])`, `e[n] = z[n] − h₀d[n]` |
| Residual eye height (1-tap) | `EH = 2·(h₀ − Σ_{m≥2}\|h_m\| − Σ_{pre}\|h_m\|)` |
| MM PD mean output | `E[e_MM] = h₁ − h₋₁` (lock point `h₁ = h₋₁`) |
