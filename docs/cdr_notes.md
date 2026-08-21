# CDR Theory Notes — BBPD Linearization, Jitter Transfer, Bandwidth and Damping

Background derivation for the CDR block specified in [`spec.md`](spec.md) §4.3, and the source of
the hand-calculation formulas used in the cross-check matrix (§8, rows X7–X10). Companion to
[`ctle_notes.md`](ctle_notes.md) and [`dfe_notes.md`](dfe_notes.md).

> **Scope note.** Theory only. This note does **not** resolve any `TBD` in `spec.md`; the numbers
> in §9 are a worked consistency check on the starting values, not a design commitment.

---

## 1. Why linearization is needed at all

A bang-bang phase detector outputs only `b[n] ∈ {−1, +1}`. A phase error of 0.001 UI and one of
0.4 UI produce the identical output, so strictly speaking the block has no transfer function and
no small-signal gain.

What rescues the analysis is that **random jitter and noise are always present**. That noise
dithers the hard quantizer, so the *average* output over many UI becomes a smooth function of the
phase error. Every result below is therefore a **statistical** linearization, valid for
small-signal behaviour around lock — not an exact description of any single UI.

## 2. Linearized BBPD gain

Let the phase error be `Δτ` and let it be perturbed by zero-mean Gaussian jitter of rms `σ_τ`:

```
E[b] = P(Δτ + n > 0) − P(Δτ + n < 0)
     = 1 − 2·Φ(−Δτ/σ_τ)
     = erf( Δτ / (σ_τ·√2) )
```

Differentiating at the lock point gives the linearized detector gain:

```
              d                2·exp(−Δτ²/2σ_τ²) │              ┌──────────────────┐
K_pd  ≜  ───────  E[b]  =  ───────────────────── │        →     │            2      │
             dΔτ                σ_τ·√(2π)        │Δτ=0          │ K_pd = ────────── │
                                                                │        σ_τ·√(2π)  │
                                                                └──────────────────┘
```

The defining property of a BBPD follows immediately:

> **`K_pd ∝ 1/σ_τ`.** More input jitter *lowers* the loop gain; less jitter *raises* it. The loop
> bandwidth is therefore a function of the operating condition, unlike a linear-PD PLL. Every
> `ω_n` and `ζ` quoted below is valid only for the `σ_τ` it was computed at — the loop is
> amplitude-dependent by construction.

### 2.1 The Mueller-Müller variant

For the sign-sign MM PD the quantizer input is a voltage, so the pulse-response slope enters the
gain. From `dfe_notes.md` §6, `E[e_MM] = h₁ − h₋₁`; differentiating with respect to sampling phase:

```
K_mm = ∂(h₁ − h₋₁)/∂τ = p′(t_s+T) − p′(t_s−T)        [V/s]

K_pd = 2·K_mm / (σ_v·√(2π))  =  2 / (σ_τ·√(2π)),      σ_τ ≜ σ_v / K_mm
```

So **the detector gain is proportional to the slope of the equalized pulse response**. Better
equalization sharpens the edges, raises `K_mm`, and widens the loop — another coupling between the
EQ and the CDR, alongside the lock-point interaction of `TBD-36`.

## 3. Loop architecture and open-loop transfer

Digital second-order loop (frequency register feeding a phase accumulator), updated once per
decimation window of `M` UI (`TBD-37`):

```
f[n] = f[n−1] + K_i·b[n]                  frequency accumulator  (integral path)
φ[n] = φ[n−1] + K_p·b[n] + f[n]           phase accumulator → DCO / PI code
```

In the z-domain:

```
φ(z)/b(z) = K_p/(1−z⁻¹) + K_i/(1−z⁻¹)²
```

Because the loop bandwidth is far below the update rate (16 MHz vs. 16 GHz — a ratio of 1000), the
approximation `1 − z⁻¹ ≈ sT` is very accurate:

```
                  K_p         K_i             (K_pd·K_p/T)·s + K_pd·K_i/T²
L(s) = K_pd·[ ───────  +  ───────── ]  =  ────────────────────────────────
                  sT         (sT)²                        s²
```

The `s²` in the denominator is **two integrators** — the frequency accumulator and the phase
accumulator — which makes this a **type-II** loop.

## 4. Jitter transfer function

```
            φ_out(s)         L(s)              (K_pd·K_p/T)·s + K_pd·K_i/T²
H(s)  ≜  ───────────  =  ───────────  =  ──────────────────────────────────────
            φ_in(s)        1 + L(s)        s² + (K_pd·K_p/T)·s + K_pd·K_i/T²
```

In standard second-order form:

```
┌────────────────────────────────────────────────────────┐
│                 2ζω_n·s + ω_n²                         │
│       H(s)  =  ──────────────────                      │
│                s² + 2ζω_n·s + ω_n²                     │
│                                                        │
│             √(K_pd·K_i)                K_p      K_pd   │
│      ω_n = ─────────────  ,     ζ  =  ─────·√( ────── )│
│                   T                     2        K_i   │
└────────────────────────────────────────────────────────┘
```

Normalizing `K_pd` into the coefficients gives the compact digital-design form:

```
ω_n·T = √K_i           ζ = K_p / (2·√K_i)
```

The surviving `s` term in the numerator — a **zero** at

```
ω_z = ω_n/(2ζ) = K_i/(K_p·T)
```

— is the signature of a type-II loop. It causes two things: `|H|` exceeds unity (**jitter
peaking**), and the high-frequency roll-off is **−20 dB/dec**, not −40.

```
|H| [dB]
  ↑
  │        ╭─╮ ← peaking, caused by the zero
0 ├────────╯ ╰──╮
  │              ╰──╮  −20 dB/dec
  │                 ╰──╮
  └──────────────────────────→ f
        f_z      f_n   f_3dB
```

Note `H(0) = 1`: a CDR is a *tracking* loop. Low-frequency jitter is deliberately passed to the
recovered clock, because jitter common to data and clock produces no sampling error.

## 5. Loop bandwidth — set by the proportional path

Solving `|H(jω)|² = 1/2` with `x = ω/ω_n`, `a = 2ζ`:

```
x⁴ − (2 + a²)·x² − 1 = 0     →     x² = [ (2+a²) + √((2+a²)² + 4) ] / 2
```

For `ζ ≳ 1.5` this collapses to a very useful approximation:

```
┌────────────────────────────────────────────┐
│   ω_3dB ≈ 2ζ·ω_n = K_pd·K_p / (T·M)        │
└────────────────────────────────────────────┘
```

**The loop bandwidth is set by the proportional gain `K_p` alone.** The integral gain `K_i` sets
`ω_n` and the frequency-tracking capability but has almost no influence on bandwidth.

What bandwidth controls:

| Effect | Direction | Note |
| --- | --- | --- |
| Low-frequency jitter tracking | wider is better | `H ≈ 1` below `f_3dB`, so data and clock jitter cancel |
| Jitter tolerance | wider is better | `JTOL ∝ 1/f²` corner moves out — §7 |
| Lock time | wider is faster | a few times `1/f_3dB` — `TBD-25` |
| Dither jitter | wider is worse | `K_p` *is* the phase step — §8 |
| Validity of the continuous-time model | `f_3dB ≲ f_baud/10` | beyond this the z-domain model and stability limits take over |

## 6. Damping factor — set by the ratio, controls peaking

`ζ` does not set bandwidth; it sets **peaking**. Maximizing `|H|²` (`u = (ω/ω_n)²`):

```
|H|² = (1 + a²u) / (u² + (a²−2)u + 1),      a = 2ζ

d|H|²/du = 0   →   a²u² + 2u − 2 = 0

        (ω_peak/ω_n)² = [ √(1 + 8ζ²) − 1 ] / (4ζ²)
```

| `ζ` | `ω_peak/ω_n` | **jitter peaking** | `ω_3dB/ω_n` |
| --- | --- | --- | --- |
| 0.5 | 0.855 | **3.33 dB** | 1.82 |
| 0.707 | 0.786 | 2.09 dB | 2.06 |
| 1.0 | 0.707 | 1.25 dB | 2.48 |
| 2.0 | 0.545 | 0.40 dB | 4.25 |
| 3.0 | 0.458 | **0.20 dB** | 6.17 |

The key structural fact: **in a type-II loop peaking can never be driven to zero — it can only be
reduced by increasing `ζ`.** BBPD CDRs are therefore designed overdamped, typically `ζ = 2…5`,
for two reasons:

1. Standards (OIF-CEI, IEEE 802.3 — `TBD-24`) cap jitter peaking, commonly at 0.1–1 dB, because
   peaking multiplies stage by stage in a repeatered link.
2. `K_pd` varies as `1/σ_τ` (§2). Since `ω_n ∝ √K_pd` and `ζ ∝ √K_pd`, both move with operating
   conditions; starting overdamped keeps the loop clear of limit cycles and slope overload as
   conditions change.

In design terms `ζ = K_p/(2√K_i)` simply says: **keep the proportional path well ahead of the
integral path** (`K_p² ≫ K_i`).

## 7. Error transfer and jitter tolerance

What the slicer actually experiences is the *difference* between data and clock phase:

```
                            s²
E(s) ≜ 1 − H(s) = ──────────────────────           high-pass
                   s² + 2ζω_n·s + ω_n²
```

Tolerable input jitter amplitude is `A_JTOL(f) = φ_margin / |E(j2πf)|`:

```
f ≪ f_n :   |E| ≈ (f/f_n)²   →   JTOL ∝ 1/f²      (+40 dB/dec)
f ≫ f_n :   |E| → 1          →   JTOL = eye margin (flat)

  JTOL [UI]
    ↑
    │╲
    │ ╲  −40 dB/dec
    │  ╲
    │   ╲______________  ← eye margin (spec §7, criterion 2)
    └──────────────────────→ f
            f_n
```

This is the main pressure to *widen* the loop: moving `f_n` right lifts the whole low-frequency
JTOL curve. The opposing pressures are peaking (§6), dither (§8), and the `f_baud/10` stability
ceiling. `spec.md` TB5 measures this curve directly.

## 8. Dither jitter — the intrinsic BBPD cost

A BBPD cannot output zero. In lock with no input jitter, `b[n]` toggles `+1, −1, +1, …` and the
phase moves by `K_p` every update:

```
dither jitter (pk-pk) ≈ K_p   [UI]
```

But §5 gave `ω_3dB ≈ K_pd·K_p/(T·M)`. The same parameter appears in both:

> **Bandwidth and dither are a direct trade-off governed by `K_p`.** Raising `K_p` for
> low-frequency jitter tolerance raises the static dither by exactly the same factor. This is why
> `TBD-22` (bandwidth) and `TBD-23` (phase step / dither) in `spec.md` are really one decision.

## 9. Consistency check on the starting values — and why `M` exists

Using the `spec.md` starting points (`f_LBW = 16 MHz`, `Δφ = UI/64`) with an assumed
`σ_τ = 0.02 UI`:

```
K_pd  = 2/(0.02·√(2π))              = 39.9      [1/UI]
K_p   = 1/64                        = 0.0156    [UI]

f_3dB ≈ K_pd·K_p·f_baud/(2π)
      = 39.9 × 0.0156 × 16e9 / 6.283 ≈ 1.6 GHz          ✗ 100× the target
```

That is also exactly at the `f_baud/10 = 1.6 GHz` stability ceiling. Updating the phase every UI at
the minimum PI step simply does not produce a usable loop.

The standard fix is **decimation / majority voting** of the PD output: accumulate `b[n]` over `M`
UI and update the phase once per window.

```
f_3dB ≈ K_pd·K_p·f_baud / (2π·M)

target 16 MHz   →   M ≈ 100          (typical silicon values are 32…128)
```

`M` is tracked as `TBD-37` in `spec.md` §4.3 and is a parameter of `models/loop_filter.va`. Without
it, `TBD-22` cannot be filled with realizable numbers.

## 10. Formula summary

| Quantity | Expression |
| --- | --- |
| BBPD mean output | `E[b] = erf(Δτ/(σ_τ√2))` |
| Linearized PD gain | `K_pd = 2/(σ_τ·√(2π))` |
| MM slope gain | `K_mm = p′(t_s+T) − p′(t_s−T)`, `σ_τ = σ_v/K_mm` |
| Open loop | `L(s) = [(K_pd K_p/T)s + K_pd K_i/T²] / s²` (type-II) |
| Jitter transfer | `H(s) = (2ζω_n s + ω_n²)/(s² + 2ζω_n s + ω_n²)` |
| Natural frequency | `ω_n = √(K_pd·K_i)/T`, or `ω_n T = √K_i` normalized |
| Damping | `ζ = (K_p/2)·√(K_pd/K_i)`, or `ζ = K_p/(2√K_i)` normalized |
| Loop zero | `ω_z = ω_n/(2ζ) = K_i/(K_p·T)` |
| −3 dB bandwidth | `x⁴ − (2+4ζ²)x² − 1 = 0`; `ω_3dB ≈ 2ζω_n = K_pd K_p/(T·M)` for `ζ ≳ 1.5` |
| Peaking frequency | `(ω_peak/ω_n)² = [√(1+8ζ²) − 1]/(4ζ²)` |
| Error transfer | `E(s) = 1 − H(s) = s²/(s² + 2ζω_n s + ω_n²)` |
| Jitter tolerance | `A_JTOL(f) = φ_margin/\|E(j2πf)\|`, `∝ 1/f²` below `f_n` |
| Dither jitter | `≈ K_p` pk-pk [UI] |
| MM lock point | `E[e_MM] = h₁ − h₋₁ = 0` (see `dfe_notes.md` §6) |
