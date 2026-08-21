# CTLE Theory Notes — Transfer Function, Boost, and Peaking Frequency

Background derivation for the CTLE block specified in [`spec.md`](spec.md) §4.1, and the source of
the hand-calculation formulas used in the cross-check matrix (§8, rows X2a / X2b).

> **Scope note.** This is a theory note only. `TBD-11` and `TBD-12` remain open until they are
> confirmed against measurement. §7 walks the procedure through a generic 20 dB example; §7.1
> applies it to the channel actually selected for this project (`TBD-01`, now resolved — 13.2 dB
> at Nyquist).

---

## 1. Why the channel attenuates

Attenuation in a backplane trace is the sum of two physical mechanisms:

```
α(f) ≈ a·√f    skin effect — skin depth δ = 1/√(π·f·μ·σ), so the
                effective conductor cross-section shrinks as √f
     + b·f     dielectric loss — proportional to the loss tangent tanδ

IL(f) [dB] ≈ −(a·√f + b·f) · ℓ          ℓ = trace length
```

The key property is that **loss in dB increases monotonically with frequency**. The channel is a
low-pass filter: at −20 dB @ 8 GHz the Nyquist-rate content arrives at 1/10 of its launched
amplitude while DC arrives essentially intact.

In the time domain this is **pulse spreading**. A 1 UI (62.5 ps) pulse emerges with a tail lasting
several UI, and that tail lands on the sampling instants of neighbouring bits:

```
y[n] = h₀·d[n] + h₁·d[n−1] + h₂·d[n−2] + … + h₋₁·d[n+1] + …
        ↑cursor  └──────── post-cursor ISI ────────┘   └ pre-cursor ISI
```

`h₀` is signal; everything else is intersymbol interference. On a −20 dB channel `h₁/h₀` typically
reaches 0.3–0.5, which closes the eye completely.

---

## 2. Equalization = inverse filtering

The objective is a flat cascaded response up to Nyquist:

```
|H_ctle(f)| · |H_ch(f)| ≈ const        for 0 ≤ f ≤ f_N

in dB:   G_ctle(f) [dB] = −IL_ch(f) [dB] + C
```

**Why a broadband amplifier does not work.** Flat gain scales signal, noise, and ISI by the same
factor and leaves `h₁/h₀` unchanged. Only a frequency-dependent *slope* removes ISI. The CTLE is
therefore a high-pass shaping network — and, as §4 shows, circuit-wise it achieves that slope by
**attenuating low frequencies**, not by creating high-frequency gain.

---

## 3. Transfer function of a source-degenerated differential pair

The standard CTLE implementation degenerates the tail of a differential pair with `R_S ∥ C_S`.
In the half-circuit each side sees `R_S/2` in parallel with `2C_S`:

```
Z_S(s) = (R_S/2) / (1 + s·R_S·C_S)          degeneration impedance
Z_L(s) = R_D / (1 + s·R_D·C_L)              load impedance

H(s) = − gm·Z_L(s) / (1 + gm·Z_S(s))
```

Expanding gives the canonical one-zero / two-pole form:

```
              gm·R_D              1 + s/ω_z
H(s) = − ─────────────────  ·  ──────────────────────────────
          1 + gm·R_S/2         (1 + s/ω_p1)(1 + s/ω_p2)
```

| Term | Expression | Physical origin |
| --- | --- | --- |
| `A_dc` | `gm·R_D / (1 + gm·R_S/2)` | DC gain **reduced** by degeneration |
| `ω_z` | `1 / (R_S·C_S)` | frequency at which `C_S` starts shorting out `R_S` |
| `ω_p1` | `ω_z · (1 + gm·R_S/2)` | degeneration fully bypassed; gain saturates at `gm·R_D` |
| `ω_p2` | `1 / (R_D·C_L)` | output-node bandwidth set by load parasitics |
| `A_hf` | `gm·R_D` | the un-degenerated gain, recovered between `ω_p1` and `ω_p2` |

Between `ω_z` and `ω_p1` the magnitude rises at **+20 dB/decade**.

---

## 4. What "boost" physically means

Directly from the pole and zero expressions:

```
ω_p1 / ω_z = 1 + gm·R_S/2  ≡  degeneration factor

Boost [dB] = 20·log₁₀(ω_p1/ω_z) = 20·log₁₀(1 + gm·R_S/2)
```

**The boost equals exactly the amount of DC gain that degeneration threw away.** At high frequency
`C_S` shorts `R_S` and the stage returns to its intrinsic gain `gm·R_D`. The CTLE does not
manufacture high-frequency gain; it trades away low-frequency gain to create slope.

Two consequences follow, and they bound how far a CTLE can be pushed:

- **Absolute amplitude cost.** 20 dB of boost means discarding a factor of 10 of DC gain. The
  signal reaching the slicer is smaller relative to downstream offset and noise, so more stages are
  needed — costing power, area, and offset.
- **Noise and crosstalk enhancement.** Being a linear filter, the CTLE amplifies high-frequency
  *noise and crosstalk* by the same factor as the signal. The eye opens, but slicer SNR does not
  improve proportionally. This is the fundamental reason a CTLE alone is not used beyond roughly
  20–25 dB, and why a DFE is added: a DFE subtracts already-decided bits, so it cancels post-cursor
  ISI with **no noise enhancement** (at the cost of error propagation).

---

## 5. Peaking frequency — exact derivation

Let `Z = ω_z²`, `P₁ = ω_p1²`, `P₂ = ω_p2²`, `u = ω²`. Then

```
|H(jω)|² = A_dc² · (1 + u/Z) / [ (1 + u/P₁)(1 + u/P₂) ]
```

Maximize the log-magnitude:

```
d/du [ ln(Z+u) − ln(P₁+u) − ln(P₂+u) ] = 0

        1/(Z+u) − 1/(P₁+u) − 1/(P₂+u) = 0
```

Clearing denominators:

```
(P₁+u)(P₂+u) = (Z+u)(P₂+u) + (Z+u)(P₁+u)

u² + u(P₁+P₂) + P₁P₂ = 2u² + u(2Z+P₁+P₂) + Z(P₁+P₂)

        u² + 2Z·u + Z(P₁+P₂) − P₁P₂ = 0
```

Taking the positive root and using `(P₁−Z)(P₂−Z) = P₁P₂ − Z(P₁+P₂) + Z²`:

```
┌──────────────────────────────────────────────────────────┐
│  ω_peak = √[ √((ω_p1² − ω_z²)(ω_p2² − ω_z²)) − ω_z² ]     │   exact
│                                                          │
│  ω_peak ≈ √(ω_p1 · ω_p2)        when  ω_z ≪ ω_p1          │   geometric
└──────────────────────────────────────────────────────────┘     mean of poles
```

**Physical reading.** The +20 dB/dec rise stops at `ω_p1` and the −20 dB/dec roll-off begins at
`ω_p2`, so the maximum sits between them — at their geometric mean on a log axis. The zero `ω_z`
governs the *height* of the peak, not its *location*.

Numerical check with `f_z = 1`, `f_p1 = 4`, `f_p2 = 12` GHz:

| Formula | Result |
| --- | --- |
| Exact expression above | **6.73 GHz** |
| `√(f_p1·f_p2)` (geometric mean of poles) | 6.93 GHz ✓ |
| `√(f_z·f_p1)` | 2.00 GHz ✗ — not the peak |

---

## 6. Peak gain — and why the asymptotic formula overstates it

The asymptotic boost `20·log₁₀(f_p1/f_z)` assumes the mid-band plateau at `gm·R_D` is actually
reached, which requires `f_p2 ≫ f_p1`. When `f_p2` is close, the roll-off eats into the peak. The
exact peak gain relative to DC is:

```
|H(ω_peak)|        √(1 + ω_peak²/ω_z²)
───────────  =  ──────────────────────────────────────────────
    A_dc        √(1 + ω_peak²/ω_p1²) · √(1 + ω_peak²/ω_p2²)
```

Sensitivity, holding the asymptotic ratio at `f_p1/f_z = 10` (i.e. nominally 20 dB) and
`f_peak ≈ 8 GHz`:

| `f_p2/f_p1` | `f_p1` [GHz] | `f_p2` [GHz] | `f_z` [GHz] | Actual peak [dB] | Deficit |
| --- | --- | --- | --- | --- | --- |
| 2 | 5.66 | 11.3 | 0.566 | 16.5 | −3.5 dB |
| 3 | 4.62 | 13.9 | 0.462 | 17.5 | −2.5 dB |
| 5 | 3.58 | 17.9 | 0.358 | 18.4 | −1.6 dB |
| ∞ | — | — | — | 20.0 | 0 |

`f_p2 = 1/(R_D·C_L)` is fixed by the output node and cannot be pushed arbitrarily far out, so
**a real design must add 2–3 dB of margin to the asymptotic boost** to land the measured peak on
target. This is exactly why `spec.md` §8 splits the cross-check into X2a (frequency) and X2b
(gain, with the exact expression).

---

## 7. Design procedure — worked example at 16 Gb/s

Target: `f_N = 8 GHz`, channel insertion loss 20 dB at Nyquist. *(Generic example — the numbers for
this project's actual channel are in §7.1.)*

```
1) Place the peak at Nyquist:      f_peak ≈ f_N = 8 GHz
                                   →  f_p1 · f_p2 ≈ 64 GHz²

2) f_p2 is set by output bandwidth. Assume f_p2 = 3·f_p1:
                                   →  f_p1 = 4.62 GHz,  f_p2 = 13.9 GHz

3) Boost target = |IL(f_N)| = 20 dB.
   Asymptotic f_p1/f_z = 10 would give only 17.5 dB (table §6),
   so add the 2.5 dB deficit:      f_p1/f_z ≈ 13.3
                                   →  f_z = 0.35 GHz

4) Verify with the exact expressions:
                                   f_peak = 7.98 GHz   ✓
                                   peak gain = 20.0 dB ✓

5) Map back to the circuit:
   gm·R_S/2 = f_p1/f_z − 1 = 12.3
   A_dc     = gm·R_D / 13.3
   R_S·C_S  = 1/(2π·f_z)  = 459 ps
   R_D·C_L  = 1/(2π·f_p2) = 11.5 ps
```

Step 3 is the one that is easy to get wrong: using the asymptotic formula alone leaves the measured
peaking 2.5 dB short of target.

### 7.1 Applied to the selected channel

`spec.md` §3 fixes the channel at **−13.19 dB at 8 GHz** (`TBD-01`, `TBD-39`). Running the same
five steps with a 13.2 dB boost target:

```
1) f_peak ≈ f_N = 8 GHz            →  f_p1 · f_p2 ≈ 64 GHz²
2) f_p2 = 3·f_p1                   →  f_p1 = 4.62 GHz,  f_p2 = 13.9 GHz
3) boost target = 13.2 dB.
   Asymptotic f_p1/f_z = 4.57 gives only 10.8 dB measured (deficit 2.4 dB),
   so use asymptotic ≈ 15.6 dB    →  f_p1/f_z = 6.03  →  f_z = 0.77 GHz
4) Verify exactly:                    f_peak = 7.90 GHz   ✓
                                      peak gain = 13.1 dB ✓
5) Circuit mapping:
   gm·R_S/2 = f_p1/f_z − 1 = 5.03
   A_dc     = gm·R_D / 6.03
   R_S·C_S  = 1/(2π·f_z)  = 208 ps
   R_D·C_L  = 1/(2π·f_p2) = 11.5 ps
```

Two things change relative to the 20 dB example:

| | 20 dB example | This channel (13.2 dB) |
| --- | --- | --- |
| `f_z` | 0.35 GHz | **0.77 GHz** — zero moves out, boost band is narrower |
| `gm·R_S/2` | 12.3 | **5.03** — much less degeneration |
| DC gain sacrificed | ÷13.3 | **÷6.03** — roughly half the amplitude penalty |
| Stage count | 2 likely | **1 sufficient** (`TBD-10`) |

The lighter degeneration is the practical payoff of the relaxed target: less DC gain is thrown
away (§4), so less high-frequency noise and crosstalk enhancement, and one stage does the job.

---

## 8. Time-domain view — the zero is a differentiator

Transforming the zero term back to the time domain:

```
y(t) = A · [ x(t) + (1/ω_z)·dx/dt ]
```

The derivative term is large at transitions and zero on flat runs, so it sharpens edges and
cancels the slow tail the channel added.

The sharper statement is **pole-zero cancellation**. Modelling the channel's dominant low-frequency
behaviour as a pole at `ω_ch`:

```
H_ch(s)·H_ctle(s) ≈ [ 1/(1 + s/ω_ch) ] · [ (1 + s/ω_z) / (…) ]
```

Choosing `ω_z = ω_ch` cancels the slow pole outright, and the long tail of the pulse response —
i.e. the post-cursors `h₁, h₂` — disappears. Flattening the magnitude response and removing ISI are
the same operation viewed in two domains.

---

## 9. Limits — why CTLE alone is not enough

A single CTLE stage offers exactly one slope, +20 dB/dec between `f_z` and `f_p1`. Channel loss goes
as `a√f + b·f`, which is not a straight line on a dB-vs-log-f plot. Hence:

| Limitation | Consequence | Spec reference |
| --- | --- | --- |
| One slope cannot match `√f + f` over a wide band | Cascade 2 stages (~10 dB each) for two degrees of freedom | `TBD-10` |
| Linear boost amplifies noise and crosstalk equally | Stop around 20–25 dB; cancel the residual post-cursors with a DFE | §4.2 |
| DFE only uses past decisions | Pre-cursor `h₋₁` must be handled by the CTLE (or TX FFE) | §1.2 |
| Finite `f_p2` caps achievable peaking | Add boost margin, or add a stage | §6 above |

This division of labour — CTLE for broadband slope and pre-cursor, DFE for the remaining
post-cursors without noise penalty — is the reason the receiver in `spec.md` uses both.

---

## 10. Formula summary

| Quantity | Expression |
| --- | --- |
| Transfer function | `H(s) = A_dc·(1 + s/ω_z) / [(1 + s/ω_p1)(1 + s/ω_p2)]` |
| DC gain | `A_dc = gm·R_D / (1 + gm·R_S/2)` |
| Zero | `ω_z = 1/(R_S·C_S)` |
| First pole | `ω_p1 = ω_z·(1 + gm·R_S/2)` |
| Second pole | `ω_p2 = 1/(R_D·C_L)` |
| Asymptotic boost | `20·log₁₀(ω_p1/ω_z) = 20·log₁₀(1 + gm·R_S/2)` |
| Peaking frequency (exact) | `ω_peak = √[√((ω_p1²−ω_z²)(ω_p2²−ω_z²)) − ω_z²]` |
| Peaking frequency (approx.) | `ω_peak ≈ √(ω_p1·ω_p2)`, valid for `ω_z ≪ ω_p1` |
| Peak gain (exact) | `\|H(ω_peak)\|/A_dc = √(1+ω_peak²/ω_z²) / [√(1+ω_peak²/ω_p1²)·√(1+ω_peak²/ω_p2²)]` |
| Equalization target | `G_ctle(f)[dB] = −IL_ch(f)[dB] + C`, for `0 ≤ f ≤ f_N` |
