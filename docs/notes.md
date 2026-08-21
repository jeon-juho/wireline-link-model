# Why This Architecture

A one-page argument for the CTLE + DFE + baud-rate BBPD CDR receiver in
[`spec.md`](spec.md). The derivations live in [`ctle_notes.md`](ctle_notes.md),
[`dfe_notes.md`](dfe_notes.md), and [`cdr_notes.md`](cdr_notes.md); this note is
about *why these three blocks and not others*.

## The problem, measured

The channel is not a design choice — it is the given. From `scripts/plot_channel.py`
on the 802.3ck 16+16 inch backplane:

| | |
|---|---|
| Insertion loss at the 8 GHz Nyquist | 13.19 dB |
| Main cursor `h₀` | 423.8 mV |
| `h₁/h₀`, `h₂/h₀` | 0.376, 0.177 |
| Total ISI, 200 UI depth | 531.1 mV |
| **Worst-case eye** | **−107 mV — closed** |

The eye is closed by more than a quarter of the main cursor. Everything the
receiver does is an attempt to buy back that 107 mV and then some.

## Why not just amplify

Loss rises monotonically with frequency (`a√f + b·f`), so the channel is a
low-pass filter and the damage is pulse spreading. Flat gain multiplies signal,
ISI, and noise by the same constant and leaves `h₁/h₀` exactly where it was.
Only a frequency-dependent *slope* removes ISI. That single observation rules
out the entire class of broadband-amplifier solutions and forces some kind of
filter.

## Why a CTLE first

The CTLE is the cheapest slope available. A source-degenerated pair gives one
zero and two poles, and the +20 dB/dec rise between `ω_z` and `ω_p1` is bought
not by manufacturing high-frequency gain but by *discarding low-frequency gain*
— boost equals exactly the DC gain that degeneration threw away.

Two properties make it the right block to go first:

- **It is the only block that can touch the precursor.** A DFE feeds back
  already-decided bits, so `h₋₁` is structurally out of its reach. Either the
  CTLE handles it or a TX FFE does, and this is an RX study.
- **It reduces `h₁/h₀` before the DFE ever sees it**, which matters because the
  DFE's error-propagation penalty grows with tap weight. Our measured `h₁/h₀ =
  0.376` sits just inside the `≲ 0.5` rule of thumb — uncomfortably close for a
  pre-equalization number.

Its limit is equally clear: being linear, it amplifies high-frequency noise and
crosstalk by the same factor as the signal, which caps useful boost around
20–25 dB. At 13.2 dB this channel lands comfortably inside that budget — one
stage, `gm·R_S/2 ≈ 5`, about half the amplitude penalty of a 20 dB design.

## Why a DFE after it

Where the CTLE *reshapes* to reduce ISI, the DFE *algebraically cancels* it.
The bit at `n−1` is already decided and `h₁` is a known channel constant, so the
postcursor contribution is not random — it is known, and can be subtracted
exactly.

The consequence that justifies the block: the feedback path carries a decided
±1, not the noisy waveform, so **noise passes through unchanged**. The eye opens
without the noise floor rising. This is precisely what a linear equalizer cannot
do, and it is why the answer to "the CTLE isn't enough" is a DFE rather than
more CTLE.

A structural point worth stating plainly: for feedback taps the zero-forcing and
MMSE solutions coincide, because `σ²` does not depend on the tap. There is no
ISI-versus-noise trade-off to tune. All of that trade-off lives in the
feedforward section — which is another way of saying the CTLE and DFE are not
two independent design problems, and the taps are simply the postcursors of the
*combined* channel + CTLE response.

The price is error propagation, and the bound it places on tap weight is the
real reason the CTLE must go first.

## Why baud-rate, bang-bang CDR

Two choices here, each with a reason beyond convention.

**Baud-rate rather than oversampled.** One sample per UI means no second
high-speed sampler. More interesting is that a Mueller-Müller detector locks
where `h₁ = h₋₁` — a condition written entirely in the pulse-response cursors
the DFE already computes. The CDR reuses the DFE's `h_m` rather than needing a
measurement of its own. That economy is the argument.

**Bang-bang rather than linear.** A binary detector needs no linear analog
comparison and feeds a purely digital loop filter, which is robust and portable.
The cost is real and must be stated: a hard quantizer has no small-signal gain
at all. What makes it analyzable is that jitter dithers the quantizer, so the
*average* output is smooth — `K_pd = 2/(σ_τ√2π)`. Two consequences follow
directly, and both are design constraints rather than footnotes:

- `K_pd ∝ 1/σ_τ`, so **loop bandwidth depends on the operating point**. Every
  `ω_n` and `ζ` is valid only for the jitter it was computed at. This is why the
  loop is designed overdamped (`ζ = 2…5`) — it must stay well-behaved as
  conditions move it around.
- The loop cannot output zero, so it dithers by `≈ K_p` pk-pk. Since bandwidth
  is also set by `K_p`, jitter tolerance and dither are one decision, not two.

Type-II (two integrators) is what lets the loop track a frequency offset with
`H(0) = 1` — a CDR should pass low-frequency jitter, because jitter common to
data and clock causes no sampling error.

## What makes this one system rather than three blocks

The reason this model is built as a single pipeline sharing one pulse response
is that the three blocks are mutually coupled, in a closed ring:

| Coupling | Mechanism |
|---|---|
| CTLE → DFE | Taps are cursors of the combined channel + CTLE response |
| DFE → CDR | `h_m = p(t_s + mT)` — tap values depend on where the CDR locks |
| CDR → DFE | Tapping the PD before or after DFE subtraction moves the lock point from `h₁ = h₋₁` to `h₋₁ = 0` |
| CTLE → CDR | `K_mm` is the slope of the *equalized* pulse response — better EQ sharpens edges, raises loop gain, widens the loop |

None of these is a second-order correction. Choosing the CTLE changes the taps,
which changes the lock point, which changes the taps again. Designing the blocks
independently and cascading them would produce numbers that do not survive
contact with a full simulation.

## What this architecture deliberately omits

No TX FFE (this is an RX study), no crosstalk, no noise in the first pass, and
an ideal slicer. The crosstalk omission is the one to keep in view: 802.3ck
links are crosstalk-limited in practice, so every margin number here is
optimistic by an amount this model does not estimate.

## The open question

An ideal 2-tap DFE alone leaves 127 mV of worst-case opening — 30 % of the main
cursor, before noise, jitter, or crosstalk. The measured postcursor tail is long
(cursor +100 is still 0.21 mV) and two taps cancel only 46 % of it. The whole
architecture rests on the CTLE shortening that tail enough that two taps finish
the job. **Whether it does is the first thing this model should answer** — and
if it does not, the honest conclusions are more taps or a TX FFE, not a larger
CTLE boost.
