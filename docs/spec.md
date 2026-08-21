# Link Specification — 16 Gbps NRZ Wireline RX

Status: **draft**. The channel is characterized (§3) and the pre-equalization
baseline measured (§5); CTLE, DFE, and CDR parameters are still open. See
[Open items](#6-open-items).

## 1. Link-level parameters

| Parameter          | Value        | Notes                                      |
|--------------------|--------------|--------------------------------------------|
| Data rate          | 16 Gbps      | NRZ (PAM-2), 1 bit/symbol                  |
| Symbol rate        | 16 GBd       |                                            |
| Unit interval (UI) | 62.5 ps      | 1 / 16 GBd                                 |
| Nyquist frequency  | 8 GHz        | Symbol rate / 2                            |
| Lanes              | 1            | Single lane; no crosstalk aggressors modeled |
| Modulation         | NRZ          |                                            |
| Target BER         | 1e-12        | Reference point for margin extraction      |
| PRBS pattern       | TBD          | e.g. PRBS7/PRBS13Q — to be fixed           |
| Differential swing | TBD          | TX launch amplitude                        |
| Coding             | None assumed | No FEC in the model                        |

## 2. Receiver architecture

Signal chain: **channel → CTLE → DFE → slicer**, with the sampling phase driven
by a bang-bang CDR.

| Block  | Type                                   | Parameters                          | Status |
|--------|----------------------------------------|-------------------------------------|--------|
| CTLE   | Continuous-time linear equalizer, 1–2 peaking stages | DC gain, peak gain, zero/pole frequencies, stage count | Topology fixed, values TBD |
| DFE    | Decision feedback equalizer, 1–2 taps   | Tap count, tap weights, adaptation on/off | Topology fixed, values TBD |
| CDR    | Baud-rate, bang-bang phase detector (Alexander-type) | Loop bandwidth, damping factor, proportional/integral gains | Topology fixed, values TBD |
| Slicer | Ideal decision device                   | Threshold, sampling phase from CDR  | Ideal assumed |

Not modeled at this stage: TX FFE, AGC/VGA, receiver noise, jitter sources,
package parasitics, crosstalk. Each is TBD as to whether it will be added.

## 3. Channel

Obtained and characterized. Run `python scripts\plot_channel.py` to regenerate
every number in this section.

| Item                | Value                                                        |
|---------------------|--------------------------------------------------------------|
| Source              | IEEE 802.3ck public backplane (Palkert, `palkert_3ck_02_0120`) |
| File                | `channel/palkert_3ck_02_0120/THRU_VL5_OD-BP-Channel_16inch_16inch.s4p` |
| Topology            | Backplane through, 16 inch + 16 inch                          |
| Format              | Touchstone `.s4p`, 4-port single-ended, RI                    |
| Frequency range     | 0 – 50 GHz, 2501 points, uniform 20 MHz step, **DC included**  |
| Reference impedance | 50 Ω single-ended (100 Ω differential after conversion)        |
| Port ordering       | Through-paired: (1,2) are the ends of one line, (3,4) the other. Differential pairs are {1,3} near / {2,4} far. Detected, not assumed — see `src/channel.py` |
| Sdd21 @ 1 GHz       | −3.83 dB                                                      |
| Sdd21 @ 4 GHz       | −8.58 dB                                                      |
| **Sdd21 @ 8 GHz (Nyquist)** | **−13.19 dB**                                         |
| Sdd21 @ 16 GHz      | −20.22 dB                                                     |
| Sdd11 (in band)     | ≈ −20 dB, with reflection ripple                              |
| Scd21 (mode conv.)  | ≈ −30 dB                                                      |
| Propagation delay   | 4.91 ns to pulse-response peak (≈ 0.55c, ε_r,eff ≈ 3.3)        |

The frequency grid is uniform and includes DC, so the impulse response follows
from a direct inverse real DFT with no extrapolation at the low end. Truncation
above 50 GHz is benign: Sdd21 is already below −45 dB there.

## 4. Simulation methods

Both methods are in scope and must agree with each other.

| Method                | Basis                     | Produces                                | Primary use              |
|-----------------------|---------------------------|-----------------------------------------|--------------------------|
| Time-domain           | Bit-by-bit transient over a PRBS sequence | Waveforms, eye diagrams, CDR transient behavior | Verifying DFE and CDR dynamics |
| Statistical           | Pulse response + ISI distribution convolution | BER vs. eye opening, bathtub curves     | Extracting margin at 1e-12 |

| Setting                   | Value | Basis |
|---------------------------|-------|-------|
| Samples per UI (M)        | 32    | `fs = 512 GHz`, `dt = 1.95 ps` — ample for CDR phase resolution |
| Impulse record length (N) | 32768 | 64 ns window, ≈ 13× the 4.91 ns channel delay |
| Statistical ISI depth     | **≥ 120 postcursors** | Measured tail convergence, below |
| Precursor depth           | 4     | Precursors beyond −1 are < 0.02 mV |
| Simulated bit count (time-domain) | TBD | Set by the BER the time-domain run must resolve |

The postcursor tail on this channel is long and decays slowly, so the ISI depth
is not a free parameter — truncating it silently inflates the eye:

| Depth (UI) | Postcursor ISI captured | Worst-case eye, ideal 2-tap DFE |
|------------|-------------------------|----------------------------------|
| 20         | 86.1 %                  | 197 mV (optimistic by 70 mV)     |
| 50         | 94.2 %                  | 157 mV                           |
| 120        | 98.7 %                  | 134 mV                           |
| 200        | 100 %                   | 127 mV                           |

Cursor +100 is still 0.21 mV. Use 120 as the working depth and 200 when
quoting a final number.

Time-domain simulation cannot reach BER 1e-12 directly — margin at that BER
comes from the statistical method, and the two are reconciled at a BER both can
resolve.

## 5. Success criteria

| # | Criterion                | Definition                                                                 | Target |
|---|--------------------------|----------------------------------------------------------------------------|--------|
| 1 | Eye diagram, pre vs. post EQ | Eye diagram at the slicer input with equalization disabled and enabled, plotted from the same stimulus | Must open from the measured closed baseline below |
| 2 | Eye margin at BER 1e-12  | Eye height (V) and eye width (UI) at the 1e-12 contour, from the statistical method | Height: ______ &nbsp;&nbsp; Width: ______ UI |
| 3 | Theory cross-check       | Each block's simulated behavior compared against its closed-form expression — CTLE frequency response vs. transfer function, DFE residual ISI vs. hand-computed cursors, CDR jitter transfer vs. linearized loop model | Agreement within a tolerance TBD per block |

### Measured baseline (no equalization)

Peak-distortion analysis of the pulse response, 1 V differential input pulse,
200-UI ISI depth:

| Quantity                         | Value      |
|----------------------------------|------------|
| Main cursor                      | 423.8 mV   |
| Precursor (−1)                   | 23.5 mV    |
| Postcursor +1 / +2               | 159.5 / 75.0 mV (0.376 / 0.177 of main) |
| Total ISI, all cursors           | 531.1 mV   |
| **Worst-case eye, no EQ**        | **−107 mV → closed** |
| Worst-case eye, ideal 2-tap DFE  | +127 mV    |

Two consequences for the design:

- The eye is **closed by 107 mV** before equalization, which is the starting
  point criterion 1 must improve on.
- An ideal 2-tap DFE cancels only 46 % of the postcursor ISI, leaving a 127 mV
  worst-case opening — 30 % of the main cursor, before any noise, jitter, or
  crosstalk is added. **The 1–2 tap DFE in §2 cannot carry this channel alone**;
  the CTLE must shorten the tail so the residual falls within reach of the taps.
  Whether 2 taps suffice once the CTLE is in place is the first question the
  model should answer.

### Sampling phase — the CDR does not lock where the eye is best

All figures above are at the pulse-response peak. That is not where a
Mueller-Müller CDR settles: it locks where `h₁ = h₋₁`
([`dfe_notes.md`](dfe_notes.md) §6). On this channel, unequalized, those are very
different phases.

| Phase | Offset | Main | `h₋₁` | Eye, ideal 2-tap DFE |
|---|---|---|---|---|
| Pulse peak | 0 UI | 423.8 mV | 23.5 mV | **+127 mV** |
| Best available | −0.063 UI | — | — | +133 mV |
| **MM lock** | **+0.230 UI** | 372.8 mV | 123.3 mV | **−12 mV — closed** |

The channel is strongly asymmetric — a long postcursor tail against an almost
negligible precursor — so forcing `h₁ = h₋₁` drags the sampler 0.23 UI late.
That costs 51 mV of main cursor and inflates `h₋₁` more than fivefold, and the
precursor is the one term a DFE structurally cannot cancel. Net cost against the
best phase: **145 mV**, enough to close an eye that is otherwise open.

Even a 4-tap DFE holds only ~40 mV at the lock phase. Three implications:

- **The CDR lock phase belongs inside the CTLE design loop**, not after it. The
  CTLE shortens the tail, which is also what pulls the lock point back toward
  the peak — one mechanism serving two purposes.
- **Where the phase detector taps is now a first-order decision, not a detail.**
  A PD observing the post-DFE signal drives toward `h₋₁ = 0` instead of
  `h₁ = h₋₁`, which on this channel is a completely different phase.
- Every margin number must state its sampling phase. Quoting a peak-phase eye
  for a link with a baud-rate CDR overstates it.

This is the CTLE↔DFE↔CDR coupling in [`notes.md`](notes.md) made numerical, and
it is the reason the model is built around a phase-resolved pulse response
rather than a single cursor vector.

## 6. Open items

| Item                                    | Blocks                          | Status |
|-----------------------------------------|---------------------------------|--------|
| Obtain 802.3ck `.s4p` channel file      | All quantitative targets        | **done** — §3 |
| Fix samples-per-UI and ISI depth         | Both simulation methods         | **done** — §4 |
| Choose CTLE pole/zero values             | Criteria 1 and 2                | next — gates everything below |
| Confirm DFE tap count (2 may not suffice) | Criteria 1 and 2               | open, see §5 baseline |
| Fix quantitative eye height / width targets | Criterion 2                  | open — set after CTLE is chosen |
| Choose PRBS pattern and TX swing         | Time-domain simulation          | open |
| **Decide where the PD taps** (pre- or post-DFE) | Sampling phase, hence every margin | **open — now first-order, see §5** |
| Choose CDR loop bandwidth and damping    | CDR verification                | open |
| Set per-block cross-check tolerances     | Criterion 3                     | open |
| Decide whether noise and jitter enter the model | Realism of criterion 2   | open |
| Fix simulated bit count (time-domain)    | Time-domain simulation          | open |

## 7. Unverified assumptions

Assumptions this specification currently rests on, none of them yet confirmed:

1. A single 802.3ck backplane channel is representative enough for the study; no
   channel sweep is planned.
2. Ignoring crosstalk is acceptable — 802.3ck links are crosstalk-limited in
   practice, so margin computed here will be optimistic.
3. The TX is ideal (no TX FFE, no transmitter jitter), so all equalization burden
   falls on the RX.
4. The slicer is ideal — infinite bandwidth, no metastability, no offset.
5. DFE error propagation is either negligible or will be modeled explicitly; which
   one is not yet decided.
6. A linearized loop model is a valid reference for a bang-bang CDR, whose phase
   detector is inherently nonlinear. Valid only in the low-jitter,
   many-transitions regime.
7. The statistical method's assumption of data-independent ISI holds — it does not
   hold exactly once a DFE with decision feedback is in the loop.
8. Noiseless operation; BER 1e-12 margin therefore reflects ISI only, not
   ISI plus noise.
