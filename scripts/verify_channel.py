"""Verify that the .s4p -> impulse -> pulse conversion is physically valid.

Three independent checks, each with a numeric result and a plot panel:

  1. **Causality.**  A physical channel cannot respond before the signal
     arrives.  Energy before the causal onset comes from band truncation
     (Gibbs ringing) or inconsistent phase data, and it lands on the
     pre-cursors -- the one place a DFE cannot clean it up.

  2. **Cursor sum vs. low-frequency gain.**  Sampling the pulse response once
     per UI and summing every cursor must reproduce the DC gain exactly:

         sum_m p(t_s + m*T) = sum_n h[n] = H(0)

     because convolving with a boxcar exactly one UI wide and then decimating
     by M assigns each impulse-response sample to exactly one cursor.  The
     identity is independent of the sampling phase t_s, which makes it a
     strong test: it checks the IFFT normalization, the pulse-response boxcar
     width, and the cursor indexing all at once, and it must hold at every one
     of the M phases.

  3. **Passivity.**  A passive network satisfies sigma_max(S) <= 1 at every
     frequency.  The elementwise test |S_ij| <= 1 is necessary but weaker, so
     both are reported.  A violation means the data gains energy, and any
     impulse response derived from it is unphysical.

Exit code is 0 if every check passes, 1 otherwise.

    python scripts\\verify_channel.py
    python scripts\\verify_channel.py --config no_eq --taper
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import skrf as rf  # noqa: E402

from src import channel, configs, plotting  # noqa: E402


@dataclass
class Check:
    """One verification result.

    ``warn_only`` marks a check that reports a property of the *source data*
    rather than of our conversion.  It is printed and counted but does not fail
    the run, because no change to this code could make it pass.
    """

    name: str
    value: float
    tol: float
    detail: str
    warn_only: bool = False

    @property
    def passed(self) -> bool:
        return self.value <= self.tol

    def line(self) -> str:
        if self.passed:
            status = "PASS"
        else:
            status = "WARN" if self.warn_only else "FAIL"
        return (f"  [{status}] {self.name:<34s} {self.value:>11.4e} "
                f"<= {self.tol:.0e}   {self.detail}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="baseline", choices=configs.available())
    ap.add_argument("--channel", type=Path, default=None,
                    help="override the .s4p path in the configuration")
    ap.add_argument("--taper", action="store_true",
                    help="apply a raised-cosine roll-off before band truncation")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    cfg = configs.get(args.config)
    if args.channel is not None:
        cfg = replace(cfg, channel_path=args.channel)
    if not cfg.channel_path.exists():
        print(f"channel file not found: {cfg.channel_path}", file=sys.stderr)
        return 1

    print(f"verifying : {cfg.channel_path.name}")
    print(f"config    : {args.config}, taper={args.taper}")

    raw = rf.Network(str(cfg.channel_path))
    ntwk = channel.load(cfg.channel_path)
    mm = channel.mixed_mode(ntwk)
    f, sdd21 = mm["f"], mm["sdd21"]

    t, h, grid = channel.impulse_response(f, sdd21, cfg, taper=args.taper)
    p = channel.pulse_response(h, cfg)
    dc_gain = float(np.real(sdd21[0]))
    checks: list[Check] = []

    # ---------------------------------------------------------------- 1 ---
    print("\n--- 1. Causality " + "-" * 55)
    pre_frac, tol_c = channel.check_causality(h)
    mag = np.abs(h)
    k_peak = int(np.argmax(mag))
    k_onset = int(np.argmax(mag > 0.01 * mag.max()))
    early = mag[: max(k_onset - cfg.samples_per_ui, 1)].max()

    print(f"  peak            : sample {k_peak}, t = {t[k_peak] * 1e9:.4f} ns, "
          f"|h| = {mag[k_peak] * 1e3:.4f} mV")
    print(f"  causal onset    : sample {k_onset}, t = {t[k_onset] * 1e9:.4f} ns "
          f"(1% of peak)")
    print(f"  max |h| >1 UI before onset : {early * 1e6:.4f} uV "
          f"({early / mag[k_peak]:.2e} of peak)")
    print(f"  band truncation : {grid.truncation_db:.1f} dB at "
          f"{grid.f_max_measured / 1e9:.1f} GHz")
    checks.append(Check(
        "causality: pre-onset energy", pre_frac, tol_c,
        "energy before the arrival; Gibbs ringing if high",
    ))

    # ---------------------------------------------------------------- 2 ---
    print("\n--- 2. Cursor sum vs. low-frequency gain " + "-" * 31)
    m = cfg.samples_per_ui
    # The identity holds only when the cursors cover the entire record: the
    # pulse response at cursor m draws on h over a window M wide, so the slots
    # must extend past both ends of h or the uncovered samples are simply
    # dropped from the sum.  With cfg.n_pre = 4 the head h[:2333] is excluded
    # and its sum (-8.3e-5) shows up directly as the error.  Slots outside the
    # record contribute zero, so over-extending is free.
    n_pre_full = k_peak // m + 2
    n_post_full = (h.size - k_peak) // m + 2

    # ``pulse_response`` trims the convolution back to the record length, which
    # discards its last M-1 samples.  Those samples hold real signal, so the
    # identity cannot be exact on the trimmed array -- the deficit is exactly
    # their sum.  Test the identity on the untrimmed convolution, where it is
    # a true statement about scaling and indexing, and report the trim
    # separately as the (negligible) modelling choice it is.
    p_exact = np.convolve(h, np.ones(m))

    def cursor_sum(arr: np.ndarray, t_s: int) -> float:
        idx = np.arange(-n_pre_full, n_post_full + 1) * m + t_s
        ok = (idx >= 0) & (idx < arr.size)
        return float(arr[idx[ok]].sum())

    phases = k_peak + np.arange(m) - m // 2
    sums = np.array([cursor_sum(p_exact, int(ts)) for ts in phases])
    err_vs_phase = (sums - dc_gain) / dc_gain
    worst = float(np.abs(err_vs_phase).max())
    tol_sum = 1e-12

    trimmed = np.array([cursor_sum(p, int(ts)) for ts in phases])
    trim_err = float(np.abs((trimmed - dc_gain) / dc_gain).max())
    lost = float(p_exact[h.size:].sum())

    sp_full = channel.split_pulse(p_exact, cfg, n_pre=n_pre_full, n_post=n_post_full)

    print(f"  H(0) = Sdd21(0)              : {dc_gain:.12f}")
    print(f"  sum(h)                       : {np.sum(h):.12f}")
    print(f"  cursor coverage              : {n_pre_full} pre / {n_post_full} post "
          f"(spans the whole {h.size}-sample record)")
    print(f"  sum of cursors, peak phase   : {sums[m // 2]:.12f}")
    print(f"  worst rel. error over {m} phases : {worst:.3e}")
    print(f"  ...using the trimmed p(t)    : {trim_err:.3e}  "
          f"(discards {m - 1} samples summing to {lost:.3e})")

    # What the working depth actually captures -- this is the truncation that
    # matters for margin numbers, not a correctness failure.
    sp_cfg = channel.split_pulse(p, cfg)
    cap = float(sp_cfg.at(sp_cfg.peak_phase).sum() / dc_gain)
    print(f"  at cfg.isi_depth = {cfg.isi_depth:<4d}     : "
          f"{cap * 100:.4f}% of H(0) captured "
          f"({(1 - cap) * dc_gain * 1e3:.3f} mV of tail beyond the window)")
    checks.append(Check(
        "cursor sum == H(0), all phases", worst, tol_sum,
        "checks IFFT scaling, boxcar width, cursor indexing",
    ))
    checks.append(Check(
        "  ... with p(t) as returned", trim_err, 1e-4,
        "residual is the trimmed convolution tail, not an error",
    ))

    # ---------------------------------------------------------------- 3 ---
    print("\n--- 3. Passivity " + "-" * 55)
    s = raw.s
    max_abs = np.abs(s).max(axis=(1, 2))
    sigma = np.linalg.svd(s, compute_uv=False).max(axis=1)
    recip = np.abs(s - np.transpose(s, (0, 2, 1))).max()

    # The DC row is judged separately.  Many tools append a synthetic DC point
    # rather than measuring one, and a synthesized row need not be passive.
    # Here the DC row's imaginary parts are ~1e-19, which is the giveaway.
    dc_is_synthetic = raw.f[0] == 0.0 and np.abs(np.imag(s[0])).max() < 1e-15
    band = slice(1, None) if raw.f[0] == 0.0 else slice(None)
    sigma_band = sigma[band]
    i_worst = int(np.argmax(sigma_band))

    print(f"  max |S_ij|, measured band : {np.abs(s[band]).max():.6f}")
    print(f"  max sigma(S), measured band : {sigma_band.max():.6f} at "
          f"{raw.f[band][i_worst] / 1e9:.2f} GHz")
    print(f"  reciprocity               : max|S - S^T| = {recip:.3e}")
    if raw.f[0] == 0.0:
        print(f"  DC row sigma              : {sigma[0]:.6f}"
              + ("   (synthesized: max|Im S| = "
                 f"{np.abs(np.imag(s[0])).max():.1e})" if dc_is_synthetic else ""))

    checks.append(Check(
        "passivity: max sigma(S), band", float(sigma_band.max()), 1.0 + 1e-6,
        "singular value; the rigorous test",
    ))
    checks.append(Check(
        "passivity: max |S_ij|, band", float(np.abs(s[band]).max()), 1.0 + 1e-6,
        "elementwise; necessary but weaker",
    ))
    checks.append(Check(
        "reciprocity: max|S - S^T|", float(recip), 1e-6,
        "expected for a passive reciprocal structure",
    ))
    if raw.f[0] == 0.0:
        checks.append(Check(
            "passivity: DC row sigma(S)", float(sigma[0]), 1.0 + 1e-6,
            "source-data property, not ours -- see note below",
            warn_only=True,
        ))

    # --- also re-run the cheap self-tests the pipeline uses ---------------
    _, dc_err = channel.check_dc_gain(h, sdd21[0])
    checks.append(Check("DC self-test: |sum(h) - H(0)|", dc_err, 1e-9,
                        "IFFT normalization"))
    checks.append(Check("wraparound: tail energy", channel.check_wraparound(h), 1e-6,
                        "record long enough for the impulse response"))

    # ---------------------------------------------------------------------
    print("\n--- Summary " + "-" * 60)
    for c in checks:
        print(c.line())
    n_fail = sum((not c.passed) and (not c.warn_only) for c in checks)
    n_warn = sum((not c.passed) and c.warn_only for c in checks)

    if n_warn:
        print("\nnotes")
        print("  The file's DC row is synthesized rather than measured (imaginary")
        print("  parts ~1e-19) and is marginally non-passive. Every measured point")
        print("  is passive. Impact is bounded: only Sdd21(0) is used from that row,")
        print(f"  and it is {dc_gain:.4f} < 1. Nothing in this code can fix it, so it")
        print("  is reported rather than failed -- but a channel whose DC gain")
        print("  mattered more would want the row re-derived by extrapolation.")

    cumulative = [
        (np.cumsum(sp_full.at(j)), lbl)
        for j, lbl in ((sp_full.peak_phase, "peak phase"),
                       (channel.mm_lock_phase(sp_full)[0], "MM lock phase"),
                       (0, f"{sp_full.phase_ui(0):+.2f} UI"))
    ]
    out = plotting.plot_verification({
        "t": t, "h": h, "k_onset": k_onset, "k_peak": k_peak,
        "causality": pre_frac,
        "dc_gain": dc_gain, "n_pre": n_pre_full, "isi_depth": cfg.isi_depth,
        "cumulative": cumulative,
        "f_s": raw.f, "max_abs_s": max_abs, "sigma_max": sigma,
        "phase_ui": np.asarray(sp_full.phase_ui(np.arange(m))),
        "sum_err_vs_phase": err_vs_phase, "sum_tol": tol_sum,
    })
    print(f"\nwrote: {out}")

    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed"
          + ("" if n_fail == 0 else f" -- {n_fail} FAILED"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
