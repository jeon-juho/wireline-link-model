"""Eye analysis: time-domain eyes, statistical BER contour, bathtubs, jitter split.

Covers spec.md criteria 1 and 2, and exercises criterion 3 by running both
simulation methods on the same pulse response and comparing them where they
overlap.

Why the two disagree, and which to believe:

  - **Reachable BER.**  A time-domain run of N symbols resolves no lower than
    about 100/N.  BER 1e-12 needs ~1e14 symbols.  Only the statistical method
    can answer criterion 2.
  - **Pattern coverage.**  Time-domain sees only the ISI combinations the PRBS
    emitted; with 200 cursors that is a vanishing fraction of 2^200.  The
    statistical method weights every combination, so it is *more pessimistic*
    in the tail.
  - **Independence.**  The statistical model treats cursors as independent.
    True for a linear channel with random data, false once a DFE feeds
    decisions back or a CDR makes the sampling phase data-dependent -- so it is
    *optimistic* exactly where those matter.

Trust the statistical curve for low-BER margin, the time-domain run for
anything involving decisions, feedback or transients, and require the two to
agree in the BER range both can resolve.  Disagreement there is diagnostic.

    python scripts\\analyze_eye.py
    python scripts\\analyze_eye.py --taps 2 --sigma-n 3e-3 --rj 0.008
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from src import analog, channel, configs, plotting, sim_stat, tx  # noqa: E402
from src.ctle import CTLE  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="baseline", choices=configs.available())
    ap.add_argument("--symbols", type=int, default=200_000,
                    help="time-domain PRBS length")
    ap.add_argument("--taps", type=int, default=2, help="ideal ZF DFE taps")
    ap.add_argument("--sigma-n", type=float, default=10e-3,
                    help="rms slicer-referred voltage noise (V); stands in for "
                         "noise plus the crosstalk allowance spec.md section 7 "
                         "flags as un-modelled")
    ap.add_argument("--rj", type=float, default=0.006,
                    help="injected random jitter, rms UI")
    ap.add_argument("--dj", type=float, default=0.020,
                    help="injected deterministic jitter, pk-pk UI")
    ap.add_argument("--swing", type=float, default=0.5,
                    help="TX symbol level (V); pk-pk differential is twice this")
    return ap.parse_args(argv)


def pulse_for(cfg, with_eq: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Impulse and pulse response, with or without the CTLE."""
    mm = channel.mixed_mode(channel.load(cfg.channel_path))
    h_f = mm["sdd21"]
    if with_eq:
        c = CTLE.from_config(cfg.ctle)
        # Normalize to 0 dB at the peak.  A CTLE creates slope by discarding DC
        # gain, not by adding gain (docs/ctle_notes.md section 4), so leaving
        # dc_gain_db at 0 makes the stage amplify by its full boost -- here a
        # 636 mV main cursor instead of 140 mV, and a Q of 258 instead of 11.
        c = CTLE(c.f_z, c.f_p1, c.f_p2, dc_gain_db=-c.boost_db,
                 n_stages=c.n_stages)
        h_f = analog.compose(h_f, c.response(mm["f"]))
    _, h, _ = channel.impulse_response(mm["f"], h_f, cfg)
    p = channel.pulse_response(h, cfg)
    return h, p, int(np.argmax(p))


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    cfg = configs.get(args.config)

    print(f"config {args.config}: {args.symbols:,} symbols, {args.taps}-tap ideal DFE")
    print(f"  noise sigma_n = {args.sigma_n * 1e3:.2f} mV, "
          f"RJ = {args.rj * 1e3:.2f} mUI rms, DJ = {args.dj * 1e3:.1f} mUI pp")
    print(f"  TX swing      = {args.swing * 2 * 1e3:.0f} mVppd")

    # ---------------------------------------------------------------- 1 ---
    print("\n--- 1. Time-domain eye, EQ off vs. on " + "-" * 30)
    w = tx.generate(cfg, args.symbols, prbs_order=15, rise_time=12e-12,
                    edge_method="raised_cosine", swing=args.swing)
    d = np.sign(w.symbols)

    def apply_ideal_dfe(rx, p, k, n_taps):
        """Subtract the known postcursors using the *transmitted* symbols.

        Assuming correct decisions excludes error propagation here just as the
        statistical model excludes it, which keeps the two comparable rather
        than making the time-domain run the fairer of the pair.
        """
        if not n_taps:
            return rx
        cur = sim_stat.cursors_at_phase(p, cfg, k, 0.0) * args.swing
        taps = cur[cfg.n_pre + 1: cfg.n_pre + 1 + n_taps]
        per_symbol = np.zeros(d.size)
        for lag, c in enumerate(taps, start=1):
            per_symbol[lag:] += c * d[:-lag]
        start = k - cfg.samples_per_ui // 2   # same window fold_eye slices out
        up = np.repeat(per_symbol, cfg.samples_per_ui)
        take = min(up.size, rx.size - start)
        out = rx.copy()
        out[start: start + take] -= up[:take]
        return out

    eyes, rx_eq = {}, None
    for label, with_eq in (("off", False), ("on", True)):
        h, p, k = pulse_for(cfg, with_eq)
        rx = sps.oaconvolve(w.waveform, h)[: w.waveform.size]
        if with_eq:
            rx = apply_ideal_dfe(rx, p, k, args.taps)
            rx_eq = rx
        eyes[label] = sim_stat.fold_eye(rx, cfg, k, d)
        eh, ph = eyes[label].eye_height()
        print(f"  EQ {label:<3s}: eye height {eh * 1e3:8.2f} mV at {ph:+.3f} UI, "
              f"width {eyes[label].eye_width():.4f} UI")

    # ---------------------------------------------------------------- 2 ---
    print("\n--- 2. Statistical eye " + "-" * 45)
    h, p, k = pulse_for(cfg, with_eq=True)
    p = p * args.swing                      # pulse response is for a unit pulse
    se = sim_stat.build_stat_eye(
        p, cfg, k, sigma_n=args.sigma_n, sigma_rj_ui=args.rj,
        dj_pp_ui=args.dj, n_taps=args.taps, n_phase=385, n_threshold=321,
        phase_span=1.4,
    )
    j, vb = se.vertical_bathtub()
    hb = se.horizontal_bathtub(0.0)
    eh12 = se.eye_height(1e-12)
    ew12 = se.eye_width(1e-12)

    print(f"  best sampling phase : {se.phase_ui[j]:+.4f} UI")
    print(f"  main cursor there   : {se.main[j] * 1e3:.2f} mV")
    print(f"  minimum BER         : {se.ber.min():.3e}")
    print(f"\n  {'target BER':>12s} {'eye height':>14s} {'eye width':>13s}")
    for tgt in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15):
        print(f"  {tgt:12.0e} {se.eye_height(tgt) * 1e3:11.2f} mV "
              f"{se.eye_width(tgt):10.4f} UI")
    print(f"\n  >>> at BER 1e-12: eye height {eh12 * 1e3:.2f} mV, "
          f"eye width {ew12:.4f} UI ({ew12 * cfg.ui * 1e12:.2f} ps)")

    # ---------------------------------------------------------------- 3 ---
    print("\n--- 3. Cross-check in the overlapping BER range " + "-" * 20)
    td_floor = 100.0 / args.symbols
    print(f"  time-domain floor (100 errors / {args.symbols:,} symbols)"
          f" : {td_floor:.2e}")
    # The time-domain run has no injected jitter, so it must be compared with a
    # statistical eye built without jitter too -- otherwise the statistical
    # bathtub is narrower purely because 20 mUI of DJ was imposed on one side
    # of the comparison and not the other.
    se_ref = sim_stat.build_stat_eye(
        p, cfg, k, sigma_n=args.sigma_n, sigma_rj_ui=0.0, dj_pp_ui=0.0,
        n_taps=args.taps, n_phase=385, n_threshold=321,
        phase_span=1.4,
    )
    hb_ref = se_ref.horizontal_bathtub(0.0)

    m = cfg.samples_per_ui
    rng = np.random.default_rng(0)
    td_phase, td_ber = [], []
    for ph in np.linspace(-0.45, 0.45, 25):
        idx = k + int(round(ph * m)) + np.arange(d.size) * m
        idx = idx[idx < rx_eq.size]
        # Add the same Gaussian noise the statistical model assumes, or the
        # time-domain link is noiseless and makes no errors at all.
        v = rx_eq[idx] + rng.normal(0.0, args.sigma_n, idx.size)
        errs = int(np.count_nonzero(np.sign(v) != d[: idx.size]))
        td_phase.append(ph)
        td_ber.append(errs / idx.size)
    td_phase, td_ber = np.array(td_phase), np.array(td_ber)

    both = (td_ber > td_floor) & (td_ber < 0.3)
    if both.any():
        # Interpolate the statistical curve in log space; it spans decades and
        # linear interpolation between them is meaningless.
        stat_at = np.exp(np.interp(td_phase[both], se_ref.phase_ui,
                                   np.log(np.maximum(hb_ref, 1e-300))))
        print(f"  {'phase':>8s} {'time-domain':>13s} {'statistical':>13s} "
              f"{'stat/td':>9s}")
        for ph, tb, sb in zip(td_phase[both], td_ber[both], stat_at):
            print(f"  {ph:+8.3f} {tb:13.3e} {sb:13.3e} {sb / tb:9.2f}")
        # Geometric mean: these ratios span decades, so an arithmetic mean is
        # dominated by whichever point happens to be largest.
        gm = float(np.exp(np.mean(np.log(stat_at / td_ber[both]))))
        print(f"\n  statistical / time-domain, geometric mean : {gm:.2f}x")
        print(f"  Agreement near 1x in this range is the result that licenses "
              f"extrapolating the")
        print(f"  statistical curve down to 1e-12.  The two are expected to "
              f"diverge only deeper in")
        print(f"  the tail, where the rare ISI patterns that {args.symbols:,} "
              f"symbols never produce start")
        print(f"  to dominate -- and there only the statistical method can "
              f"speak at all.")
    else:
        print("  no phase where both methods resolve a BER -- widen the sweep")

    # ---------------------------------------------------------------- 4 ---
    print("\n--- 4. Jitter decomposition from the bathtub " + "-" * 23)
    q_lo, q_hi = 3.0, 7.0
    fit = sim_stat.fit_dual_dirac(se.phase_ui, hb, q_lo=q_lo, q_hi=q_hi)
    print(f"  fitted over Q = {q_lo:.0f}..{q_hi:.0f} "
          f"(BER 1e-3 to 1e-12), {fit.n_points} points")
    print(f"  left  edge : sigma = {fit.sigma_left * 1e3:7.3f} mUI, "
          f"mu = {fit.mu_left:+.4f} UI")
    print(f"  right edge : sigma = {fit.sigma_right * 1e3:7.3f} mUI, "
          f"mu = {fit.mu_right:+.4f} UI")
    print(f"\n  {fit.summary()}")
    print(f"\n  injected RJ {args.rj * 1e3:.2f} mUI -> recovered "
          f"{fit.rj_rms_ui * 1e3:.2f} mUI "
          f"({fit.rj_rms_ui / args.rj * 100:.1f}%)")
    print(f"  injected DJ {args.dj * 1e3:.1f} mUI -> recovered "
          f"{fit.dj_pp_ui * 1e3:.1f} mUI")
    print(f"  the excess DJ is data-dependent jitter: residual ISI closes the eye")
    print(f"  horizontally and is deterministic, so the fit attributes it to DJ.")
    print(f"\n  eye width from the fit  : {fit.eye_width():.4f} UI")
    print(f"  eye width measured      : {ew12:.4f} UI")

    # Validate the fitter itself.  With vertical noise present, the bathtub edge
    # slope is set by noise-divided-by-slew-rate as well as by jitter, so the
    # fit legitimately returns an *effective* RJ well above what was injected --
    # vertical noise converts into apparent horizontal jitter.  Removing the
    # noise isolates the jitter, and the fit should then return what went in.
    se_q = sim_stat.build_stat_eye(
        p, cfg, k, sigma_n=1e-5, sigma_rj_ui=args.rj, dj_pp_ui=args.dj,
        n_taps=args.taps, n_phase=385, n_threshold=33, phase_span=1.4,
    )
    fq = sim_stat.fit_dual_dirac(se_q.phase_ui, se_q.horizontal_bathtub(0.0),
                                 q_lo=q_lo, q_hi=q_hi)
    print(f"\n  fitter check -- same jitter, noise removed (sigma_n = 0.01 mV):")
    print(f"    RJ {fq.rj_rms_ui * 1e3:6.2f} mUI vs {args.rj * 1e3:.2f} injected "
          f"({fq.rj_rms_ui / args.rj * 100:.0f}%)")
    print(f"    DJ {fq.dj_pp_ui * 1e3:6.1f} mUI vs {args.dj * 1e3:.1f} injected "
          f"-- excess is data-dependent jitter from residual ISI")
    print(f"  So of the {fit.rj_rms_ui * 1e3:.1f} mUI 'RJ' fitted above, only "
          f"~{fq.rj_rms_ui * 1e3:.1f} mUI is real random jitter;")
    print(f"  the rest is {args.sigma_n * 1e3:.0f} mV of vertical noise "
          f"reappearing as horizontal jitter.")

    with np.errstate(divide="ignore", invalid="ignore"):
        q_curve = sim_stat.q_of_ber(np.clip(hb, 1e-300, 0.5 - 1e-12))

    out = plotting.plot_eye_analysis({
        "eye_off": eyes["off"], "eye_on": eyes["on"], "stat_eye": se,
        "horizontal": hb, "eye_height_1e12": eh12, "eye_width_1e12": ew12,
        "fit": fit, "q_curve": q_curve, "q_lo": q_lo, "q_hi": q_hi,
        "td_bathtub": (td_phase[td_ber > 0], td_ber[td_ber > 0]),
        "td_floor": td_floor, "n_td": args.symbols,
    })
    print(f"\nwrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
