"""Sanity-check the channel .s4p: mixed-mode response, impulse, pulse, cursors.

Prerequisite step from docs/architecture.md -- run this before trusting any
equalization result, since every downstream number derives from the pulse
response produced here.

    python scripts\\plot_channel.py
    python scripts\\plot_channel.py --config no_eq
    python scripts\\plot_channel.py --channel path\\to\\other.s4p
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import skrf as rf  # noqa: E402

from src import analog, channel, configs, plotting  # noqa: E402

N_SHOW = 12  # cursors printed; totals always use the full cfg.isi_depth


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="baseline", choices=configs.available(),
                    help="experiment configuration (default: baseline)")
    ap.add_argument("--channel", type=Path, default=None,
                    help="override the .s4p path in the configuration")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    cfg = configs.get(args.config)
    if args.channel is not None:
        cfg = replace(cfg, channel_path=args.channel)
    if not cfg.channel_path.exists():
        print(f"channel file not found: {cfg.channel_path}", file=sys.stderr)
        return 1

    print(f"config  : {args.config}")
    print(f"channel : {cfg.channel_path.name}")

    probe = rf.Network(str(cfg.channel_path))
    order = channel.detect_port_order(probe)
    mm = channel.mixed_mode(channel.load(cfg.channel_path))
    f, sdd21 = mm["f"], mm["sdd21"]

    print(f"ports   : {probe.nports}, detected ordering = {order}")
    print(f"freq    : {f[0] / 1e9:.3f} .. {f[-1] / 1e9:.1f} GHz, "
          f"{f.size} points, step {(f[1] - f[0]) / 1e6:.1f} MHz")
    print(f"link    : {cfg.bit_rate / 1e9:.0f} Gbps NRZ, UI = {cfg.ui * 1e12:.2f} ps, "
          f"Nyquist = {cfg.nyquist / 1e9:.0f} GHz")
    print(f"grid    : M = {cfg.samples_per_ui} samples/UI, fs = {cfg.fs / 1e9:.0f} GHz, "
          f"N = {cfg.n_time}, window = {cfg.time_window * 1e9:.1f} ns")
    print(f"depth   : {cfg.n_pre} pre / {cfg.isi_depth} post cursors")
    print(f"blocks  : CTLE {'on' if cfg.ctle.enabled else 'off'}, "
          f"DFE {'on' if cfg.dfe.enabled else 'off'} "
          f"({int(cfg.dfe.n_taps.value)} tap), "
          f"CDR {'on' if cfg.cdr.enabled else 'off'}")

    print("\ninsertion loss")
    for ftgt in (1e9, 2e9, 4e9, cfg.nyquist, 12e9, 16e9):
        print(f"  {ftgt / 1e9:5.1f} GHz : {channel.loss_at(f, sdd21, ftgt):6.2f} dB")

    t, h = analog.impulse_response(f, sdd21, cfg)
    p = analog.pulse_response(h, cfg)
    sp = analog.split_pulse(p, cfg)

    total, err = analog.check_dc_gain(h, sdd21[0])
    print(f"\nDC self-test : sum(h) = {total:.6f}, Sdd21(0) = {np.real(sdd21[0]):.6f}, "
          f"error = {err:.2e}")
    print(f"peak delay   : {t[sp.k_peak] * 1e9:.3f} ns")

    # --- sampling phase -----------------------------------------------------
    n_taps = int(cfg.dfe.n_taps.value) if cfg.dfe.enabled else 0
    j_peak = sp.peak_phase
    j_lock, lock_ui = analog.mm_lock_phase(sp)
    eye_curve = analog.eye_vs_phase(sp, n_taps=n_taps)
    j_best = int(np.argmax(eye_curve))

    print(f"\nsampling phase (relative to pulse peak)")
    print(f"  peak      : {sp.phase_ui(j_peak):+.4f} UI")
    print(f"  MM lock   : {lock_ui:+.4f} UI   (h1 == h_-1)")
    print(f"  best eye  : {sp.phase_ui(j_best):+.4f} UI   "
          f"({eye_curve[j_best] * 1e3:.1f} mV with {n_taps}-tap DFE)")
    print(f"  lock cost : {(eye_curve[j_best] - eye_curve[j_lock]) * 1e3:.1f} mV "
          f"vs. the best phase")

    c = sp.at(j_lock)
    print(f"\ncursors (mV) at the MM lock phase, first {N_SHOW} postcursors "
          f"of {cfg.isi_depth}")
    for m, v in zip(range(-cfg.n_pre, N_SHOW + 1), c[: cfg.n_pre + N_SHOW + 1]):
        tag = "  <-- main" if m == 0 else ""
        print(f"  {m:+3d} : {v * 1e3:8.3f}{tag}")

    print("\npeak-distortion analysis")
    for label, j in (("peak phase", j_peak), ("MM lock phase", j_lock)):
        d = analog.peak_distortion(sp, j, n_taps=n_taps)
        print(f"  {label:<14s} main {d.main * 1e3:7.2f} mV | "
              f"ISI {d.isi * 1e3:7.2f} mV | "
              f"eye {d.eye * 1e3:8.2f} mV | "
              f"with {n_taps}-tap DFE {d.eye_with_dfe * 1e3:7.2f} mV "
              f"({'OPEN' if d.open else 'CLOSED'})")

    d = analog.peak_distortion(sp, j_lock, n_taps=max(n_taps, 2))
    print(f"\nZF tap weights at the MM lock phase (w_m = h_m/h0)")
    for i, w in enumerate(d.taps, start=1):
        flag = "" if abs(w) <= cfg.dfe.max_tap_weight else \
            f"  <-- exceeds |w| <= {cfg.dfe.max_tap_weight} (error propagation)"
        print(f"  w{i} = {w:+.4f}{flag}")

    print("\ntunable parameters")
    for dotted, b in sorted(cfg.knobs().items()):
        print(f"  {dotted:<20s} {b.describe()}")

    out = [
        plotting.plot_channel_response(mm, cfg),
        plotting.plot_impulse_pulse(t, h, p, sp, cfg, j=j_lock),
        plotting.plot_cursor_stem(sp, j=j_lock, n_show=20),
        plotting.plot_eye_vs_phase(sp, cfg, lock_phase_ui=lock_ui),
    ]
    print("\nwrote:")
    for o in out:
        print(f"  {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
