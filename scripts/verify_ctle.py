"""Measure the discrete-time CTLE's real frequency response against H(s).

The response is *measured*, not read off the filter coefficients: an impulse is
pushed through ``CTLE.apply`` -- the same ``lfilter`` path the time-domain
engine uses -- and the output is transformed.  Since the impulse response fully
characterises an LTI filter this is exact, and unlike ``freqz`` it validates the
whole application path rather than just the coefficient design.  The two are
compared against each other as a separate check.

Three discretization modes are compared, because the difference between them is
the entire point of prewarping:

    no prewarp             plain bilinear; singularities land low, error ~ f^2
    prewarp                each singularity pre-distorted onto target
    prewarp + match_peak   singularities scaled so the *peak* lands on target

A PRBS-driven estimate is also computed, to show why the impulse is the right
stimulus: an NRZ spectrum has nulls at multiples of the symbol rate, and the
transfer-function estimate is meaningless there.

    python scripts\\verify_ctle.py
    python scripts\\verify_ctle.py --boost 10 --peak 8e9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from src import configs, plotting, tx  # noqa: E402
from src.ctle import CTLE, refine_peak  # noqa: E402

MODES = {
    "no prewarp": dict(prewarp=False, match_peak=False),
    "prewarp": dict(prewarp=True, match_peak=False),
    "prewarp + match_peak": dict(prewarp=True, match_peak=True),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="baseline", choices=configs.available())
    ap.add_argument("--boost", type=float, default=13.19,
                    help="target boost in dB (default: this channel's Nyquist loss)")
    ap.add_argument("--peak", type=float, default=None,
                    help="target peak frequency in Hz (default: Nyquist)")
    ap.add_argument("--fp2", type=float, default=13.9e9,
                    help="second (output-node) pole in Hz")
    ap.add_argument("--nfft", type=int, default=1 << 18,
                    help="impulse-response length for the measurement")
    return ap.parse_args(argv)


def measure(ctle: CTLE, cfg, nfft: int, **mode) -> dict:
    """Measure the discrete filter by driving an impulse through ``apply``.

    ``rfft(delta) == 1``, so the transform of the output *is* the transfer
    function -- no division, hence no conditioning problem.
    """
    x = np.zeros(nfft)
    x[0] = 1.0
    y = ctle.apply(x, cfg, **mode)

    h = np.fft.rfft(y)
    f = np.fft.rfftfreq(nfft, cfg.dt)
    mag = np.abs(h)

    f_peak, mag_peak = refine_peak(f, mag)
    dc = mag[0]
    return {
        "f": f,
        "mag_db": 20.0 * np.log10(mag + 1e-30),
        "phase_deg": np.degrees(np.unwrap(np.angle(h))),
        "peak_hz": f_peak,
        "boost_db": 20.0 * np.log10(mag_peak / dc),
        "dc_gain": float(dc),
        "impulse": y,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    cfg = configs.get(args.config)
    peak_target = cfg.nyquist if args.peak is None else args.peak

    ctle = CTLE.from_spec(args.boost, peak_target, args.fp2)

    print("CTLE verification")
    print(f"  target : boost {args.boost:.3f} dB, peak "
          f"{peak_target / 1e9:.3f} GHz, f_p2 {args.fp2 / 1e9:.3f} GHz")
    print(f"  solved : f_z {ctle.f_z / 1e9:.4f} GHz, f_p1 {ctle.f_p1 / 1e9:.4f} GHz")
    print(f"           asymptotic boost {ctle.asymptotic_boost_db:.2f} dB "
          f"(deficit {ctle.asymptotic_boost_db - ctle.boost_db:.2f} dB)")
    print(f"  grid   : fs {cfg.fs / 1e9:.0f} GHz ({cfg.samples_per_ui} samples/UI), "
          f"N = {args.nfft}, bin = {cfg.fs / args.nfft / 1e6:.3f} MHz")

    # -- analog reference on the same grid --------------------------------
    f = np.fft.rfftfreq(args.nfft, cfg.dt)
    f_safe = np.maximum(f, 1e3)
    analog_db, analog_phase = ctle.bode(f_safe)
    a_peak, a_mag = refine_peak(f_safe, np.abs(ctle.response(f_safe)))
    print(f"\nanalog H(s) reference (exact)")
    print(f"  peak  {a_peak / 1e9:.6f} GHz   boost {ctle.boost_db:.6f} dB")

    # -- measured ----------------------------------------------------------
    band = f <= 2.0 * cfg.nyquist          # the band the link actually occupies
    results: dict[str, dict] = {}
    print("\nmeasured from the discrete-time filter (impulse -> FFT through apply())")
    print(f"  {'mode':<22s} {'peak [GHz]':>11s} {'err %':>9s} "
          f"{'boost [dB]':>11s} {'err %':>9s} {'d(dB)':>8s} {'max dev':>9s}")
    for tag, mode in MODES.items():
        m = measure(ctle, cfg, args.nfft, **mode)
        m["peak_err"] = m["peak_hz"] / peak_target - 1.0
        m["boost_err"] = m["boost_db"] / args.boost - 1.0
        m["dev_db"] = float(np.abs(m["mag_db"][band] - analog_db[band]).max())
        results[tag] = m
        print(f"  {tag:<22s} {m['peak_hz'] / 1e9:11.6f} "
              f"{m['peak_err'] * 100:+9.4f} {m['boost_db']:11.6f} "
              f"{m['boost_err'] * 100:+9.4f} {m['boost_db'] - args.boost:+8.4f} "
              f"{m['dev_db']:9.4f}")
    print(f"  {'':22s} {'':11s} {'':9s} {'':11s} {'':9s} {'':8s}"
          f"  (dev over DC-{2 * cfg.nyquist / 1e9:.0f} GHz)")

    # -- coefficient response must agree with the measured one -------------
    print("\ncross-checks")
    for tag, mode in MODES.items():
        b, a = ctle.discretize(cfg, **mode)
        _, h_fz = sps.freqz(b, a, worN=2 * np.pi * f / cfg.fs)
        diff = float(np.abs(np.abs(h_fz) - 10 ** (results[tag]["mag_db"] / 20)).max())
        print(f"  freqz vs. measured impulse, {tag:<22s} max |diff| = {diff:.3e}"
              f"   {'ok' if diff < 1e-9 else 'MISMATCH'}")

    dc_err = abs(results["prewarp"]["dc_gain"] / ctle.a_dc - 1.0)
    print(f"  DC gain vs A_dc                              "
          f"rel err   = {dc_err:.3e}   {'ok' if dc_err < 1e-9 else 'MISMATCH'}")

    # -- why the impulse and not a data pattern ----------------------------
    # The pattern occupies only the first half of the record and the rest is
    # zeros.  This matters: lfilter performs *linear* convolution while the FFT
    # implies a *circular* one, so a stimulus that is non-zero at both ends of
    # the record makes Y/X disagree with H by an edge artifact -- 2.4 dB here,
    # which would otherwise be misread as a conditioning problem.  Zero-padding
    # lets the linear convolution finish inside the record, leaving |X|
    # conditioning as the only error source, which is the point being made.
    n_sym = (args.nfft // 2) // cfg.samples_per_ui
    wave = np.zeros(args.nfft)
    pattern = tx.generate(cfg, n_sym, prbs_order=15, rise_time=0.0,
                          edge_method="ideal").waveform
    wave[: pattern.size] = pattern

    yq = ctle.apply(wave, cfg, prewarp=True)
    X, Y = np.fft.rfft(wave), np.fft.rfft(yq)
    ref = np.abs(np.fft.rfft(results["prewarp"]["impulse"]))
    null = int(np.argmin(np.abs(f - cfg.symbol_rate)))

    print(f"\nPRBS-driven estimate (why the impulse is the right stimulus)")
    print(f"  {'|X| threshold':<22s} {'usable bins':>14s} {'worst error':>14s}")
    for thr in (0.05, 0.01, 1e-3, 1e-5):
        keep = np.abs(X) > thr * np.abs(X).max()
        err = float(np.abs(20 * np.log10(np.abs(Y[keep] / X[keep]) / ref[keep])).max())
        print(f"  > {thr:<20.0e} {keep.sum():>8d} ({keep.sum() / f.size * 100:4.1f}%) "
              f"{err:>13.2e} dB")
    print(f"  |X| at the {cfg.symbol_rate / 1e9:.0f} GHz NRZ null : "
          f"{np.abs(X[null]) / np.abs(X).max():.2e} of peak -- "
          f"an NRZ pattern carries no energy there, so Y/X is 0/0")

    out = plotting.plot_ctle_verification({
        "f": f_safe,
        "analog_db": analog_db,
        "analog_phase": analog_phase,
        "modes": results,
        "target_peak": peak_target,
        "nyquist": cfg.nyquist,
        "fs": cfg.fs,
    })
    print(f"\nwrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
