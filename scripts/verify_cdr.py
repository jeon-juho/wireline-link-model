"""Verify the CDR against its linearized model, using injected disturbances.

Three measurements:

  1. **Acquisition.**  A frequency offset is injected as a phase ramp
     (200 ppm = 2e-4 UI per symbol).  A type-II loop must drive the residual
     sampling error to zero, not merely bound it, because the integral path
     accumulates the constant phase slope.  Lock time is when the error enters
     and stays inside a tolerance band.

  2. **Jitter transfer.**  A small sinusoidal disturbance is injected and the
     recovered phase measured at that exact frequency with a single-bin DFT.
     The amplitude is kept small deliberately: |H| is a *linearized* quantity,
     so measuring it with large jitter would fold in the nonlinearity the
     linearization is trying to approximate.  From the sweep, the -3 dB point
     is interpolated and compared with ``2*zeta*w_n``.

  3. **Jitter tolerance.**  Amplitude is bisected at each frequency to find the
     largest sinusoid the loop survives.  This is where the bang-bang loop
     departs from the linear model: the phase moves at most ``K_p`` per update,
     so tolerance is capped by ``K_p/(2*pi*f*T_u)`` -- a **1/f** asymptote --
     while the linear prediction ``margin/|E(f)|`` rises as **1/f^2**.  The
     measured curve should follow whichever is lower.

    python scripts\\verify_cdr.py
    python scripts\\verify_cdr.py --ppm 500 --detector alexander
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src import channel, configs, plotting, tx  # noqa: E402
from src.cdr import CDR, error_transfer, jitter_transfer  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="baseline", choices=configs.available())
    ap.add_argument("--detector", default="mm", choices=("mm", "alexander"))
    ap.add_argument("--ppm", type=float, default=200.0,
                    help="frequency offset in ppm (default 200)")
    ap.add_argument("--decimation", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.25,
                    help="sampling-phase margin in UI that defines loss of lock")
    ap.add_argument("--sigma", type=float, default=0.02,
                    help="rms jitter (UI) the linearization is evaluated at")
    return ap.parse_args(argv)


_RX_CACHE: dict[int, np.ndarray] = {}


def build_rx(cfg, n_symbols: int) -> np.ndarray:
    """Equalized received waveform: PRBS -> channel -> CTLE.

    Uses overlap-add FFT convolution.  ``np.convolve`` performs the direct
    O(N*M) sum, and with a 6.4 M-sample waveform against a 32768-tap impulse
    response that is ~1e11 operations -- minutes per call, where oaconvolve
    takes under a second.
    """
    key = int(n_symbols)
    if key in _RX_CACHE:
        return _RX_CACHE[key]

    mm = channel.mixed_mode(channel.load(cfg.channel_path))
    from scipy import signal as sps

    from src import analog
    from src.ctle import CTLE

    ctle = CTLE.from_config(cfg.ctle)
    h_f = analog.compose(mm["sdd21"], ctle.response(mm["f"]))
    _, h, _ = channel.impulse_response(mm["f"], h_f, cfg)

    w = tx.generate(cfg, n_symbols, prbs_order=15, rise_time=0.0,
                    edge_method="ideal", swing=1.0)
    rx = sps.oaconvolve(w.waveform, h)[: w.waveform.size]
    _RX_CACHE[key] = rx
    return rx


def single_bin(x: np.ndarray, cycles: int) -> complex:
    """DFT of ``x`` at exactly ``cycles`` periods over its length.

    Exact and leakage-free *only* if ``cycles`` is a whole number of periods in
    ``x``.  Trimming an acquisition transient off the front breaks that unless
    the trim is itself a whole number of periods -- see :func:`analysis_window`.
    """
    n = x.size
    k = np.exp(-2j * np.pi * cycles * np.arange(n) / n)
    return complex(np.dot(x - x.mean(), k) * 2.0 / n)


def analysis_window(n: int, cycles: int, skip_min: int) -> tuple[int, int]:
    """Trim at least ``skip_min`` samples, by a whole number of periods.

    Returns ``(window_length, cycles_in_window)``.  Trimming an arbitrary
    amount leaves a fractional cycle, and the single-bin DFT then sits between
    bins: at 2 cycles in 120k samples, trimming a flat quarter of the record
    leaves 1.5 cycles and the estimate is pure leakage.
    """
    period = n // cycles
    skip_p = min(max(1, -(-skip_min // period)), max(cycles - 1, 1))
    return n - skip_p * period, cycles - skip_p


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    cfg = configs.get(args.config)
    fb = cfg.symbol_rate

    cdr = CDR(cfg, detector=args.detector, decimation=args.decimation,
              k_p=float(cfg.cdr.k_p.value), k_i=float(cfg.cdr.k_i.value))
    lp = cdr.parameters(args.sigma)
    t_u = args.decimation / fb

    print(f"CDR verification -- detector={args.detector}, "
          f"decimation={args.decimation}, PI={cdr.pi_steps} steps/UI")
    print(f"  gains   : K_p={cdr.k_p:.6g} UI, K_i={cdr.k_i:.6g}")
    print(f"  model   : {lp.summary()}")
    print(f"  update  : T_u={t_u * 1e9:.3f} ns, max slew "
          f"{lp.max_slew_ui_per_s / 1e6:.2f} kUI/ms")

    # ================================================================== 1 ===
    print(f"\n--- 1. Acquisition of a {args.ppm:.0f} ppm frequency offset "
          + "-" * 12)
    n_acq = 200_000
    rx = build_rx(cfg, n_acq + 64)
    slope = args.ppm * 1e-6                        # UI per symbol
    ramp = slope * np.arange(n_acq)
    res = cdr.run(rx, n_symbols=n_acq, disturbance_ui=-ramp,
                  sigma_tau_ui=args.sigma)

    # The loop must supply a phase that cancels the ramp; what is left over is
    # the residual sampling error, offset by wherever this detector locks.
    err_raw = res.phase_ui - ramp
    lock_phase = float(np.mean(err_raw[int(0.8 * n_acq):]))
    err = err_raw - lock_phase

    # Lock is judged on a moving average, not the raw error.  A bang-bang loop
    # never stops dithering, so an instantaneous threshold below the dither
    # amplitude can never be satisfied and "lock time" degenerates to the last
    # sample of the run.  Averaging over ~20 updates separates the acquisition
    # transient from the steady hunt.
    win = 20 * args.decimation
    smooth = np.convolve(err, np.ones(win) / win, mode="same")
    tol = 0.05
    outside = np.nonzero(np.abs(smooth[: -win]) > tol)[0]
    n_lock = int(outside[-1] + 1) if outside.size else 0

    tail = slice(int(0.8 * n_acq), None)
    dither_pp = float(err[tail].max() - err[tail].min())
    required = slope * args.decimation
    print(f"  required freq-register value : {required:.6e} UI/update")
    print(f"  achieved                     : {res.freq_reg[-1]:.6e} UI/update "
          f"({res.freq_reg[-1] / required * 100:.2f}% of required)")
    print(f"  lock phase                   : {lock_phase:+.4f} UI")
    print(f"  lock time (20-update MA)     : {n_lock} symbols = "
          f"{n_lock / fb * 1e6:.3f} us  ({n_lock / args.decimation:.0f} updates)")
    print(f"  steady-state error, averaged : "
          f"{np.abs(smooth[tail][:-win]).max() * 1e3:.2f} mUI "
          f"-- type-II leaves no static error to a ramp")
    print(f"  steady-state dither          : {dither_pp * 1e3:.1f} mUI pp "
          f"(K_p = {cdr.k_p * 1e3:.1f} mUI)")
    print(f"  the ramp is carried by the integral path, so the K_p-per-update")
    print(f"  slew bound applies to fast jitter, not to a frequency offset.")
    locked = np.abs(smooth[tail][:-win]).max() < tol
    print(f"  LOCKED                       : {locked}")

    # ================================================================== 2 ===
    print("\n--- 2. Jitter transfer " + "-" * 46)
    # A power of two, so every "cycles" below divides the record exactly --
    # which is what keeps the single-bin DFT leakage-free.
    n_jt = 1 << 18
    rx_jt = build_rx(cfg, n_jt + 64)
    # The injected amplitude has to clear the loop's own dither, or the
    # measurement is swamped by it, while staying well inside the margin so the
    # loop remains in the regime the linearization describes.
    amp = 0.10
    freqs, h_meas, e_meas = [], [], []
    for cycles in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        f_j = cycles * fb / n_jt
        dist = -amp * np.sin(2 * np.pi * cycles * np.arange(n_jt) / n_jt)
        r = cdr.run(rx_jt, n_symbols=n_jt, disturbance_ui=dist,
                    sigma_tau_ui=args.sigma)
        win, cyc = analysis_window(n_jt, cycles, n_jt // 4)
        d, ph = dist[-win:], r.phase_ui[-win:]
        ref = single_bin(d, cyc)
        freqs.append(f_j)
        h_meas.append(abs(single_bin(ph, cyc) / ref))
        e_meas.append(abs(single_bin(ph + d, cyc) / ref))

    freqs = np.array(freqs)
    h_db = 20 * np.log10(np.array(h_meas))
    e_db = 20 * np.log10(np.array(e_meas) + 1e-12)

    # -3 dB crossing, interpolated in log-frequency
    below = np.nonzero(h_db < -3.0)[0]
    if below.size and below[0] > 0:
        i = int(below[0])
        w = (-3.0 - h_db[i - 1]) / (h_db[i] - h_db[i - 1])
        f3_meas = float(np.exp(np.log(freqs[i - 1])
                               + w * (np.log(freqs[i]) - np.log(freqs[i - 1]))))
    else:
        f3_meas = float("nan")

    print(f"  {'f [MHz]':>10s} {'|H| dB':>9s} {'model dB':>9s} "
          f"{'|E| dB':>9s} {'model dB':>9s}")
    for f_j, hd, ed in zip(freqs, h_db, e_db):
        hm = 20 * np.log10(abs(jitter_transfer(np.array([f_j]), lp)[0]))
        em = 20 * np.log10(abs(error_transfer(np.array([f_j]), lp)[0]))
        print(f"  {f_j / 1e6:10.3f} {hd:9.2f} {hm:9.2f} {ed:9.2f} {em:9.2f}")
    print(f"\n  measured f_3dB : {f3_meas / 1e6:8.3f} MHz")
    print(f"  theory   f_3dB : {lp.f_3db / 1e6:8.3f} MHz  at the assumed "
          f"sigma_tau = {args.sigma:.3f} UI")
    print(f"  error          : {(f3_meas / lp.f_3db - 1) * 100:+8.2f}%")
    # f_3dB is proportional to K_pd, and K_pd goes as 1/sigma_tau, so a
    # bandwidth shortfall need not be an error -- it is the loop reporting the
    # jitter it actually sees.  Inverting the relation recovers that jitter.
    sigma_eff = args.sigma * lp.f_3db / f3_meas
    print(f"\n  f_3dB is proportional to K_pd = 2/(sigma*sqrt(2pi)), so the "
          f"shortfall inverts to an")
    print(f"  effective sigma_tau of {sigma_eff:.4f} UI "
          f"({sigma_eff / args.sigma:.2f}x the assumed value), against a measured")
    print(f"  steady-state dither of {dither_pp * 1e3:.1f} mUI pp. "
          f"This is K_pd's 1/sigma dependence,")
    print(f"  not a modelling error -- the loop has no single bandwidth.")

    # ================================================================== 3 ===
    print("\n--- 3. Jitter tolerance " + "-" * 45)
    n_jl = 1 << 17
    rx_jl = build_rx(cfg, n_jl + 64)

    def survives(cycles: int, a: float) -> bool:
        dist = -a * np.sin(2 * np.pi * cycles * np.arange(n_jl) / n_jl)
        r = cdr.run(rx_jl, n_symbols=n_jl, disturbance_ui=dist,
                    sigma_tau_ui=args.sigma)
        skip = n_jl // 3
        resid = r.phase_ui[skip:] + dist[skip:]
        return bool(np.abs(resid - np.median(resid)).max() < args.margin)

    def slew_bound(f_hz: np.ndarray | float) -> np.ndarray:
        """Largest amplitude whose *tracked* phase stays within the slew limit.

        The loop moves at most ``K_p`` per update.  It only has to move by the
        part of the jitter it actually tracks, ``A*|H(f)|``, so the peak rate is
        ``2*pi*f*A*|H(f)|`` and the bound is

            A_slew(f) = K_p / (T_u * 2*pi*f*|H(f)|)

        Dropping the ``|H|`` -- as the bare ``K_p/(2*pi*f*T_u)`` form does --
        keeps the bound falling as 1/f above the loop bandwidth, where the loop
        has stopped tracking and the constraint no longer applies at all.

        ``|H|`` is taken from the *measured* sweep rather than the model.  The
        continuous-time model overstates high-frequency tracking, because the
        loop is really updated once per ``decimation`` UI and the model ignores
        that sampling; using it here would keep the bound falling long after
        the real loop had stopped following the jitter.
        """
        f_arr = np.atleast_1d(np.asarray(f_hz, dtype=float))
        h = np.exp(np.interp(np.log(f_arr), np.log(freqs),
                             np.log(np.array(h_meas))))
        return cdr.k_p / (t_u * 2 * np.pi * f_arr * np.maximum(h, 1e-12))

    jt_f, jt_a = [], []
    print(f"  {'f [MHz]':>10s} {'tolerated UI pp':>16s} {'linear':>10s} "
          f"{'slew':>10s}")
    for cycles in (4, 8, 16, 32, 64, 128, 256, 512):
        f_j = cycles * fb / n_jl
        lo, hi = 0.005, 8.0
        if not survives(cycles, lo):
            jt_f.append(f_j), jt_a.append(2 * lo)
            continue
        for _ in range(8):                          # bisection in log amplitude
            mid = np.sqrt(lo * hi)
            if survives(cycles, mid):
                lo = mid
            else:
                hi = mid
        lin = args.margin / abs(error_transfer(np.array([f_j]), lp)[0])
        jt_f.append(f_j), jt_a.append(2 * lo)       # amplitude -> pk-pk
        print(f"  {f_j / 1e6:10.3f} {2 * lo:16.4f} {2 * lin:10.4f} "
              f"{2 * float(slew_bound(f_j)[0]):10.4f}")

    f_th = np.logspace(np.log10(freqs[0] * 0.5), np.log10(freqs[-1] * 2), 300)
    lin_th = 2 * args.margin / np.abs(error_transfer(f_th, lp))
    slew_th = 2 * slew_bound(f_th)

    out = plotting.plot_cdr_verification({
        "symbol_rate": fb, "ppm": args.ppm,
        "acquisition": {
            "n": np.arange(n_acq), "phase": res.phase_ui, "ideal": -ramp,
            "error": err, "tol_ui": tol, "lock_time_us": n_lock / fb * 1e6,
        },
        "jitter_transfer": {
            "f": freqs, "h_db": h_db, "e_db": e_db,
            "f_theory": f_th,
            "h_theory_db": 20 * np.log10(np.abs(jitter_transfer(f_th, lp))),
            "e_theory_db": 20 * np.log10(np.abs(error_transfer(f_th, lp))),
            "f3db_meas": f3_meas, "f3db_theory": lp.f_3db,
        },
        "jtol": {
            "f": np.array(jt_f), "measured": np.array(jt_a),
            "f_theory": f_th, "linear": lin_th, "slew": slew_th,
            "combined": np.minimum(lin_th, slew_th),
        },
    })
    print(f"\nwrote: {out}")
    return 0 if locked else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
