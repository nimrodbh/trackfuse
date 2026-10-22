"""Command line: `python -m trackfuse demo` (plot + accuracy table) or `python -m trackfuse bench`."""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import metrics, native, pyekf, sim

P0 = np.array([1.0, 1.0, 0.01, 0.25])


def _fuse(run, impl):
    s = run.scenario
    return impl.run(run.types, run.vals, run.x0, P0, s.sigma_a, s.sigma_w)


def demo(out: str | None) -> None:
    scn = sim.Scenario()
    run = sim.generate(scn, 1)
    impl = native if native.available() else pyekf
    est, _, _ = _fuse(run, impl)

    dt, dv = metrics.predict_only(run)
    dr, _, _ = impl.run(dt, dv, run.x0, P0, scn.sigma_a, scn.sigma_w)
    pred = run.types == sim.PREDICT
    gps = run.types == sim.GPS

    rows = [
        ("Raw GPS fixes (1 Hz)", metrics.gps_only_rmse(run)),
        ("Dead reckoning (IMU only)", metrics.pos_rmse(dr, run.truth[pred])),
        ("EKF: IMU + GPS + landmarks", metrics.pos_rmse(est, run.truth)),
    ]
    print(f"{'estimator':32s} position RMSE (m)")
    for name, val in rows:
        print(f"{name:32s} {val:8.2f}")

    if out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(run.truth[:, 0], run.truth[:, 1], "k-", lw=1.5, label="truth")
        ax.plot(dr[:, 0], dr[:, 1], "C1--", lw=1, label="dead reckoning")
        ax.scatter(run.vals[gps][:, 0], run.vals[gps][:, 1], s=8, c="C2", alpha=0.6, label="GPS fixes")
        ax.plot(est[:, 0], est[:, 1], "C0-", lw=1, label="EKF")
        ax.scatter(sim.LANDMARKS[:, 0], sim.LANDMARKS[:, 1], marker="^", c="C3", label="landmarks")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title("Figure-eight drive, simulated sensors")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")


def bench(duration: float, repeats: int) -> None:
    if not native.available():
        raise SystemExit("C++ core not built (make -C core build/libtrackfuse.so)")
    run = sim.generate(sim.Scenario(duration=duration), 1)
    n = len(run.types)

    def best(impl) -> float:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            impl.run(run.types, run.vals, run.x0, P0, run.scenario.sigma_a, run.scenario.sigma_w, 0.0, False)
            times.append(time.perf_counter() - t0)
        return min(times)

    py, cpp = best(pyekf), best(native)
    print(f"events: {n}")
    print(f"NumPy reference: {py * 1e3:8.1f} ms  ({n / py / 1e3:8.1f} k events/s)")
    print(f"C++ via ctypes : {cpp * 1e3:8.1f} ms  ({n / cpp / 1e3:8.1f} k events/s)")
    print(f"speedup        : {py / cpp:8.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser(prog="trackfuse")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--plot", default=None, help="write a trajectory plot to this PNG path")
    b = sub.add_parser("bench")
    b.add_argument("--duration", type=float, default=300.0)
    b.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "demo":
        demo(args.plot)
    else:
        bench(args.duration, args.repeats)


if __name__ == "__main__":
    main()
