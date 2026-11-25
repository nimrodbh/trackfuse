"""What the filter does not model, measured.

    python -m trackfuse stress --plot docs/stress.png

The standard simulator errs exactly the way the filter assumes: white noise on every sensor. Real
sensors also lie in ways the filter knows nothing about. This module switches four such effects
on, one at a time and then together, and measures what each one costs over many random drives:

  gyro bias            the yaw-rate sensor reads a constant 0.5 degrees per second too high
  accelerometer bias   the acceleration sensor reads a constant 0.1 m/s^2 too high
  side slip            in a turn the car moves about one degree off the way it points
  GPS errors linger    a GPS error fades over about 30 seconds, so a few fixes do not average it out

Each filter is run twice: as built, and "loose", told that its motion sensors are ten times
noisier than their specification. Loosening is the cheap remedy (it makes the filter lean on its
measurements). The proper one is to estimate the biases as part of the state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics, native, pyekf, sim

P0 = np.array([1.0, 1.0, 0.01, 0.25])
LOOSE = 10.0  # the remedy: assume the motion sensors are this many times noisier than specified

CASES: dict[str, dict[str, float]] = {
    "as the filter assumes": {},
    "gyro bias": {"gyro_bias": float(np.radians(0.5))},
    "accelerometer bias": {"accel_bias": 0.1},
    "side slip": {"slip": 0.05},
    "GPS errors linger": {"gps_correlation": 30.0},
}
CASES["all four"] = {key: value for case in CASES.values() for key, value in case.items()}

INK, MUTED, GRID, PAPER = "#1c2430", "#6b7785", "#e3e7ec", "#fafbfc"
C_GPS, C_BUILT, C_LOOSE = "#2e9e5b", "#c0392b", "#1f5fd0"


def _fuse(run: sim.Run, keep: np.ndarray | None, loosen: float) -> tuple[float, float]:
    """Replay a run through the filter. Returns (position RMSE, average NEES)."""
    types, vals, truth = run.types, run.vals, run.truth
    if keep is not None:
        types, vals, truth = types[keep], vals[keep], truth[keep]
    scn = run.scenario
    engine = native if native.available() else pyekf
    est, cov, _ = engine.run(types, vals, run.x0, P0, scn.sigma_a * loosen, scn.sigma_w * loosen)
    return metrics.pos_rmse(est, truth), float(np.mean(metrics.nees(est, cov, truth)))


def measure(runs: int = 50, duration: float = 80.0, cases: dict[str, dict[str, float]] | None = None) -> dict:
    """For every case, arrays over `runs` random drives.

      "gps"                          the GPS fixes themselves, no filter
      "gps_imu", "gps_imu_loose"     the filter with GPS and motion sensors (as built, loosened)
      "all", "all_loose"             the same with landmarks too
    and for each filter name, name + "_nees": how honest its own uncertainty is. About 4 is honest
    for this 4-state filter; far above 4 means it is surer of itself than it has any right to be.
    """
    results: dict[str, dict[str, np.ndarray]] = {}
    for case, effects in (cases or CASES).items():
        row = {name: np.empty(runs) for name in ("gps",)}
        for name in ("gps_imu", "gps_imu_loose", "all", "all_loose"):
            row[name], row[name + "_nees"] = np.empty(runs), np.empty(runs)
        for b in range(runs):
            run = sim.generate(sim.Scenario(duration=duration, **effects), b + 1)
            row["gps"][b] = metrics.gps_only_rmse(run)
            no_landmarks = run.types != sim.LANDMARK
            for name, keep, loosen in (
                ("gps_imu", no_landmarks, 1.0),
                ("gps_imu_loose", no_landmarks, LOOSE),
                ("all", None, 1.0),
                ("all_loose", None, LOOSE),
            ):
                row[name][b], row[name + "_nees"][b] = _fuse(run, keep, loosen)
        results[case] = row
    return results


def table(results: dict) -> str:
    head = (
        f"{'median over the drives':24s} {'GPS':>6s} | {'GPS + motion sensors':^27s} | {'+ landmarks':^27s}\n"
        f"{'':24s} {'alone':>6s} | {'as built':>8s} {'NEES':>5s} {'loose':>6s} {'NEES':>5s} |"
        f" {'as built':>8s} {'NEES':>5s} {'loose':>6s} {'NEES':>5s}"
    )
    rows = [head]
    for case, row in results.items():
        m = {name: float(np.median(values)) for name, values in row.items()}
        rows.append(
            f"{case:24s} {m['gps']:6.2f} | {m['gps_imu']:8.2f} {m['gps_imu_nees']:5.0f} {m['gps_imu_loose']:6.2f}"
            f" {m['gps_imu_loose_nees']:5.0f} | {m['all']:8.2f} {m['all_nees']:5.0f} {m['all_loose']:6.2f}"
            f" {m['all_loose_nees']:5.0f}"
        )
    return "\n".join(rows)


def plot(results: dict, out: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})  # fmt: skip
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.6), dpi=150, facecolor="white", sharey=True)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.76, bottom=0.12, wspace=0.08)
    runs = len(next(iter(results.values()))["gps"])
    fig.text(0.03, 0.93, "When the sensors lie", fontsize=20, fontweight="bold", color=INK)
    fig.text(
        0.03, 0.875,
        f"Position error, median of {runs} random drives. Loose: the filter is told its motion sensors"
        f" are {LOOSE:.0f} times noisier than specified.",
        fontsize=12, color=MUTED,
    )  # fmt: skip
    cases = list(results)
    y = np.arange(len(cases))[::-1]  # first case on top
    for ax, built, loose, title in (
        (left, "gps_imu", "gps_imu_loose", "GPS + motion sensors"),
        (right, "all", "all_loose", "GPS + motion sensors + landmarks"),
    ):
        ax.barh(y + 0.19, [np.median(results[c][built]) for c in cases], height=0.36, color=C_BUILT, label="as built")
        ax.barh(y - 0.19, [np.median(results[c][loose]) for c in cases], height=0.36, color=C_LOOSE, label="loose")
        for position, case in zip(y, cases, strict=True):
            for offset, name in ((0.19, built), (-0.19, loose)):
                value = float(np.median(results[case][name]))
                ax.text(value * 1.12, position + offset, f"{value:.2f} m", va="center", fontsize=9, color=INK)
        gps = float(np.median(results[cases[0]]["gps"]))
        ax.axvline(gps, color=C_GPS, lw=1.6, ls=(0, (4, 3)))
        ax.text(gps * 1.08, len(cases) - 0.45, "GPS alone", color=C_GPS, fontsize=9, fontweight="bold", va="center")
        ax.set_xscale("log")
        ax.set_xlim(0.08, 60)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["10 cm", "1 m", "10 m"])
        ax.set_ylim(-0.6, len(cases) - 0.15)
        ax.set_yticks(y)
        ax.set_yticklabels(cases)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(PAPER)
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
    right.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out
