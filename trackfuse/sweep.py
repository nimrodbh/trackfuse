"""Many drives, not one: how the error moves when the conditions get worse.

    python -m trackfuse sweep --plot docs/robustness.png

One good run proves little: the seed may be lucky. Each point here is a number of random drives
(50 by default), and the chart shows the median with the band between the 10th and 90th percentile.

Two questions:
  1. GPS gets noisier (1 m to 12 m). What does fusing buy, with and without landmarks?
  2. GPS disappears (0 to 60 seconds). How far off does the estimate get before it comes back?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics, native, pyekf, sim

P0 = np.array([1.0, 1.0, 0.01, 0.25])
GPS_SIGMAS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)  # meters
OUTAGES = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0)  # seconds
OUTAGE_START = 20.0  # every outage begins here, after the filter has settled

INK, MUTED, GRID, PAPER = "#1c2430", "#6b7785", "#e3e7ec", "#fafbfc"
C_GPS, C_DR, C_EKF, C_MID = "#2e9e5b", "#e07b1a", "#1f5fd0", "#7a52c7"


def _engine():
    return native if native.available() else pyekf


def _fuse(run: sim.Run, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a run through the filter, optionally with some events left out. Returns (estimate, truth, times)."""
    types, vals, truth, times = run.types, run.vals, run.truth, run.times
    if keep is not None:
        types, vals, truth, times = types[keep], vals[keep], truth[keep], times[keep]
    scn = run.scenario
    est, _, _ = _engine().run(types, vals, run.x0, P0, scn.sigma_a, scn.sigma_w, 0.0, False)
    return est, truth, times


def gps_noise_sweep(sigmas=GPS_SIGMAS, runs: int = 50, duration: float = 80.0) -> dict[str, np.ndarray]:
    """Position RMSE over a whole drive, per GPS noise level and per random drive.

    Each array has shape (len(sigmas), runs):
      "gps"          the GPS fixes themselves, with no filter;
      "gps_imu"      the filter given GPS and motion sensors, no landmarks;
      "all"          the filter given GPS, motion sensors and landmarks.
    """
    out = {name: np.empty((len(sigmas), runs)) for name in ("gps", "gps_imu", "all")}
    for a, sigma in enumerate(sigmas):
        for b in range(runs):
            run = sim.generate(sim.Scenario(duration=duration, sigma_gps=sigma), b + 1)
            out["gps"][a, b] = metrics.gps_only_rmse(run)
            est, truth, _ = _fuse(run)
            out["all"][a, b] = metrics.pos_rmse(est, truth)
            est, truth, _ = _fuse(run, run.types != sim.LANDMARK)
            out["gps_imu"][a, b] = metrics.pos_rmse(est, truth)
    return out


def outage_sweep(lengths=OUTAGES, runs: int = 50, tail: float = 10.0) -> dict[str, np.ndarray]:
    """The worst position error while GPS is away, per outage length and per random drive.

    The worst moment, not the last one: on a figure-eight a heading error pushes the estimate away and then
    partly back as the road curves, so the error at the end of an outage goes up and down with the geometry.
    The worst error can only grow with the length of the outage. A length of 0 looks at one second of
    normal driving, as the baseline.

    Each array has shape (len(lengths), runs):
      "no_landmarks"   motion sensors alone carry the estimate through the outage;
      "landmarks"      landmarks keep correcting it.
    """
    out = {name: np.empty((len(lengths), runs)) for name in ("no_landmarks", "landmarks")}
    for a, length in enumerate(lengths):
        back = OUTAGE_START + length
        outage = (OUTAGE_START, back) if length > 0 else None
        for b in range(runs):
            run = sim.generate(sim.Scenario(duration=back + tail, gps_outage=outage), b + 1)
            for name, keep in (("landmarks", None), ("no_landmarks", run.types != sim.LANDMARK)):
                est, truth, times = _fuse(run, keep)
                away = (times >= OUTAGE_START) & (times < OUTAGE_START + max(length, 1.0))
                out[name][a, b] = np.max(np.hypot(*(est[away, :2] - truth[away, :2]).T))
    return out


def summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median, 10th and 90th percentile across the random drives of each point."""
    return np.median(values, axis=1), np.percentile(values, 10, axis=1), np.percentile(values, 90, axis=1)


def table(noise: dict[str, np.ndarray], outage: dict[str, np.ndarray], sigmas=GPS_SIGMAS, lengths=OUTAGES) -> str:
    rows = ["GPS noise (m)   GPS alone   GPS + motion   + landmarks      (median position RMSE, m)"]
    for a, sigma in enumerate(sigmas):
        g, m, f = (np.median(noise[k][a]) for k in ("gps", "gps_imu", "all"))
        rows.append(f"{sigma:13.0f} {g:11.2f} {m:14.2f} {f:12.2f}")
    rows += ["", "GPS outage (s)   motion only   with landmarks   (median of the worst error during the outage, m)"]
    for a, length in enumerate(lengths):
        n, w = (np.median(outage[k][a]) for k in ("no_landmarks", "landmarks"))
        rows.append(f"{length:14.0f} {n:13.2f} {w:16.2f}")
    return "\n".join(rows)


def plot(noise: dict[str, np.ndarray], outage: dict[str, np.ndarray], out: str | Path, sigmas=GPS_SIGMAS, lengths=OUTAGES) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})  # fmt: skip
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.6), dpi=150, facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.975, top=0.78, bottom=0.12, wspace=0.2)
    runs = next(iter(noise.values())).shape[1]
    fig.text(0.07, 0.93, "Not one lucky run", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.07, 0.875, f"{runs} random drives per point. Line: the median. Band: 10th to 90th percentile.",
             fontsize=12, color=MUTED)  # fmt: skip

    def draw(ax, x, values, color, label, style="-"):
        mid, low, high = summary(values)
        ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
        ax.plot(x, mid, style, color=color, lw=2, marker="o", ms=5, label=label)

    draw(left, sigmas, noise["gps"], C_GPS, "GPS alone")
    draw(left, sigmas, noise["gps_imu"], C_MID, "GPS + motion sensors")
    draw(left, sigmas, noise["all"], C_EKF, "GPS + motion sensors + landmarks")
    left.set_title("When GPS gets noisier", loc="left", fontsize=13, color=INK, pad=10)
    left.set_xlabel("GPS noise (meters)")
    left.set_ylabel("position error over the drive")

    draw(right, lengths, outage["no_landmarks"], C_DR, "motion sensors only")
    draw(right, lengths, outage["landmarks"], C_EKF, "with landmarks")
    right.set_title("When GPS disappears", loc="left", fontsize=13, color=INK, pad=10)
    right.set_xlabel("seconds without GPS")
    right.set_ylabel("worst position error during the outage")

    for ax in (left, right):
        ax.set_yscale("log")
        ax.set_ylim(0.03, 30)
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(["10 cm", "1 m", "10 m"])
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_facecolor(PAPER)
        ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out
