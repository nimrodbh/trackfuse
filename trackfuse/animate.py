"""Render one drive as a short clip.

    python -m trackfuse animate docs/drive.gif      # needs matplotlib (it brings Pillow)
    python -m trackfuse animate docs/drive.mp4      # also needs ffmpeg
    python -m trackfuse animate frame.png --at 47   # one still frame

The clip is a picture of what the filter did, not a drawing of what it should do. The drive comes from
the simulator, it is replayed through the filter (the C++ core when it is built, the NumPy reference
otherwise), and every number on screen is computed from that run.

One 80 second drive tells three things:
  1. normal driving: GPS, motion sensors and landmarks together;
  2. GPS lost for 20 seconds: the landmarks hold the estimate;
  3. one GPS fix about 200 m from the truth: the filter refuses it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics, native, pyekf, sim

P0 = np.array([1.0, 1.0, 0.01, 0.25])
OUTAGE = (28.0, 48.0)  # seconds with no GPS
BAD_FIX_AT = 64.0  # the time of one GPS fix far from the truth
GATE_CHI2 = 13.82  # refuse a measurement this unlikely (chi-square, 2 degrees of freedom, 99.9 percent)

INK, MUTED, GRID, PAPER, ROAD = "#1c2430", "#6b7785", "#e3e7ec", "#fafbfc", "#c9d0d8"
C_GPS, C_DR, C_EKF, C_LM, C_BAD = "#2e9e5b", "#e07b1a", "#1f5fd0", "#8e44ad", "#c0392b"
HALF_W, HALF_H = 21.0, 15.5  # half the size, in meters, of the view that follows the car


@dataclass
class Story:
    """One drive and what each estimator made of it."""

    scenario: sim.Scenario
    engine: str  # "C++" or "NumPy"
    t: np.ndarray  # (n,) time of every event
    truth: np.ndarray  # (n, 4) the true state after every event
    est: np.ndarray  # (n, 4) the fused estimate after every event
    cov: np.ndarray  # (n, 4, 4) its covariance
    err: np.ndarray  # (n,) position error of the fused estimate, meters
    t_dr: np.ndarray  # (m,) times of the motion-sensor events
    dr: np.ndarray  # (m, 4) dead reckoning: motion sensors only
    err_dr: np.ndarray  # (m,)
    gps_t: np.ndarray  # (k,) times of the GPS fixes
    gps_xy: np.ndarray  # (k, 2) the fixes
    gps_truth: np.ndarray  # (k, 2) where the car really was
    gps_err: np.ndarray  # (k,)
    gps_bad: np.ndarray  # (k,) bool: a fix the scenario made wildly wrong
    gps_rejected: np.ndarray  # (k,) bool: the filter refused this fix


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("nan")


def build_story(seed: int = 1) -> Story:
    scn = sim.Scenario(duration=80.0, gps_outage=OUTAGE, outliers=[BAD_FIX_AT])
    run = sim.generate(scn, seed)
    engine = native if native.available() else pyekf
    est, cov, _ = engine.run(run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w, GATE_CHI2)
    dr_types, dr_vals = metrics.predict_only(run)
    dr, _, _ = engine.run(dr_types, dr_vals, run.x0, P0, scn.sigma_a, scn.sigma_w, 0.0, False)

    # A refused measurement leaves the state exactly as it was.
    unchanged = np.zeros(len(run.types), dtype=bool)
    unchanged[1:] = np.all(est[1:] == est[:-1], axis=1)

    moved, gps = run.types == sim.PREDICT, run.types == sim.GPS
    gps_xy, gps_truth = run.vals[gps][:, :2], run.truth[gps][:, :2]
    return Story(
        scenario=scn,
        engine="C++" if engine is native else "NumPy",
        t=run.times,
        truth=run.truth,
        est=est,
        cov=cov,
        err=np.hypot(*(est[:, :2] - run.truth[:, :2]).T),
        t_dr=run.times[moved],
        dr=dr,
        err_dr=np.hypot(*(dr[:, :2] - run.truth[moved][:, :2]).T),
        gps_t=run.times[gps],
        gps_xy=gps_xy,
        gps_truth=gps_truth,
        gps_err=np.hypot(*(gps_xy - gps_truth).T),
        gps_bad=np.isin(np.round(run.times[gps], 6), np.round(scn.outliers, 6)),
        gps_rejected=unchanged[gps],
    )


def caption(story: Story, now: float) -> str:
    """The line under the title. Every number in it is computed from the run up to `now`."""
    s = story
    seen = (s.gps_t <= now) & ~s.gps_bad
    if now >= s.scenario.duration:
        return (
            f"The whole {s.scenario.duration:.0f} second drive. GPS alone: {rmse(s.gps_err[~s.gps_bad]):.1f} m off. "
            f"Motion sensors only: {rmse(s.err_dr):.1f} m. Fused: {rmse(s.err):.2f} m."
        )
    if s.scenario.gps_outage and s.scenario.gps_outage[0] <= now < s.scenario.gps_outage[1] + 2.0:
        start, end = s.scenario.gps_outage
        inside = (s.t >= start) & (s.t <= min(now, end))
        return f"GPS lost for {end - start:.0f} seconds. Landmarks hold the estimate: {rmse(s.err[inside]):.2f} m off."
    recent_bad = np.flatnonzero(s.gps_bad & (s.gps_t <= now) & (s.gps_t > now - 7.0))
    if len(recent_bad):
        i = recent_bad[-1]
        verdict = "The filter refuses it." if s.gps_rejected[i] else "The filter takes it in."
        return f"A GPS fix {s.gps_err[i]:.0f} m from the truth arrives. {verdict}"
    if not seen.any():
        return "Fusing motion sensors, GPS and landmarks into one position."
    return (
        f"GPS alone: {rmse(s.gps_err[seen]):.1f} m off. "
        f"Fused with motion sensors and landmarks: {rmse(s.err[s.t <= now]):.2f} m."
    )


class _Scene:
    """The figure and its moving parts. `draw(now)` moves everything to one moment of the drive."""

    def __init__(self, story: Story, width: int, playback: float) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse, Rectangle
        from matplotlib.ticker import MultipleLocator

        self.s = s = story
        self.plt = plt
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.edgecolor": GRID,
                "axes.labelcolor": MUTED,
                "xtick.color": MUTED,
                "ytick.color": MUTED,
            }
        )
        self.fig = fig = plt.figure(figsize=(12.8, 7.2), dpi=width / 12.8, facecolor="white")
        grid = fig.add_gridspec(
            2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1.05],
            left=0.045, right=0.975, top=0.83, bottom=0.085, wspace=0.13, hspace=0.3,
        )  # fmt: skip
        self.ax = ax = fig.add_subplot(grid[:, 0])
        self.ax_map = ax_map = fig.add_subplot(grid[0, 1])
        self.ax_err = ax_err = fig.add_subplot(grid[1, 1])

        fig.text(0.045, 0.94, "Where is the car?", fontsize=22, fontweight="bold", color=INK)
        self.caption = fig.text(0.045, 0.895, "", fontsize=13.5, color=INK)
        fig.text(
            0.045, 0.026,
            f"Simulated drive and sensors. Filter: trackfuse ({s.engine} core). Playback {playback:.1f}x.",
            fontsize=9, color=MUTED,
        )  # fmt: skip

        # --- the view that follows the car ---------------------------------------------------------
        ax.set_aspect("equal")
        ax.set_facecolor(PAPER)
        ax.grid(True, color=GRID, lw=0.8)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        ax.plot(s.truth[:, 0], s.truth[:, 1], color=ROAD, lw=7, solid_capstyle="round", zorder=1)
        (self.path_true,) = ax.plot([], [], color=INK, lw=1.6, zorder=3)
        (self.path_dr,) = ax.plot([], [], color=C_DR, lw=1.6, ls=(0, (4, 3)), zorder=4)
        (self.path_ekf,) = ax.plot([], [], color=C_EKF, lw=2.0, zorder=5)
        self.gps_links = LineCollection([], linewidths=0.9, zorder=2)
        self.lm_links = LineCollection([], colors=C_LM, linewidths=0.8, linestyles=":", zorder=2)
        ax.add_collection(self.gps_links)
        ax.add_collection(self.lm_links)
        self.gps_dots = ax.scatter([], [], s=70, zorder=6, edgecolor="white", linewidth=0.8)
        ax.scatter(sim.LANDMARKS[:, 0], sim.LANDMARKS[:, 1], marker="^", s=170, color=C_LM, zorder=6, edgecolor="white")
        self.ellipse = Ellipse((0, 0), 1, 1, facecolor=C_EKF, alpha=0.18, edgecolor=C_EKF, lw=1.2, zorder=7)
        ax.add_patch(self.ellipse)
        (self.dot_dr,) = ax.plot([], [], "o", ms=11, mfc="white", mec=C_DR, mew=2.2, zorder=8)
        (self.dot_ekf,) = ax.plot([], [], "o", ms=11, mfc=C_EKF, mec="white", mew=1.5, zorder=9)
        (self.dot_true,) = ax.plot([], [], "o", ms=4.5, color=INK, zorder=10)
        self.clock = ax.set_title("", loc="left", fontsize=12, color=INK, pad=8)
        self.no_gps = ax.text(
            0.985, 0.965, "NO GPS", transform=ax.transAxes, ha="right", va="top", fontsize=13, fontweight="bold",
            color="white", bbox={"boxstyle": "round,pad=0.45", "facecolor": C_BAD, "edgecolor": "none"},
            visible=False, zorder=20,
        )  # fmt: skip
        self.bad_arrow = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), zorder=19, visible=False,
            arrowprops={"arrowstyle": "-|>", "color": C_BAD, "lw": 2.2, "mutation_scale": 18},
        )  # fmt: skip
        self.bad_note = ax.text(
            0, 0, "", fontsize=11, fontweight="bold", color=C_BAD, ha="center", va="center", visible=False, zorder=19,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": C_BAD},
        )  # fmt: skip
        ax.legend(
            handles=[
                Line2D([], [], color=INK, lw=1.6, label="true path"),
                Line2D([], [], color=C_DR, lw=1.6, ls=(0, (4, 3)), label="motion sensors only (drifts)"),
                Line2D([], [], color=C_EKF, lw=2.0, label="fused estimate"),
                Line2D([], [], color=C_GPS, marker="o", ms=8, ls="", label="GPS fixes (1 per second)"),
                Line2D([], [], color=C_LM, marker="^", ms=9, ls="", label="landmarks"),
            ],
            loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.95,
        ).set_zorder(30)  # fmt: skip

        # --- the whole drive ---------------------------------------------------------------------------
        ax_map.plot(s.truth[:, 0], s.truth[:, 1], color=ROAD, lw=3)
        (self.map_path,) = ax_map.plot([], [], color=INK, lw=1.4)
        ax_map.scatter(sim.LANDMARKS[:, 0], sim.LANDMARKS[:, 1], marker="^", s=45, color=C_LM)
        self.map_view = Rectangle((0, 0), 2 * HALF_W, 2 * HALF_H, fill=False, edgecolor=C_EKF, lw=1.4)
        ax_map.add_patch(self.map_view)
        (self.map_car,) = ax_map.plot([], [], "o", ms=6, color=C_EKF, zorder=5)
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_facecolor(PAPER)
        length = float(np.sum(np.hypot(*np.diff(s.truth[:, :2], axis=0).T)))
        ax_map.set_title(f"The whole drive ({length:.0f} m)", loc="left", fontsize=12, color=INK, pad=8)

        # --- position error over time ------------------------------------------------------------------
        duration = s.scenario.duration
        if s.scenario.gps_outage:
            ax_err.axvspan(*s.scenario.gps_outage, color=C_BAD, alpha=0.08, lw=0)
            ax_err.text(sum(s.scenario.gps_outage) / 2, 13, "no GPS", color=C_BAD, fontsize=9, ha="center", va="center")
        self.err_gps = ax_err.scatter([], [], s=16, color=C_GPS, alpha=0.8, zorder=3)
        (self.err_bad,) = ax_err.plot([], [], "x", ms=8, mew=2.2, color=C_BAD, zorder=4, clip_on=False)
        self.err_bad_note = ax_err.text(0, 0, "", color=C_BAD, fontsize=9, fontweight="bold", ha="right", va="center")
        (self.err_dr,) = ax_err.plot([], [], color=C_DR, lw=1.6, ls=(0, (4, 3)))
        (self.err_ekf,) = ax_err.plot([], [], color=C_EKF, lw=1.5)
        self.cursor = ax_err.axvline(0, color=MUTED, lw=0.8)
        self.tags = [
            ax_err.text(0, 1, "", color=color, fontsize=10, fontweight="bold", va="center")
            for color in (C_GPS, C_DR, C_EKF)
        ]
        ax_err.set_yscale("log")
        ax_err.set_ylim(0.01, 20)
        ax_err.set_xlim(0, duration * 1.38)  # room on the right for the labels that ride on the cursor
        ax_err.set_xticks(np.arange(0, duration + 1, 20))
        ax_err.set_yticks([0.01, 0.1, 1, 10])
        ax_err.set_yticklabels(["1 cm", "10 cm", "1 m", "10 m"])
        ax_err.grid(True, color=GRID, lw=0.8)
        ax_err.set_facecolor(PAPER)
        ax_err.set_title("Position error over time (seconds)", loc="left", fontsize=12, color=INK, pad=8)

    def draw(self, now: float) -> None:
        s = self.s
        i = max(int(np.searchsorted(s.t, now, side="right")) - 1, 0)
        j = max(int(np.searchsorted(s.t_dr, now, side="right")) - 1, 0)
        car, fused, drift = s.truth[i, :2], s.est[i, :2], s.dr[j, :2]
        self.caption.set_text(caption(s, now))

        # the view that follows the car
        ax = self.ax
        ax.set_xlim(car[0] - HALF_W, car[0] + HALF_W)
        ax.set_ylim(car[1] - HALF_H, car[1] + HALF_H)
        self.clock.set_text(f"Following the car   t = {now:4.1f} s   grid = 5 m")
        self.path_true.set_data(s.truth[: i + 1, 0], s.truth[: i + 1, 1])
        self.path_dr.set_data(s.dr[: j + 1, 0], s.dr[: j + 1, 1])
        self.path_ekf.set_data(s.est[: i + 1, 0], s.est[: i + 1, 1])
        self.dot_true.set_data([car[0]], [car[1]])
        self.dot_ekf.set_data([fused[0]], [fused[1]])
        self.dot_dr.set_data([drift[0]], [drift[1]])

        shown = (s.gps_t <= now) & ~s.gps_bad
        fade = np.clip(1.0 - (now - s.gps_t[shown]) / 12.0, 0.38, 1.0)
        rgb = np.array([int(C_GPS[k : k + 2], 16) / 255 for k in (1, 3, 5)])
        colors = np.column_stack([np.tile(rgb, (len(fade), 1)), fade]) if len(fade) else np.empty((0, 4))
        self.gps_dots.set_offsets(s.gps_xy[shown] if shown.any() else np.empty((0, 2)))
        self.gps_dots.set_facecolors(colors)
        self.gps_links.set_segments(list(zip(s.gps_truth[shown], s.gps_xy[shown], strict=True)))
        self.gps_links.set_color(colors * [1, 1, 1, 0.7] if len(fade) else colors)

        reach = np.hypot(*(sim.LANDMARKS - car).T) <= s.scenario.landmark_range
        self.lm_links.set_segments([(car, lm) for lm in sim.LANDMARKS[reach]])

        values, vectors = np.linalg.eigh(s.cov[i][:2, :2])
        k95 = np.sqrt(5.991)  # 95 percent of a two-dimensional Gaussian
        self.ellipse.set_center(fused)
        self.ellipse.set_width(2 * k95 * np.sqrt(values[1]))
        self.ellipse.set_height(2 * k95 * np.sqrt(values[0]))
        self.ellipse.set_angle(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))

        outage = s.scenario.gps_outage
        self.no_gps.set_visible(bool(outage) and outage[0] <= now < outage[1])
        bad = np.flatnonzero(s.gps_bad & (s.gps_t <= now) & (s.gps_t > now - 7.0))
        self.bad_arrow.set_visible(len(bad) > 0)
        self.bad_note.set_visible(len(bad) > 0)
        if len(bad):
            b = bad[-1]
            toward = (s.gps_xy[b] - car) / np.hypot(*(s.gps_xy[b] - car))
            self.bad_arrow.set_position(car + 2.0 * toward)
            self.bad_arrow.xy = car + 9.0 * toward
            verdict = "refused" if s.gps_rejected[b] else "taken in"
            self.bad_note.set_text(f"GPS fix {s.gps_err[b]:.0f} m away: {verdict}")
            self.bad_note.set_position(car + 11.5 * toward)

        # the whole drive
        self.map_path.set_data(s.truth[: i + 1, 0], s.truth[: i + 1, 1])
        self.map_view.set_xy((car[0] - HALF_W, car[1] - HALF_H))
        self.map_car.set_data([car[0]], [car[1]])

        # position error over time
        self.err_gps.set_offsets(np.column_stack([s.gps_t[shown], s.gps_err[shown]]) if shown.any() else np.empty((0, 2)))
        past_bad = s.gps_bad & (s.gps_t <= now)
        top = self.ax_err.get_ylim()[1]
        self.err_bad.set_data(s.gps_t[past_bad], np.full(int(past_bad.sum()), top * 0.8))
        if past_bad.any():
            b = int(np.flatnonzero(past_bad)[-1])
            self.err_bad_note.set_text(f"{s.gps_err[b]:.0f} m")
            self.err_bad_note.set_position((s.gps_t[b] - 2.0, top * 0.8))
        else:
            self.err_bad_note.set_text("")
        self.err_dr.set_data(s.t_dr[: j + 1], np.maximum(s.err_dr[: j + 1], 1e-3))
        self.err_ekf.set_data(s.t[: i + 1], np.maximum(s.err[: i + 1], 1e-3))
        self.cursor.set_xdata([now, now])
        gps_now = rmse(s.gps_err[shown]) if shown.any() else float("nan")
        fused_now = rmse(s.err[: i + 1])
        levels = [gps_now, float(s.err_dr[j]), fused_now]
        heights = [gps_now, max(float(s.err_dr[j]), fused_now * 1.9), fused_now]  # keep two labels from overlapping
        for tag, name, level, height in zip(self.tags, ("GPS", "drift", "fused"), levels, heights, strict=True):
            visible = bool(np.isfinite(level))
            tag.set_visible(visible)
            if visible:
                tag.set_text(f"{name}  {level:.1f} m" if level >= 0.95 else f"{name}  {level * 100:.0f} cm")
                tag.set_position((now + 1.5, float(np.clip(height, 0.013, 15))))


def _writer(out: Path, fps: int):
    from matplotlib import animation

    if out.suffix.lower() == ".gif":
        return animation.PillowWriter(fps=fps)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg  # an optional package that carries its own ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit(
                "an .mp4 needs ffmpeg: install it (or `pip install imageio-ffmpeg`), or write a .gif"
            ) from None
    import matplotlib

    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    # yuv420p and an even size are what phones, browsers and LinkedIn all play.
    return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])


def render(
    out: str | Path,
    *,
    seconds: float = 22.0,
    fps: int = 30,
    width: int = 1920,
    hold: float = 2.0,
    at: float | None = None,
    seed: int = 1,
    max_frames: int | None = None,
) -> Path:
    """Write the clip (.gif or .mp4) or, with `at`, one still frame (.png). Returns the path written."""
    from matplotlib import animation

    out = Path(out)
    story = build_story(seed)
    duration = story.scenario.duration
    scene = _Scene(story, width, playback=duration / seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        scene.draw(duration if at is None else at)
        scene.fig.savefig(out, facecolor="white")
    else:
        moving = np.linspace(0.0, duration, round(seconds * fps) + 1)[1:]
        times = np.concatenate([moving, np.full(round(hold * fps), duration)])[:max_frames]
        clip = animation.FuncAnimation(scene.fig, lambda k: scene.draw(float(times[k])), frames=len(times), blit=False)
        clip.save(out, writer=_writer(out, fps))
    scene.plt.close(scene.fig)
    return out
