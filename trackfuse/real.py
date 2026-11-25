"""Real drives: the smartLoc logs through the same filter.

    python -m trackfuse real data/smartloc/berlin1_potsdamer_platz [more folders]

smartLoc (TU Chemnitz) recorded four drives through Berlin and Frankfurt with a mass-market GPS
receiver, the car's own yaw-rate and wheel-speed sensors, and a survey-grade reference for the
truth. The data is not part of this repository (it is published for academic use under
CC BY-NC-SA 4.0): download a scenario from
https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en and unpack it into its own folder.

What changes against the simulator:
  - positions are longitude and latitude, turned into meters east and north of the first point;
  - there is no accelerometer, so the acceleration the filter expects is the change of the wheel
    speed from one moment to the next (which means the filter's speed follows the wheel sensor);
  - there are no landmarks: this is the GPS and motion sensors case only;
  - the filter starts from the true position and heading, as it does in the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import native, pyekf, sim

EARTH_RADIUS = 6378137.0  # meters, WGS84 equator
P0 = np.array([1.0, 1.0, 0.01, 0.25])

# How much the filter is told to trust each sensor on a real car. The last two were chosen on the two Berlin
# drives (the best of a small grid); the two Frankfurt drives were not used for choosing anything.
SIGMA_A = 0.05  # m/s^2: the wheel speed does not drift, so its change is trusted
SIGMA_W = 0.02  # rad/s: wide, because the filter does not estimate the yaw-rate sensor's bias
GPS_SIGMA = 5.0  # meters


@dataclass
class Drive:
    name: str
    t: np.ndarray  # (n,) time of each GPS fix, seconds
    truth: np.ndarray  # (n, 2) the reference position at each fix, meters east and north of the first one
    heading: np.ndarray  # (n,) the reference heading, radians counterclockwise from east
    gps: np.ndarray  # (n, 2) the mass-market receiver's fix, same frame
    claimed: np.ndarray  # (n,) the accuracy the receiver claims for each fix, meters
    can_t: np.ndarray  # (m,) time of each sample from the car's own sensors
    yaw_rate: np.ndarray  # (m,) rad/s
    speed: np.ndarray  # (m,) m/s, from the wheels


def read_table(path: Path, columns: int) -> np.ndarray:
    """A semicolon-separated smartLoc file as numbers. Missing fields become NaN; the header line is skipped."""
    rows = []
    with Path(path).open("r", encoding="latin-1") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(";")
            parts += [""] * (columns - len(parts))
            rows.append([float(p) if p.strip() else np.nan for p in parts[:columns]])
    return np.array(rows, dtype=float).reshape(-1, columns)


def to_meters(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Longitude and latitude in degrees as meters east and north of (lon0, lat0). Flat-earth, fine for a few km."""
    east = np.radians(lon - lon0) * EARTH_RADIUS * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * EARTH_RADIUS
    return np.column_stack([east, north])


def load(folder: str | Path) -> Drive:
    folder = Path(folder)
    nav = read_table(folder / "NAV-POSLLH.csv", 23)  # reference and receiver fix, 5 per second
    yaw = read_table(folder / "YAWRATE.csv", 3)  # the car's yaw-rate sensor, 50 per second
    vel = read_table(folder / "VELOCITY.csv", 3)  # the car's wheel speed, 50 per second
    lon0, lat0 = nav[0, 2], nav[0, 4]
    heading = nav[:, 8]
    wheel_speed = np.interp(nav[:, 1], vel[:, 1], vel[:, 2])
    if np.nanmedian(nav[wheel_speed > 2.0, 12]) < 0:
        # In one scenario (Frankfurt, Westend Tower) the reference reports the speed as negative and the
        # heading half a circle away from the direction of travel. Turn it around.
        heading = heading + np.pi
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    return Drive(
        name=folder.name,
        t=nav[:, 1],
        truth=to_meters(nav[:, 2], nav[:, 4], lon0, lat0),
        heading=heading,
        gps=to_meters(nav[:, 17], nav[:, 18], lon0, lat0),
        claimed=nav[:, 21],
        can_t=yaw[:, 1],
        yaw_rate=yaw[:, 2],
        speed=np.interp(yaw[:, 1], vel[:, 1], vel[:, 2]),
    )


STOPPED = 0.05  # m/s: below this wheel speed the car is standing
SETTLE = 3.0  # seconds: the body still rocks right after a stop, so calibration waits this long


def calibrate_at_standstill(can_t: np.ndarray, yaw_rate: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """The yaw-rate readings with the sensor's bias taken out, using only what the car knows at that moment.

    A car whose wheels do not turn is not turning either, so whatever the yaw-rate sensor reads at a
    standstill is its bias. The bias in use at any moment is the average of everything the sensor has read
    so far in the drive while the car had been standing for more than `SETTLE` seconds (zero until the
    first such stop: a short stop tells nothing, the measured readings there scatter several times wider
    than the bias itself). While the car stands, the corrected yaw rate is exactly zero.
    """
    out = np.empty_like(yaw_rate)
    bias, total, count, stopped_at = 0.0, 0.0, 0, None
    for i in range(len(yaw_rate)):
        if speed[i] < STOPPED:
            stopped_at = can_t[i] if stopped_at is None else stopped_at
            if can_t[i] - stopped_at >= SETTLE:
                total, count = total + yaw_rate[i], count + 1
                bias = total / count
            out[i] = 0.0
        else:
            stopped_at = None
            out[i] = yaw_rate[i] - bias
    return out


def to_events(drive: Drive, gps_sigma: float = GPS_SIGMA, use_gps: bool = True, standstill: bool = True):
    """The drive as the filter's event stream, starting at the first fix. Returns (types, vals, times, x0).

    Time moves from one instant to the next over the car's sensor samples and the GPS fixes together. Each
    step is one motion event: its acceleration is the change in wheel speed over the step and its yaw rate
    is the sensor's reading at the middle of the step. A GPS event follows the step that ends at its fix.

    With `standstill`, the car uses what it knows when its wheels do not turn: the yaw-rate sensor is
    calibrated there (see `calibrate_at_standstill`), and GPS fixes are ignored, since a standing car has
    not moved whatever the receiver says.
    """
    d = drive
    t0 = d.t[0]
    fix_t = d.t[1:]
    fixes = d.gps[1:]
    usable = np.all(np.isfinite(fixes), axis=1) if use_gps else np.zeros(len(fix_t), dtype=bool)
    yaw_rate = calibrate_at_standstill(d.can_t, d.yaw_rate, d.speed) if standstill else d.yaw_rate
    if standstill:
        usable &= np.interp(fix_t, d.can_t, d.speed) >= STOPPED
    instants = np.unique(np.concatenate([d.can_t[(d.can_t > t0) & (d.can_t <= d.t[-1])], fix_t]))
    before = np.concatenate([[t0], instants[:-1]])
    dt = instants - before
    speed = np.interp(np.concatenate([[t0], instants]), d.can_t, d.speed)
    motion = np.zeros((len(instants), 6))
    motion[:, 0] = dt
    motion[:, 1] = np.diff(speed) / dt
    motion[:, 2] = np.interp((before + instants) / 2, d.can_t, yaw_rate)

    gps = np.zeros((int(usable.sum()), 6))
    gps[:, :2] = fixes[usable]
    gps[:, 2] = gps_sigma
    times = np.concatenate([instants, fix_t[usable]])
    types = np.concatenate([np.full(len(instants), sim.PREDICT), np.full(len(gps), sim.GPS)]).astype(np.int32)
    order = np.lexsort((types, times))  # by time; at the same instant the motion step comes before the fix
    x0 = np.array([d.truth[0, 0], d.truth[0, 1], d.heading[0], speed[0]])
    return types[order], np.concatenate([motion, gps])[order], times[order], x0


def position_errors(
    drive: Drive,
    sigma_a: float = SIGMA_A,
    sigma_w: float = SIGMA_W,
    gps_sigma: float = GPS_SIGMA,
    gate_chi2: float = 0.0,
    use_gps: bool = True,
    standstill: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the filter over the drive. Returns (error at every fix time in meters, the estimates there)."""
    types, vals, times, x0 = to_events(drive, gps_sigma, use_gps, standstill)
    engine = native if native.available() else pyekf
    est, _, _ = engine.run(types, vals, x0, P0, sigma_a, sigma_w, gate_chi2, False)
    at_fix = np.searchsorted(times, drive.t[1:], side="right") - 1  # the last event at or before each fix
    return np.hypot(*(est[at_fix, :2] - drive.truth[1:]).T), est[at_fix]


def gps_errors(drive: Drive) -> np.ndarray:
    """The error of the receiver's own fixes, with no filter."""
    return np.hypot(*(drive.gps[1:] - drive.truth[1:]).T)


def gps_persistence(drive: Drive, seconds: float = 5.0) -> float:
    """How much of a GPS error is still there `seconds` later: the correlation of the east error with itself.

    Near 0 means every fix errs on its own, which is what the filter assumes. Near 1 means the error is
    the same one, carried along.
    """
    east = (drive.gps - drive.truth)[:, 0]
    east = east[np.isfinite(east)]
    lag = round(seconds / float(np.median(np.diff(drive.t))))
    return float(np.corrcoef(east[:-lag], east[lag:])[0, 1])


def rmse(errors: np.ndarray) -> float:
    errors = errors[np.isfinite(errors)]
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("nan")


def measure(folder: str | Path) -> dict:
    """The numbers for one drive. Position errors are RMSE in meters."""
    drive = load(folder)
    standing = np.interp(drive.t, drive.can_t, drive.speed) < STOPPED
    return {
        "name": drive.name,
        "length": float(np.sum(np.hypot(*np.diff(drive.truth, axis=0).T))),
        "minutes": float(drive.t[-1] - drive.t[0]) / 60,
        "standing": float(np.mean(standing)),  # share of the time
        "gps": rmse(gps_errors(drive)),
        "gps_claimed": float(np.nanmedian(drive.claimed)),  # the accuracy the receiver claims, median
        "gps_persistence": gps_persistence(drive),  # of an error, 5 seconds later
        "motion": rmse(position_errors(drive, use_gps=False, standstill=False)[0]),
        "motion_calibrated": rmse(position_errors(drive, use_gps=False)[0]),
        "fused": rmse(position_errors(drive)[0]),
    }


def report(folders: list[str | Path]) -> str:
    head = (
        f"{'drive':26s} {'length':>7s} {'standing':>8s} | {'GPS alone':>9s} {'(claims)':>8s} {'lingers':>7s} |"
        f" {'motion only':>11s} {'calibrated':>10s} | {'fused':>6s}"
    )
    rows = [head]
    for folder in folders:
        m = measure(folder)
        rows.append(
            f"{m['name']:26s} {m['length']:5.0f} m {m['standing'] * 100:7.0f}% | {m['gps']:7.2f} m {m['gps_claimed']:6.2f} m"
            f" {m['gps_persistence']:7.2f} | {m['motion']:9.1f} m {m['motion_calibrated']:8.1f} m | {m['fused']:4.2f} m"
        )
    return "\n".join(rows)
