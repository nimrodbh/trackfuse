"""Synthetic driving scenario: a vehicle on a figure-eight, with noisy IMU, GPS and landmark sensors.

The truth is generated from the same unicycle model the filter uses, driven by the TRUE
controls. The filter only sees MEASURED controls (truth plus Gaussian noise), so the filter's
process noise is exactly the IMU noise and a correct filter is statistically consistent.

That is the fair test of the math, and a kind one: real sensors also lie in ways the filter does
not model. Four optional effects make the drive harder (all off by default, so every number
measured without them stays exactly as it was): a constant gyro bias, a constant accelerometer
bias, side slip in turns, and GPS errors that linger from one fix to the next. `stress.py`
measures what each one costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PREDICT, GPS, LANDMARK = 0, 1, 2

LANDMARKS = np.array(
    [[60.0, 40.0], [-60.0, 40.0], [60.0, -40.0], [-60.0, -40.0], [0.0, 90.0], [0.0, -90.0]]
)


@dataclass
class Scenario:
    duration: float = 80.0
    dt: float = 0.02  # 50 Hz IMU
    gps_every: int = 50  # 1 Hz GPS
    landmark_every: int = 10  # 5 Hz landmark observations
    landmark_range: float = 90.0
    sigma_a: float = 0.2  # accel noise, m/s^2
    sigma_w: float = 0.01  # yaw-rate noise, rad/s
    sigma_gps: float = 3.0  # m
    sigma_range: float = 0.5  # m
    sigma_bearing: float = 0.01  # rad
    speed: float = 10.0
    gps_outage: tuple[float, float] | None = None  # (start, end) seconds with no GPS
    outliers: list[float] = field(default_factory=list)  # times at which GPS is wildly wrong
    # What the filter does not model. All zero: the sensors err exactly as the filter assumes.
    gyro_bias: float = 0.0  # rad/s, a constant offset in the measured yaw rate
    accel_bias: float = 0.0  # m/s^2, a constant offset in the measured acceleration
    slip: float = 0.0  # seconds: in a turn the car moves at an angle of slip * yaw_rate to its heading
    gps_correlation: float = 0.0  # seconds: GPS errors fade over this long (0 = each fix errs on its own)


@dataclass
class Run:
    types: np.ndarray  # (n,) int32
    vals: np.ndarray  # (n, 6) float64
    truth: np.ndarray  # (n, 4) truth state after each event: x, y, heading, speed
    times: np.ndarray  # (n,) seconds
    x0: np.ndarray  # (4,) initial truth
    scenario: Scenario


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def generate(scn: Scenario, seed: int) -> Run:
    rng = np.random.default_rng(seed)
    steps = round(scn.duration / scn.dt)
    x = np.array([0.0, 0.0, 0.0, scn.speed])
    x0 = x.copy()
    types: list[int] = []
    vals: list[list[float]] = []
    truth: list[np.ndarray] = []
    times: list[float] = []

    def emit(kind: int, v: list[float], t: float) -> None:
        types.append(kind)
        vals.append(v + [0.0] * (6 - len(v)))
        truth.append(x.copy())
        times.append(t)

    outlier_steps = {round(t / scn.dt) for t in scn.outliers}
    gps_error = np.zeros(2)  # used only when GPS errors linger
    gps_error_at: float | None = None
    for k in range(1, steps + 1):
        t = k * scn.dt
        # True controls: gentle speed oscillation and a figure-eight steering pattern.
        a_true = 0.3 * np.sin(0.5 * t)
        w_true = 0.35 * np.sin(2 * np.pi * t / 40.0)
        course = x[2] + scn.slip * w_true  # where the car is really going; equal to its heading without slip
        x = np.array(
            [
                x[0] + x[3] * np.cos(course) * scn.dt,
                x[1] + x[3] * np.sin(course) * scn.dt,
                _wrap(x[2] + w_true * scn.dt),
                x[3] + a_true * scn.dt,
            ]
        )
        a_meas = a_true + scn.accel_bias + rng.normal(0, scn.sigma_a)
        w_meas = w_true + scn.gyro_bias + rng.normal(0, scn.sigma_w)
        emit(PREDICT, [scn.dt, a_meas, w_meas], t)

        if k % scn.gps_every == 0:
            in_outage = scn.gps_outage is not None and scn.gps_outage[0] <= t < scn.gps_outage[1]
            if not in_outage:
                fresh = rng.normal(0, scn.sigma_gps, 2)
                if scn.gps_correlation > 0 and gps_error_at is not None:
                    # First-order Gauss-Markov: part of the last error carries over, the size stays sigma_gps.
                    keep = float(np.exp(-(t - gps_error_at) / scn.gps_correlation))
                    fresh = keep * gps_error + np.sqrt(1 - keep**2) * fresh
                gps_error, gps_error_at = fresh, t
                z = x[:2] + fresh
                if k in outlier_steps:
                    z = z + np.array([150.0, -150.0])
                emit(GPS, [z[0], z[1], scn.sigma_gps], t)

        if k % scn.landmark_every == 0:
            for lm in LANDMARKS:
                dx, dy = lm[0] - x[0], lm[1] - x[1]
                rho = float(np.hypot(dx, dy))
                if rho <= scn.landmark_range:
                    rng_meas = rho + rng.normal(0, scn.sigma_range)
                    brg_meas = _wrap(np.arctan2(dy, dx) - x[2] + rng.normal(0, scn.sigma_bearing))
                    emit(LANDMARK, [lm[0], lm[1], rng_meas, brg_meas, scn.sigma_range, scn.sigma_bearing], t)

    return Run(
        types=np.array(types, dtype=np.int32),
        vals=np.array(vals, dtype=np.float64),
        truth=np.array(truth),
        times=np.array(times),
        x0=x0,
        scenario=scn,
    )
