"""Reading recorded drives. The logs themselves are not in the repository, so these tests write a small
made-up drive in the same file format; one last test runs on the real files when they are present."""

from pathlib import Path

import numpy as np
import pytest

from trackfuse import real, sim

LON0, LAT0 = 13.4, 52.5
SPEED = 10.0  # m/s while driving
BIAS = 0.003  # rad/s, what the made-up yaw-rate sensor reads when the car is not turning


def _lon(east: np.ndarray) -> np.ndarray:
    return LON0 + np.degrees(east / (real.EARTH_RADIUS * np.cos(np.radians(LAT0))))


def write_drive(folder: Path, *, stop: float = 10.0, backward_reference: bool = False, gps_north: float = 0.0) -> Path:
    """A car driving east for 20 s, standing for `stop` seconds, then driving east for 20 s more."""
    folder.mkdir(parents=True)
    can_t = 1000.0 + np.arange(0.0, 40.0 + stop + 1e-9, 0.02)
    moving = (can_t - 1000.0 < 20.0) | (can_t - 1000.0 >= 20.0 + stop)
    speed = np.where(moving, SPEED, 0.0)
    east = np.concatenate([[0.0], np.cumsum(speed[1:] * 0.02)])

    fix_t = can_t[::10]  # 5 fixes per second
    fix_east = east[::10]
    fix_speed = speed[::10]
    nav = np.zeros((len(fix_t), 23))
    nav[:, 0], nav[:, 1] = 1900, fix_t
    nav[:, 2], nav[:, 4] = _lon(fix_east), LAT0
    nav[:, 8] = np.pi if backward_reference else 0.0  # heading: east is 0
    nav[:, 12] = -fix_speed if backward_reference else fix_speed
    nav[:, 17] = _lon(fix_east)
    nav[:, 18] = LAT0 + np.degrees(gps_north / real.EARTH_RADIUS)
    nav[:, 21] = 1.0

    def save(name: str, header: str, table: np.ndarray) -> None:
        lines = [header] + [";".join(repr(float(v)) for v in row) for row in table]
        (folder / name).write_text("\n".join(lines) + "\n", encoding="latin-1")

    save("NAV-POSLLH.csv", ";".join(f"column {i}" for i in range(23)), nav)
    save("YAWRATE.csv", "week;seconds;yaw rate", np.column_stack([np.full(len(can_t), 1900), can_t, np.full(len(can_t), BIAS)]))
    save("VELOCITY.csv", "week;seconds;speed", np.column_stack([np.full(len(can_t), 1900), can_t, speed]))
    return folder


def test_degrees_become_meters():
    north = real.to_meters(np.array([LON0]), np.array([LAT0 + 0.001]), LON0, LAT0)
    assert north[0] == pytest.approx([0.0, 111.32], abs=0.01)
    east = real.to_meters(np.array([10.001]), np.array([60.0]), 10.0, 60.0)
    assert east[0, 0] == pytest.approx(111.32 / 2, abs=0.01)  # a degree of longitude is half as long at 60 north


def test_missing_fields_are_read_as_gaps(tmp_path):
    (tmp_path / "t.csv").write_text("a;b;c\n1;2;3\n4;;\n5;6\n", encoding="latin-1")
    table = real.read_table(tmp_path / "t.csv", 3)
    assert table[0].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(table[1, 1]) and np.isnan(table[1, 2]) and np.isnan(table[2, 2])


def test_a_drive_is_loaded_in_meters_from_its_first_point(tmp_path):
    drive = real.load(write_drive(tmp_path / "drive"))
    assert drive.truth[0].tolist() == [0.0, 0.0]
    assert drive.truth[-1] == pytest.approx([400.0, 0.0], abs=0.01)  # two legs of 20 s at 10 m/s
    assert np.allclose(drive.heading, 0.0)


def test_a_reference_that_points_backward_is_turned_around(tmp_path):
    drive = real.load(write_drive(tmp_path / "drive", backward_reference=True))
    assert np.allclose(np.cos(drive.heading), 1.0)  # east again


def test_the_motion_step_comes_before_the_fix_at_the_same_instant(tmp_path):
    drive = real.load(write_drive(tmp_path / "drive"))
    types, vals, times, x0 = real.to_events(drive, standstill=False)
    assert np.all(np.diff(times) >= 0)
    same = np.flatnonzero(np.diff(times) == 0)
    assert len(same) > 0 and np.all(types[same] == sim.PREDICT) and np.all(types[same + 1] == sim.GPS)
    motion = types == sim.PREDICT
    assert np.sum(vals[motion, 0]) == pytest.approx(drive.t[-1] - drive.t[0])  # the steps cover the whole drive
    assert np.sum(vals[motion, 1] * vals[motion, 0]) == pytest.approx(0.0, abs=1e-9)  # it ends at the speed it began
    assert x0.tolist() == [0.0, 0.0, 0.0, SPEED]


def test_a_long_stop_calibrates_the_yaw_rate_sensor_and_a_short_one_does_not(tmp_path):
    for stop, calibrated in ((10.0, True), (2.0, False)):
        drive = real.load(write_drive(tmp_path / f"stop{stop:.0f}", stop=stop))
        fixed = real.calibrate_at_standstill(drive.can_t, drive.yaw_rate, drive.speed)
        seconds = drive.can_t - drive.can_t[0]
        assert np.allclose(fixed[seconds < 19.9], BIAS)  # nothing is known before the first stop
        assert np.allclose(fixed[(seconds > 20.1) & (seconds < 19.9 + stop)], 0.0)  # a standing car does not turn
        assert np.allclose(fixed[seconds > 20.1 + stop], 0.0 if calibrated else BIAS)


def test_standing_still_fixes_are_ignored_and_the_bias_stops_bending_the_path(tmp_path):
    drive = real.load(write_drive(tmp_path / "drive", gps_north=5.0))  # every fix is 5 m off to the north
    types, _, times, _ = real.to_events(drive)
    during_stop = (times > drive.t[0] + 20.5) & (times < drive.t[0] + 29.5)
    assert not np.any(types[during_stop] == sim.GPS)

    plain, _ = real.position_errors(drive, use_gps=False, standstill=False)
    calibrated, _ = real.position_errors(drive, use_gps=False)
    second_leg = drive.t[1:] > drive.t[0] + 30.0

    def growth(errors):
        return errors[second_leg][-1] - errors[second_leg][0]

    assert growth(calibrated) < 0.5 * growth(plain)  # with the bias known, the second leg bends far less


SMARTLOC = Path(__file__).resolve().parent.parent / "data" / "smartloc" / "berlin1_potsdamer_platz"


@pytest.mark.skipif(not (SMARTLOC / "NAV-POSLLH.csv").exists(), reason="the smartLoc logs are not downloaded")
def test_the_first_berlin_drive_reads_as_measured():
    m = real.measure(SMARTLOC)
    assert m["length"] == pytest.approx(1553, abs=2)
    assert m["gps"] == pytest.approx(5.89, abs=0.02)  # the receiver's own fixes, no filter
    assert m["motion"] == pytest.approx(28.6, abs=0.3)  # the car's sensors alone, from a true start
    assert m["fused"] == pytest.approx(6.43, abs=0.05)
