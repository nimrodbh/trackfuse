"""The harder simulator does what it says, and costs the filter what the README says it costs."""

import numpy as np
import pytest

from trackfuse import sim, stress


def _pair(**effects):
    """The same random drive twice: as the filter assumes, and with the given effects switched on."""
    return sim.generate(sim.Scenario(duration=40.0), 5), sim.generate(sim.Scenario(duration=40.0, **effects), 5)


def test_every_effect_is_off_by_default():
    scn = sim.Scenario()
    assert (scn.gyro_bias, scn.accel_bias, scn.slip, scn.gps_correlation) == (0.0, 0.0, 0.0, 0.0)


def test_a_bias_shifts_the_measured_controls_by_exactly_the_bias():
    plain, biased = _pair(gyro_bias=0.01, accel_bias=0.1)
    moved = plain.types == sim.PREDICT
    assert np.allclose(biased.vals[moved][:, 1] - plain.vals[moved][:, 1], 0.1)  # acceleration
    assert np.allclose(biased.vals[moved][:, 2] - plain.vals[moved][:, 2], 0.01)  # yaw rate
    assert np.allclose(biased.truth, plain.truth)  # the car itself drives the same road


def test_slip_moves_the_car_off_its_heading_but_leaves_the_heading_alone():
    plain, slipping = _pair(slip=0.05)
    # Compare at the motion-sensor events: there is one per time step in both drives, while the number of
    # landmark sightings differs once the two cars are in different places.
    a, b = plain.truth[plain.types == sim.PREDICT], slipping.truth[slipping.types == sim.PREDICT]
    assert np.allclose(b[:, 2], a[:, 2])
    assert np.max(np.hypot(*(b[:, :2] - a[:, :2]).T)) > 0.5


def _gps_errors(run):
    gps = run.types == sim.GPS
    return run.vals[gps][:, 0] - run.truth[gps][:, 0]


def test_lingering_gps_errors_carry_over_from_one_fix_to_the_next():
    def carry_over(errors):
        return float(np.corrcoef(errors[:-1], errors[1:])[0, 1])

    plain = sim.generate(sim.Scenario(duration=300.0), 5)
    lingering = sim.generate(sim.Scenario(duration=300.0, gps_correlation=30.0), 5)
    assert abs(carry_over(_gps_errors(plain))) < 0.3
    assert carry_over(_gps_errors(lingering)) > 0.8
    assert np.std(_gps_errors(lingering)) < 2 * plain.scenario.sigma_gps  # they linger, they do not grow


def test_a_bias_breaks_the_filter_as_built_and_loosening_helps():
    cases = {"ideal": {}, "gyro bias": stress.CASES["gyro bias"]}
    median = {case: {name: float(np.median(v)) for name, v in row.items()}
              for case, row in stress.measure(runs=4, duration=60.0, cases=cases).items()}
    ideal, biased = median["ideal"], median["gyro bias"]
    assert ideal["gps_imu"] < ideal["gps"] and ideal["gps_imu_nees"] < 10  # in its own world: better than GPS, honest
    assert biased["gps_imu"] > 3 * ideal["gps_imu"] and biased["gps_imu_nees"] > 20  # far off, and far too sure
    assert biased["gps_imu_loose"] < biased["gps_imu"]  # the cheap remedy helps
    assert ideal["gps_imu_loose"] > ideal["gps_imu"]  # and has a price when nothing is wrong


def test_table_and_chart(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    results = stress.measure(runs=2, duration=20.0, cases={"ideal": {}, "all four": stress.CASES["all four"]})
    assert "all four" in stress.table(results)
    with Image.open(stress.plot(results, tmp_path / "chart.png")) as image:
        assert image.size == (1920, 840)
