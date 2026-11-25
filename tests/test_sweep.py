"""The sweep's own claims, on a handful of short drives so the test stays quick."""

import numpy as np
import pytest

from trackfuse import sweep


def test_fusing_beats_raw_gps_and_landmarks_beat_none():
    noise = sweep.gps_noise_sweep(sigmas=(1.0, 8.0), runs=4, duration=30.0)
    assert all(values.shape == (2, 4) for values in noise.values())
    median = {name: np.median(values, axis=1) for name, values in noise.items()}
    assert np.all(median["all"] < median["gps_imu"]) and np.all(median["gps_imu"] < median["gps"])
    assert median["gps"][1] > 4 * median["gps"][0]  # raw GPS follows its own noise
    assert median["all"][1] < 1.5 * median["all"][0]  # with landmarks in view, worse GPS barely matters


def test_a_longer_outage_can_only_be_worse():
    outage = sweep.outage_sweep(lengths=(10.0, 30.0), runs=4)
    for values in outage.values():
        assert values.shape == (2, 4)
        assert np.all(values[1] >= values[0])  # drive by drive: the 30 s outage contains the 10 s one
    assert np.median(outage["landmarks"][1]) < np.median(outage["no_landmarks"][1])


def test_summary_and_chart(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    noise = sweep.gps_noise_sweep(sigmas=(1.0, 8.0), runs=3, duration=20.0)
    outage = sweep.outage_sweep(lengths=(0.0, 10.0), runs=3)
    middle, low, high = sweep.summary(noise["gps"])
    assert np.all(low <= middle) and np.all(middle <= high)
    assert "GPS noise" in sweep.table(noise, outage, sigmas=(1.0, 8.0), lengths=(0.0, 10.0))
    chart = sweep.plot(noise, outage, tmp_path / "chart.png", sigmas=(1.0, 8.0), lengths=(0.0, 10.0))
    with Image.open(chart) as image:
        assert image.size == (1920, 840)
