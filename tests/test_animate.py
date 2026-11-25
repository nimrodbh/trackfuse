"""The clip is built from a real run, so what it claims is tested like any other result."""

import numpy as np
import pytest

from trackfuse import animate


@pytest.fixture(scope="module")
def story():
    return animate.build_story()


def test_the_run_contains_the_three_things_the_clip_shows(story):
    start, end = animate.OUTAGE
    assert not np.any((story.gps_t >= start) & (story.gps_t < end))  # no GPS fix arrives during the outage
    during = (story.t >= start) & (story.t < end)
    assert animate.rmse(story.err[during]) < 0.5  # the landmarks hold the estimate

    assert story.gps_bad.sum() == 1  # one wildly wrong fix
    bad = int(np.flatnonzero(story.gps_bad)[0])
    assert story.gps_t[bad] == pytest.approx(animate.BAD_FIX_AT)
    assert story.gps_err[bad] > 150 and story.gps_rejected[bad]  # and the filter refused it

    fused, drift, gps = animate.rmse(story.err), animate.rmse(story.err_dr), animate.rmse(story.gps_err[~story.gps_bad])
    assert fused < 0.25 < drift < gps  # the order the clip's last line states


def test_captions_follow_the_drive_and_quote_the_run(story):
    seen = story.gps_t <= 20.0
    early = animate.caption(story, 20.0)
    assert early.startswith("GPS alone") and f"{animate.rmse(story.gps_err[seen]):.1f} m off" in early
    assert f"{animate.rmse(story.err[story.t <= 20.0]):.2f} m." in early
    assert animate.caption(story, 35.0).startswith("GPS lost for 20 seconds")
    assert animate.caption(story, animate.BAD_FIX_AT + 2.0).endswith("The filter refuses it.")
    assert animate.caption(story, story.scenario.duration).startswith("The whole 80 second drive")


def test_renders_a_still_and_a_short_clip(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    still = animate.render(tmp_path / "frame.png", at=47.0, width=320)
    with Image.open(still) as frame:
        assert frame.size == (320, 180)
    clip = animate.render(tmp_path / "clip.gif", width=320, fps=5, max_frames=3)
    with Image.open(clip) as gif:
        assert gif.n_frames == 3 and gif.size == (320, 180)
