import numpy as np
import pytest

from trackfuse import metrics, native, pyekf, sim

P0 = np.array([1.0, 1.0, 0.01, 0.25])

needs_native = pytest.mark.skipif(not native.available(), reason="C++ core not built")


def _run_both(run, gate=0.0):
    s = run.scenario
    args = (run.types, run.vals, run.x0, P0, s.sigma_a, s.sigma_w, gate)
    return pyekf.run(*args), native.run(*args)


@needs_native
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cpp_matches_numpy_reference(seed):
    run = sim.generate(sim.Scenario(duration=30.0), seed)
    (xp, pp, rp), (xc, pc, rc) = _run_both(run)
    assert np.abs(xp - xc).max() < 1e-9
    assert np.abs(pp - pc).max() < 1e-9
    assert rp == rc == 0


@needs_native
def test_cpp_matches_reference_with_gate_and_outliers():
    run = sim.generate(sim.Scenario(duration=40.0, outliers=[10.0, 25.0]), 4)
    (xp, _, rp), (xc, _, rc) = _run_both(run, gate=9.21)
    assert rp == rc >= 2
    assert np.abs(xp - xc).max() < 1e-9


def test_fusion_beats_dead_reckoning_and_raw_gps():
    scn = sim.Scenario()
    run = sim.generate(scn, 7)
    est, _, _ = pyekf.run(run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w)
    fused = metrics.pos_rmse(est, run.truth)

    dt, dv = metrics.predict_only(run)
    dr, _, _ = pyekf.run(dt, dv, run.x0, P0, scn.sigma_a, scn.sigma_w)
    dead = metrics.pos_rmse(dr, run.truth[run.types == sim.PREDICT])

    assert fused < metrics.gps_only_rmse(run) / 5
    assert fused < dead


def test_covariance_stays_symmetric_positive_definite():
    scn = sim.Scenario(duration=40.0)
    run = sim.generate(scn, 5)
    _, cov, _ = pyekf.run(run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w)
    assert np.allclose(cov, np.swapaxes(cov, 1, 2), atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > -1e-12


def test_filter_is_statistically_consistent():
    """Monte Carlo NEES: with the true noise model, the mean NEES over runs should sit near the
    state dimension (4). A filter that is overconfident scores far above 4; too cautious, far below."""
    scn = sim.Scenario(duration=40.0)
    per_run = []
    for seed in range(40):
        run = sim.generate(scn, 100 + seed)
        # Draw the initial estimate error from the filter's own P0 so the start is consistent too.
        rng = np.random.default_rng(900 + seed)
        x_init = run.x0 + rng.normal(0, np.sqrt(P0))
        est, cov, _ = pyekf.run(run.types, run.vals, x_init, P0, scn.sigma_a, scn.sigma_w)
        n = metrics.nees(est, cov, run.truth)
        per_run.append(n[len(n) // 4 :].mean())  # skip the transient
    mean = float(np.mean(per_run))
    assert 3.0 < mean < 5.5, f"mean NEES {mean:.2f}, expected about 4"


def test_gps_outage_degrades_gracefully():
    """With GPS gone for 20 s the landmarks still bound the error; without them it drifts."""
    scn = sim.Scenario(duration=60.0, gps_outage=(20.0, 40.0))
    run = sim.generate(scn, 11)
    est, _, _ = pyekf.run(run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w)
    outage = (run.times >= 20.0) & (run.times < 40.0)
    with_lm = metrics.pos_rmse(est, run.truth, outage)

    keep = run.types != sim.LANDMARK
    est2, _, _ = pyekf.run(run.types[keep], run.vals[keep], run.x0, P0, scn.sigma_a, scn.sigma_w)
    without_lm = metrics.pos_rmse(est2, run.truth[keep], outage[keep])
    assert with_lm < 2.0
    assert without_lm > 3 * with_lm


def test_gating_rejects_gps_outliers():
    scn = sim.Scenario(duration=50.0, outliers=[15.0, 30.0, 45.0])
    run = sim.generate(scn, 13)
    args = (run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w)
    # Compare on GPS-only so the effect of an outlier is not masked by landmarks.
    keep = run.types != sim.LANDMARK
    ungated, _, r0 = pyekf.run(*(a[keep] if i < 2 else a for i, a in enumerate(args)), 0.0)
    gated, _, r1 = pyekf.run(*(a[keep] if i < 2 else a for i, a in enumerate(args)), 13.82)
    truth = run.truth[keep]
    assert r0 == 0 and r1 >= 3
    assert metrics.pos_rmse(gated, truth) < metrics.pos_rmse(ungated, truth)


def test_heading_stays_wrapped():
    scn = sim.Scenario(duration=80.0)
    run = sim.generate(scn, 3)
    est, _, _ = pyekf.run(run.types, run.vals, run.x0, P0, scn.sigma_a, scn.sigma_w)
    assert np.all(np.abs(est[:, 2]) <= np.pi + 1e-9)


def test_unknown_event_type_is_an_error():
    with pytest.raises(ValueError):
        pyekf.run(np.array([9], dtype=np.int32), np.zeros((1, 6)), np.zeros(4), P0, 0.2, 0.01)
