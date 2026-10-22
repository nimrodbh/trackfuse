from __future__ import annotations

import numpy as np

from .sim import GPS, PREDICT, Run


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pos_rmse(est: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = est[:, :2] - truth[:, :2]
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def gps_only_rmse(run: Run) -> float:
    """Error of the raw GPS fixes themselves (what you get with no filter)."""
    m = run.types == GPS
    err = run.vals[m][:, :2] - run.truth[m][:, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def predict_only(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Dead reckoning: the same events with every measurement update removed."""
    m = run.types == PREDICT
    return run.types[m], run.vals[m]


def nees(est: np.ndarray, cov: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Normalized estimation error squared per event. A consistent 4-state filter averages about 4."""
    e = est - truth
    e[:, 2] = _wrap(e[:, 2])
    sol = np.linalg.solve(cov, e[:, :, None])[:, :, 0]
    return np.einsum("ij,ij->i", e, sol)
