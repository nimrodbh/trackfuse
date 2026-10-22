"""Reference implementation in NumPy: the same filter as core/ekf.cpp, written for clarity.

It exists so the C++ can be trusted: tests require both to agree to rounding error.
"""

from __future__ import annotations

import numpy as np

from .sim import GPS, LANDMARK, PREDICT


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(
    types: np.ndarray,
    vals: np.ndarray,
    x0: np.ndarray,
    p0_diag: np.ndarray,
    sigma_a: float,
    sigma_w: float,
    gate_chi2: float = 0.0,
    want_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    n = len(types)
    x = np.array(x0, dtype=float)
    p = np.diag(np.asarray(p0_diag, dtype=float))
    out_x = np.empty((n, 4))
    out_p = np.empty((n, 4, 4)) if want_cov else None
    rejected = 0
    qu = np.diag([sigma_a**2, sigma_w**2])

    def update(h: np.ndarray, y: np.ndarray, r: np.ndarray) -> bool:
        nonlocal x, p, rejected
        s = h @ p @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            rejected += 1
            return False
        if gate_chi2 > 0.0 and float(y @ s_inv @ y) > gate_chi2:
            rejected += 1
            return False
        k = p @ h.T @ s_inv
        x = x + k @ y
        x[2] = wrap(x[2])
        i_kh = np.eye(4) - k @ h
        p = i_kh @ p @ i_kh.T + k @ r @ k.T
        p = 0.5 * (p + p.T)
        return True

    for i in range(n):
        v = vals[i]
        kind = int(types[i])
        if kind == PREDICT:
            dt, a, w = v[0], v[1], v[2]
            h, sp = x[2], x[3]
            c, s = np.cos(h), np.sin(h)
            f = np.eye(4)
            f[0, 2], f[0, 3] = -sp * s * dt, c * dt
            f[1, 2], f[1, 3] = sp * c * dt, s * dt
            g = np.zeros((4, 2))
            g[2, 1], g[3, 0] = dt, dt
            x = np.array([x[0] + sp * c * dt, x[1] + sp * s * dt, wrap(h + w * dt), sp + a * dt])
            p = f @ p @ f.T + g @ qu @ g.T
        elif kind == GPS:
            h = np.zeros((2, 4))
            h[0, 0] = h[1, 1] = 1.0
            update(h, np.array([v[0] - x[0], v[1] - x[1]]), np.eye(2) * v[2] ** 2)
        elif kind == LANDMARK:
            dx, dy = v[0] - x[0], v[1] - x[1]
            q = dx * dx + dy * dy
            rho = np.sqrt(q)
            if rho < 1e-6:
                rejected += 1
            else:
                h = np.zeros((2, 4))
                h[0, 0], h[0, 1] = -dx / rho, -dy / rho
                h[1, 0], h[1, 1], h[1, 2] = dy / q, -dx / q, -1.0
                y = np.array([v[2] - rho, wrap(v[3] - (np.arctan2(dy, dx) - x[2]))])
                update(h, y, np.diag([v[4] ** 2, v[5] ** 2]))
        else:
            raise ValueError(f"unknown event type {kind}")
        out_x[i] = x
        if out_p is not None:
            out_p[i] = p
    return out_x, out_p, rejected
