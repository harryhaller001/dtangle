"""Core dtangle-style proportion estimation routines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def baseline_exprs(
    y: np.ndarray,
    pure_samples: list[np.ndarray],
    markers: dict[str, np.ndarray],
    summary_fn: Callable[[np.ndarray], float],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, (ct, marker_idx) in enumerate(markers.items()):
        vals = []
        for gene_idx in marker_idx:
            vals.append(float(summary_fn(y[pure_samples[i], int(gene_idx)])))
        out[ct] = np.asarray(vals, dtype=float)
    return out


def est_phats(
    y: np.ndarray,
    markers: dict[str, np.ndarray],
    baseline_ests: dict[str, np.ndarray],
    gamma: float,
    summary_fn: Callable[[np.ndarray], float],
    inv_scale: Callable[[np.ndarray], np.ndarray] = lambda x: np.power(2.0, x),
) -> np.ndarray:
    contribs = []
    for ct, marker_idx in markers.items():
        yi = y[:, marker_idx]
        baseline = baseline_ests[ct]
        baseline_adj = yi - baseline
        agg = np.apply_along_axis(summary_fn, axis=1, arr=baseline_adj / gamma)
        amt = inv_scale(agg)
        contribs.append(np.asarray(amt, dtype=float))

    contrib_mtx = np.column_stack(contribs)
    denom = np.sum(contrib_mtx, axis=1, keepdims=True)
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    return contrib_mtx / denom
