"""Marker discovery and marker count utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

GAMMA_BY_DATA_TYPE: dict[str, float] = {
    "microarray-probe": 0.4522564,
    "microarray-gene": 0.6999978,
    "rna-seq": 0.9433902,
}


def normalize_markers(
    markers: Mapping[str, Sequence[int | str]] | Sequence[Sequence[int | str]],
    *,
    cell_types: list[str],
    gene_names: pd.Index,
) -> dict[str, np.ndarray]:
    if isinstance(markers, Mapping):
        marker_map = cast(Mapping[str, Sequence[int | str]], markers)
        by_type: dict[str, np.ndarray] = {}
        for ct in cell_types:
            if ct not in marker_map:
                raise ValueError(f"Missing markers for cell type: {ct}")
            by_type[ct] = marker_items_to_indices(marker_map[ct], gene_names)
        return by_type

    marker_seq = cast(Sequence[Sequence[int | str]], markers)
    marker_values = list(marker_seq)
    if len(marker_values) != len(cell_types):
        raise ValueError("markers length must match number of cell types")

    return {ct: marker_items_to_indices(marker_values[i], gene_names) for i, ct in enumerate(cell_types)}


def marker_items_to_indices(items: Sequence[int | str], gene_names: pd.Index) -> np.ndarray:
    gene_to_pos = {name: i for i, name in enumerate(gene_names.astype(str))}
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            if item not in gene_to_pos:
                raise ValueError(f"Unknown marker gene: {item}")
            out.append(gene_to_pos[item])
        else:
            out.append(int(item))
    arr = np.asarray(out, dtype=int)
    if arr.size == 0:
        raise ValueError("Each marker group must contain at least one marker")
    return arr


def get_gamma(data_type: str | None) -> float:
    if data_type is None:
        return 1.0
    if data_type not in GAMMA_BY_DATA_TYPE:
        valid = ", ".join(sorted(GAMMA_BY_DATA_TYPE))
        raise ValueError(f"Unknown data_type '{data_type}'. Expected one of: {valid}")
    return GAMMA_BY_DATA_TYPE[data_type]


def find_markers(
    y: np.ndarray,
    pure_samples: list[np.ndarray],
    cell_types: list[str],
    gamma: float,
    marker_method: str,
) -> dict[str, np.ndarray]:
    k = len(pure_samples)
    n_genes = y.shape[1]

    method = marker_method
    if method == "p.value" and any(len(group) == 1 for group in pure_samples):
        method = "diff"

    c = np.zeros((k, n_genes), dtype=float)

    if method == "ratio":
        eta_hats = []
        for rows in pure_samples:
            eta_hats.append(np.mean(np.power(2.0, y[rows, :]), axis=0) / gamma)
        eta_hats_arr = np.vstack(eta_hats)
        for i in range(k):
            denom = np.sum(eta_hats_arr[np.arange(k) != i, :], axis=0)
            denom = np.where(denom == 0, np.finfo(float).eps, denom)
            c[i, :] = eta_hats_arr[i, :] / denom

    elif method == "regression":
        pure = np.concatenate(pure_samples)
        yp = y[pure, :]
        for i, rows in enumerate(pure_samples):
            x = np.isin(pure, rows).astype(float)
            x0 = np.column_stack([np.ones_like(x), x])
            beta, _, _, _ = np.linalg.lstsq(x0, yp, rcond=None)
            c[i, :] = beta[1, :]

    elif method == "diff":
        for i, rows in enumerate(pure_samples):
            c[i, :] = np.median(y[rows, :], axis=0)
        sorted_c = np.sort(c, axis=0)
        second_highest = sorted_c[-2, :] if k > 1 else sorted_c[-1, :]
        c = c - second_highest

    elif method == "p.value":
        for i, rows in enumerate(pure_samples):
            c[i, :] = np.mean(y[rows, :], axis=0)

        out = np.zeros((k, n_genes), dtype=float)
        for g in range(n_genes):
            xg = c[:, g]
            top = int(np.argmax(xg))
            second = int(np.argsort(xg)[-2]) if k > 1 else top

            pvals = np.zeros(k, dtype=float)
            for j in range(k):
                x1 = y[pure_samples[j], g]
                x2 = y[pure_samples[second], g]
                if len(x1) < 2 or len(x2) < 2:
                    pvals[j] = np.nan
                    continue
                _, pval = stats.ttest_ind(x1, x2, equal_var=True, nan_policy="omit")
                pvals[j] = np.nan if np.isnan(pval) else pval
            pvals[np.arange(k) != top] = 0.0
            out[:, g] = pvals
        c = out

    else:
        raise ValueError("Marker method not found")

    top_idx = np.argmax(c, axis=0)
    top_val = c[top_idx, np.arange(n_genes)]

    markers_by_type: dict[str, np.ndarray] = {}
    for i, ct in enumerate(cell_types):
        genes = np.where(top_idx == i)[0]
        if genes.size == 0:
            # Fall back to per-type ranking when no gene is uniquely assigned.
            markers_by_type[ct] = np.argsort(-c[i, :])
            continue
        order = np.argsort(-top_val[genes])
        markers_by_type[ct] = genes[order]

    return markers_by_type


def resolve_n_markers(
    n_markers: int | float | Sequence[int | float] | None,
    marker_lengths: list[int],
) -> list[int]:
    k = len(marker_lengths)
    if n_markers is None:
        values = [max(int(np.floor(0.1 * marker_lengths[i])), 1) for i in range(k)]
        return values

    if isinstance(n_markers, int | float | np.integer | np.floating):
        vals = [float(n_markers)] * k
    else:
        vals = [float(v) for v in n_markers]
        if len(vals) != k:
            raise ValueError("n_markers length must match number of cell types")

    out: list[int] = []
    for i, val in enumerate(vals):
        if val <= 0:
            raise ValueError("n_markers values must be > 0")
        if val < 1:
            out.append(max(int(np.floor(val * marker_lengths[i])), 1))
        else:
            out.append(max(int(val), 1))
    return out


def process_markers(
    y: np.ndarray,
    pure_samples: list[np.ndarray],
    cell_types: list[str],
    gene_names: pd.Index,
    *,
    n_markers: int | float | Sequence[int | float] | None,
    markers: Mapping[str, Sequence[int | str]] | Sequence[Sequence[int | str]] | None,
    marker_method: str,
    gamma: float,
) -> tuple[dict[str, np.ndarray], list[int]]:
    if markers is None:
        ranked = find_markers(y, pure_samples, cell_types, gamma=gamma, marker_method=marker_method)
    else:
        ranked = normalize_markers(markers, cell_types=cell_types, gene_names=gene_names)

    lengths = [len(ranked[ct]) for ct in cell_types]
    counts = resolve_n_markers(n_markers=n_markers, marker_lengths=lengths)

    out: dict[str, np.ndarray] = {}
    for ct, n in zip(cell_types, counts, strict=True):
        n_use = min(n, len(ranked[ct]))
        out[ct] = np.asarray(ranked[ct][:n_use], dtype=int)

    return out, counts
