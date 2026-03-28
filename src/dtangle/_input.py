"""Input extraction and normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from dtangle._types import MatrixInput, PreparedInput


def _unique_feature_positions(gene_names: pd.Index) -> tuple[pd.Index, np.ndarray]:
    """Return unique feature names and first-occurrence positions."""
    first_pos: dict[str, int] = {}
    for idx, name in enumerate(gene_names):
        key = str(name)
        if key not in first_pos:
            first_pos[key] = idx

    unique_names = pd.Index(list(first_pos.keys()), dtype=str)
    positions = np.fromiter(first_pos.values(), dtype=int)
    return unique_names, positions


def extract_input(data: AnnData | np.ndarray | None, layer: str | None, var_key: str | None) -> MatrixInput:
    if data is None:
        raise ValueError("Input matrix is required")

    if isinstance(data, AnnData):
        if layer is None:
            matrix_source = data.X
        else:
            if layer not in data.layers:
                raise KeyError(f"AnnData .layers has no '{layer}' layer")
            matrix_source = data.layers[layer]

        if sparse.issparse(matrix_source):
            matrix = np.asarray(matrix_source.toarray(), dtype=float)
        else:
            matrix = np.asarray(matrix_source, dtype=float)

        if var_key is None:
            gene_names = pd.Index(data.var_names.astype(str))
        else:
            if var_key not in data.var.columns:
                raise KeyError(f"AnnData .var has no '{var_key}' column")
            gene_names = pd.Index(np.asarray(data.var[var_key], dtype=str))

        sample_names = pd.Index(data.obs_names.astype(str))
        return MatrixInput(matrix=matrix, sample_names=sample_names, gene_names=gene_names)

    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Y must be 2-dimensional")

    n_rows, n_cols = matrix.shape
    sample_names = pd.Index([f"sample_{i}" for i in range(n_rows)])
    gene_names = pd.Index([f"gene_{j}" for j in range(n_cols)])
    return MatrixInput(matrix=matrix, sample_names=sample_names, gene_names=gene_names)


def combine_inputs(
    y_input: MatrixInput,
    ref_input: MatrixInput | None,
    pure_samples: Mapping[str, Sequence[int | str]] | Sequence[Sequence[int | str]] | None,
) -> tuple[PreparedInput, np.ndarray]:
    y = y_input.matrix
    y_gene_names = y_input.gene_names

    if ref_input is not None:
        ref = ref_input.matrix

        y_unique_names, y_unique_pos = _unique_feature_positions(y_gene_names)
        ref_unique_names, ref_unique_pos = _unique_feature_positions(ref_input.gene_names)

        common = y_unique_names.intersection(ref_unique_names)
        if common.empty:
            raise ValueError("Y and references do not share any features")

        y_pos_lookup = pd.Series(y_unique_pos, index=y_unique_names)
        ref_pos_lookup = pd.Series(ref_unique_pos, index=ref_unique_names)
        y_idx = y_pos_lookup.loc[common].to_numpy(dtype=int)
        r_idx = ref_pos_lookup.loc[common].to_numpy(dtype=int)
        y = y[:, y_idx]
        ref = ref[:, r_idx]

        combined_y = np.vstack([ref, y])
        pure_rows = np.arange(ref.shape[0], dtype=int)

        if pure_samples is None:
            cell_types = list(ref_input.sample_names.astype(str))
            ps = [np.array([i], dtype=int) for i in pure_rows]
        else:
            cell_types, ps = normalize_pure_samples(pure_samples, sample_names=ref_input.sample_names)

        prepared = PreparedInput(y=combined_y, pure_samples=ps, cell_types=cell_types, gene_names=common)
        return prepared, pure_rows

    if pure_samples is None:
        raise ValueError("Either references or pure_samples must be provided")

    cell_types, ps = normalize_pure_samples(pure_samples, sample_names=y_input.sample_names)
    prepared = PreparedInput(y=y, pure_samples=ps, cell_types=cell_types, gene_names=y_gene_names)
    return prepared, np.array([], dtype=int)


def normalize_pure_samples(
    pure_samples: Mapping[str, Sequence[int | str]] | Sequence[Sequence[int | str]],
    sample_names: pd.Index,
) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(pure_samples, Mapping):
        pure_samples_map = cast(Mapping[str, Sequence[int | str]], pure_samples)
        cell_types = [str(k) for k in pure_samples_map]
        values: list[Sequence[int | str]] = list(pure_samples_map.values())
    else:
        pure_samples_seq = cast(Sequence[Sequence[int | str]], pure_samples)
        values = list(pure_samples_seq)
        cell_types = [f"cell_type_{i + 1}" for i in range(len(values))]

    if len(values) == 0:
        raise ValueError("pure_samples must contain at least one cell type")

    name_to_pos = {name: i for i, name in enumerate(sample_names.astype(str))}
    normalized: list[np.ndarray] = []
    for items in values:
        rows: list[int] = []
        for item in items:
            if isinstance(item, str):
                if item not in name_to_pos:
                    raise ValueError(f"Unknown sample label in pure_samples: {item}")
                rows.append(name_to_pos[item])
            else:
                rows.append(int(item))
        arr = np.asarray(rows, dtype=int)
        if arr.size == 0:
            raise ValueError("Each pure_samples entry must contain at least one sample")
        normalized.append(arr)

    return cell_types, normalized
