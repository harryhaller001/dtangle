"""Public deconvolution API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from dtangle._estimation import baseline_exprs, est_phats
from dtangle._input import combine_inputs, extract_input
from dtangle._markers import get_gamma, process_markers


def _pure_samples_from_reference_obs(reference: AnnData, reference_annotation_col: str) -> Mapping[str, Sequence[str]]:
    if reference_annotation_col not in reference.obs.columns:
        raise KeyError(f"AnnData .obs has no '{reference_annotation_col}' column")

    labels = reference.obs[reference_annotation_col]
    if pd.isna(labels).any():
        raise ValueError(f"AnnData .obs['{reference_annotation_col}'] contains missing values")

    resolved: dict[str, list[str]] = {}
    for sample_name, label in zip(reference.obs_names.astype(str), labels, strict=True):
        cell_type = str(label)
        resolved.setdefault(cell_type, []).append(str(sample_name))

    if not resolved:
        raise ValueError("references AnnData must contain at least one sample")

    return resolved


def deconvolut(
    Y: AnnData,
    references: AnnData,
    reference_annotation_col: str,
    n_markers: int | float | Sequence[int | float] | None = None,
    data_type: str | None = None,
    gamma: float | None = None,
    markers: Mapping[str, Sequence[int | str]] | Sequence[Sequence[int | str]] | None = None,
    marker_method: str = "ratio",
    summary_fn: Callable[[np.ndarray], float] = np.mean,
    *,
    layer: str | None = None,
    var_key: str | None = None,
    key_added: str = "dtangle",
    copy: bool = False,
) -> AnnData | None:
    """Estimate cell type mixing proportions using a dtangle-style estimator.

    Args:
        Y: Mixture expression AnnData with shape (samples, genes).
        references: Reference expression AnnData where each row is a pure sample.
        reference_annotation_col: AnnData .obs column in references containing cell
            type annotations used to group pure samples.
        n_markers: Marker count control. Supports scalar integer, per-type integer
            vector, scalar fraction in (0,1), or per-type fraction vector.
        data_type: Optional data type key used to choose built-in gamma.
        gamma: Optional explicit gamma overriding data_type.
        markers: Optional marker indices per cell type. If omitted, markers are
            computed internally using marker_method.
        marker_method: Marker ranking method: ratio, regression, diff, p.value.
        summary_fn: Summary function used by baseline and per-sample aggregation.
        layer: Optional AnnData layer key to use instead of X.
        var_key: Optional adata.var key used for feature alignment and marker names.
        key_added: Output key for AnnData mode.
        copy: Return modified AnnData in AnnData mode.

    Returns:
        None (in-place) or AnnData if copy=True. Results are written to
        adata.obsm[key_added] and adata.uns[key_added].
    """
    if not isinstance(Y, AnnData):
        raise TypeError("Y must be AnnData")
    if not isinstance(references, AnnData):
        raise TypeError("references must be AnnData")

    if gamma is None:
        gamma = get_gamma(data_type)

    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    pure_samples = _pure_samples_from_reference_obs(references, reference_annotation_col)

    y_in = extract_input(Y, layer=layer, var_key=var_key)
    ref_in = extract_input(references, layer=layer, var_key=var_key)

    prepared, pure_input_rows = combine_inputs(y_in, ref_in, pure_samples)

    marker_list, marker_counts = process_markers(
        prepared.y,
        prepared.pure_samples,
        prepared.cell_types,
        prepared.gene_names,
        n_markers=n_markers,
        markers=markers,
        marker_method=marker_method,
        gamma=gamma,
    )

    baseline = baseline_exprs(prepared.y, prepared.pure_samples, marker_list, summary_fn=summary_fn)
    estimates_all = est_phats(prepared.y, marker_list, baseline, gamma=gamma, summary_fn=summary_fn)

    keep_rows = np.ones(estimates_all.shape[0], dtype=bool)
    keep_rows[pure_input_rows] = False
    estimates = estimates_all[keep_rows, :]
    sample_names = y_in.sample_names

    estimates_df = pd.DataFrame(estimates, index=pd.Index(sample_names), columns=pd.Index(prepared.cell_types))

    adata = Y.copy() if copy else Y
    adata.obsm[key_added] = estimates_df
    adata.uns[key_added] = {
        "markers": {ct: list(vals) for ct, vals in marker_list.items()},
        "n_markers": np.asarray(marker_counts, dtype=int),
        "gamma": float(gamma),
        "marker_method": marker_method,
        "cell_types": prepared.cell_types,
    }
    return adata if copy else None
