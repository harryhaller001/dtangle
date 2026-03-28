"""Differential proportion testing for dtangle outputs."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpro.get_transformed_props import get_transformed_props_counts
from scanpro.linear_model import create_design
from scanpro.result import ScanproResult

from scanpro.scanpro import anova, t_test


def _as_covariate_list(covariates: str | list[str] | None) -> list[str]:
    if covariates is None:
        return []
    if isinstance(covariates, str):
        return [covariates]
    if not isinstance(covariates, list):
        raise ValueError("covariates must be a string or a list of strings")
    if not all(isinstance(cov, str) for cov in covariates):
        raise ValueError("covariates must contain only strings")
    return covariates


def _validate_conditions(
    obs: pd.DataFrame,
    conds_col: str,
    conditions: list[str] | tuple[str, ...] | np.ndarray | None,
) -> list[str]:
    all_conditions = obs[conds_col].astype(str).unique().tolist()
    if conditions is None:
        selected = all_conditions
    else:
        if not isinstance(conditions, (list, tuple, np.ndarray)):
            raise ValueError("conditions must be a list, tuple, or numpy array")
        selected = [str(cond) for cond in conditions]
        missing = [cond for cond in selected if cond not in all_conditions]
        if missing:
            raise ValueError(
                f"The following conditions were not found in adata.obs[{conds_col!r}]: {', '.join(missing)}"
            )

    if len(selected) < 2:
        raise ValueError("At least two conditions are required for differential proportion testing")
    return selected


def _get_proportion_matrix(adata: AnnData, proportions_key: str) -> pd.DataFrame:
    if proportions_key not in adata.obsm:
        raise KeyError(f"AnnData .obsm has no {proportions_key!r} key")

    props_raw = adata.obsm[proportions_key]
    if isinstance(props_raw, pd.DataFrame):
        props = props_raw.copy()
    else:
        props = pd.DataFrame(props_raw, index=adata.obs_names)

    props.index = props.index.astype(str)
    obs_index = adata.obs_names.astype(str)
    if len(props.index) != len(obs_index):
        raise ValueError("Proportion matrix in obsm must have the same number of rows as adata.obs")
    if not props.index.equals(obs_index):
        raise ValueError(
            "Proportion matrix index in obsm must match adata.obs_names exactly. "
            "Ensure proportions are aligned to samples in AnnData."
        )

    non_numeric_cols = props.columns[~props.dtypes.apply(pd.api.types.is_numeric_dtype)].tolist()
    if non_numeric_cols:
        raise ValueError(
            f"All proportion columns must be numeric. Non-numeric columns: {', '.join(map(str, non_numeric_cols))}"
        )

    props.columns = props.columns.astype(str)
    return props


def _subset_design_and_props(
    design: pd.DataFrame,
    props: pd.DataFrame,
    prop_trans: pd.DataFrame,
    all_conditions: list[str],
    conditions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design_columns = [col for col in design.columns if col not in all_conditions or col in conditions]
    design_sub = design[design_columns]

    included_samples = design_sub[design_sub.sum(axis=1) != 0].index.tolist()
    design_sub = design_sub.loc[included_samples, :]
    props_sub = props.loc[included_samples, :]
    prop_trans_sub = prop_trans.loc[included_samples, :]

    # Remove clusters absent in selected samples.
    nonzero = props_sub.sum(axis=0) != 0
    props_sub = props_sub.loc[:, nonzero]
    prop_trans_sub = prop_trans_sub.loc[:, nonzero]
    return design_sub, props_sub, prop_trans_sub


def _add_baseline_props(results: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    baseline = props.mean(axis=0).reindex(results.index)
    out = results.copy()
    out.insert(0, "baseline_props", baseline.values)
    out.index.name = "clusters"
    return out


def differential_proportion(
    adata: AnnData,
    conds_col: str,
    *,
    proportions_key: str = "dtangle",
    samples_col: str | None = None,
    covariates: str | list[str] | None = None,
    conditions: list[str] | tuple[str, ...] | np.ndarray | None = None,
    transform: Literal["logit", "arcsin"] = "logit",
    robust: bool = True,
    key_added: str = "differential_proportion",
) -> ScanproResult:
    """Test differential cell-type proportions from dtangle proportion outputs.

    Args:
        adata: AnnData object containing sample metadata in ``adata.obs`` and
            estimated cell-type proportions in ``adata.obsm[proportions_key]``.
        conds_col: Condition column in ``adata.obs``.
        proportions_key: Key in ``adata.obsm`` with sample x cell-type
            proportion matrix.
        samples_col: Optional sample id column in ``adata.obs``. If omitted,
            ``adata.obs_names`` are used.
        covariates: Optional covariate column(s) in ``adata.obs``.
        conditions: Optional subset of conditions to test. Must include at
            least two conditions.
        transform: Transformation used before modeling (``logit`` or ``arcsin``).
        robust: Use robust empirical Bayes in scanpro statistical tests.
        key_added: Key used to store results in ``adata.uns``.

    Returns:
        ScanproResult with statistical outputs and intermediate matrices.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be AnnData")

    if transform not in {"logit", "arcsin"}:
        raise ValueError("transform must be either 'logit' or 'arcsin'")

    if conds_col not in adata.obs.columns:
        raise KeyError(f"AnnData .obs has no {conds_col!r} column")

    covariate_list = _as_covariate_list(covariates)
    missing_covariates = [cov for cov in covariate_list if cov not in adata.obs.columns]
    if missing_covariates:
        raise KeyError(f"The following covariates are missing from adata.obs: {', '.join(missing_covariates)}")

    props = _get_proportion_matrix(adata, proportions_key)

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs[conds_col] = obs[conds_col].astype(str)
    selected_conditions = _validate_conditions(obs, conds_col, conditions)

    sample_col_name = "__sample_id"
    if samples_col is not None:
        if samples_col not in obs.columns:
            raise KeyError(f"AnnData .obs has no {samples_col!r} column")
        obs[sample_col_name] = obs[samples_col].astype(str)
    else:
        obs[sample_col_name] = obs.index.astype(str)

    matrix_input = props.copy()
    matrix_input[sample_col_name] = obs[sample_col_name].values
    matrix_input[conds_col] = obs[conds_col].values
    for cov in covariate_list:
        matrix_input[cov] = obs[cov].values

    # Validate unique sample ids for matrix-based model input.
    if matrix_input[sample_col_name].duplicated().any():
        raise ValueError(
            "sample identifiers must be unique for matrix-based differential_proportion. "
            "Provide a unique samples_col or ensure adata.obs_names are unique."
        )

    props_matrix, prop_trans = get_transformed_props_counts(
        matrix_input,
        transform=transform,
        sample_col=sample_col_name,
        meta_cols=[conds_col] + covariate_list,
        normalize=False,
    )

    design = create_design(
        data=matrix_input,
        sample_col=sample_col_name,
        conds_col=conds_col,
        covariates=covariate_list,
    )

    all_conditions = obs[conds_col].unique().tolist()
    design_sub, props_sub, prop_trans_sub = _subset_design_and_props(
        design,
        props_matrix,
        prop_trans,
        all_conditions,
        selected_conditions,
    )

    if len(selected_conditions) == 2:
        contrasts = np.zeros(len(design_sub.columns))
        contrasts[0] = 1
        contrasts[1] = -1
        results = t_test(props_sub, prop_trans_sub, design_sub, contrasts, robust=robust, verbosity=0)
    else:
        coef = np.arange(len(selected_conditions))
        results = anova(props_sub, prop_trans_sub, design_sub, coef, robust=robust, verbosity=0)

    results = _add_baseline_props(results, props_sub)

    out = ScanproResult()
    out.results = results
    out.props = props_matrix
    out.prop_trans = prop_trans
    out.design = design
    out.all_conditions = all_conditions
    out.conditions = selected_conditions
    out.conds_col = conds_col
    out.covariates = covariate_list
    out.proportions_key = proportions_key
    out.transform = transform
    out.robust = robust
    out.pairwise = False
    out.repd = "all"

    adata.uns[key_added] = {
        "results": results.copy(),
        "proportions_key": proportions_key,
        "conditions": selected_conditions,
        "all_conditions": all_conditions,
        "conds_col": conds_col,
        "covariates": covariate_list,
        "transform": transform,
        "robust": robust,
        "pairwise": False,
        "repd": "all",
    }

    return out
