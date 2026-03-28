from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scanpro.result import ScanproResult

from dtangle import differential_proportion


def _make_proportion_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "condition": ["ctrl", "ctrl", "trt", "trt"],
            "sample": ["s1", "s2", "s3", "s4"],
            "batch": ["b1", "b2", "b1", "b2"],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    adata = AnnData(X=np.zeros((4, 1)), obs=obs)
    adata.obsm["dtangle"] = pd.DataFrame(
        {
            "A": [0.70, 0.68, 0.28, 0.30],
            "B": [0.20, 0.22, 0.52, 0.50],
            "C": [0.07, 0.06, 0.15, 0.14],
            "D": [0.03, 0.04, 0.05, 0.06],
        },
        index=obs.index,
    )
    return adata


def _make_three_condition_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "condition": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "sample": ["s1", "s2", "s3", "s4", "s5", "s6"],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )
    adata = AnnData(X=np.zeros((6, 1)), obs=obs)
    adata.obsm["dtangle"] = pd.DataFrame(
        {
            "A": [0.70, 0.67, 0.45, 0.48, 0.20, 0.18],
            "B": [0.18, 0.20, 0.30, 0.28, 0.45, 0.47],
            "C": [0.07, 0.08, 0.15, 0.14, 0.23, 0.24],
            "D": [0.05, 0.05, 0.10, 0.10, 0.12, 0.11],
        },
        index=obs.index,
    )
    return adata


def test_differential_proportion_ttest_and_uns_writeback() -> None:
    adata = _make_proportion_adata()

    out = differential_proportion(
        adata,
        conds_col="condition",
        proportions_key="dtangle",
        samples_col="sample",
        key_added="dp",
    )

    assert isinstance(out, ScanproResult)
    assert isinstance(out.results, pd.DataFrame)
    assert all(col in out.results.columns for col in ["baseline_props", "p_values", "adjusted_p_values", "prop_ratio"])
    assert "dp" in adata.uns
    assert "results" in adata.uns["dp"]
    assert isinstance(adata.uns["dp"]["results"], pd.DataFrame)
    assert out.pairwise is False


def test_differential_proportion_plot_does_not_raise() -> None:
    adata = _make_proportion_adata()

    out = differential_proportion(
        adata,
        conds_col="condition",
        proportions_key="dtangle",
        samples_col="sample",
    )

    axes = out.plot(clusters=["A", "B"], show=False)
    assert isinstance(axes, list)


def test_differential_proportion_with_covariate() -> None:
    adata = _make_proportion_adata()

    out = differential_proportion(
        adata,
        conds_col="condition",
        proportions_key="dtangle",
        samples_col="sample",
        covariates="batch",
    )

    assert isinstance(out, ScanproResult)
    assert out.covariates == ["batch"]
    non_condition_cols = [col for col in out.design.columns if col not in out.all_conditions]
    assert len(non_condition_cols) >= 1


def test_differential_proportion_anova_path() -> None:
    adata = _make_three_condition_adata()

    out = differential_proportion(
        adata,
        conds_col="condition",
        proportions_key="dtangle",
        samples_col="sample",
    )

    assert isinstance(out, ScanproResult)
    assert all(col in out.results.columns for col in ["f_statistics", "p_values", "adjusted_p_values"])


def test_differential_proportion_raises_for_missing_obsm_key() -> None:
    adata = _make_proportion_adata()

    with pytest.raises(KeyError, match="obsm"):
        differential_proportion(adata, conds_col="condition", proportions_key="missing")


def test_differential_proportion_raises_for_index_mismatch() -> None:
    adata = _make_proportion_adata()
    adata.obsm["bad"] = adata.obsm["dtangle"].copy()
    adata.obsm["bad"].index = pd.Index(["x1", "x2", "x3", "x4"])

    with pytest.raises(ValueError, match="index"):
        differential_proportion(adata, conds_col="condition", proportions_key="bad")


def test_differential_proportion_raises_for_duplicate_samples() -> None:
    adata = _make_proportion_adata()
    adata.obs["sample_dup"] = ["s", "s", "t", "u"]

    with pytest.raises(ValueError, match="unique"):
        differential_proportion(
            adata,
            conds_col="condition",
            proportions_key="dtangle",
            samples_col="sample_dup",
        )
