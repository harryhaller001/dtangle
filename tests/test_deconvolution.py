from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from dtangle import deconvolut


def _toy_matrix() -> np.ndarray:
    # 4 samples x 3 genes with two clear pure groups.
    return np.array(
        [
            [3.0, 1.0, 0.0],
            [2.8, 1.1, 0.1],
            [0.3, 2.9, 0.1],
            [0.2, 3.1, 0.0],
        ],
        dtype=float,
    )


def test_deconvolut_returns_expected_output_structure() -> None:
    """Test that deconvolut returns a dictionary with the expected keys and types when using pure_samples and markers."""
    y = _toy_matrix()
    pure_samples = {"A": [0, 1], "B": [2, 3]}
    markers = {"A": [0], "B": [1]}

    out = deconvolut(y, pure_samples=pure_samples, markers=markers, n_markers=1)

    estimates = out["estimates"]
    assert isinstance(estimates, pd.DataFrame)
    assert list(estimates.columns) == ["A", "B"]
    assert estimates.shape == (4, 2)
    np.testing.assert_allclose(estimates.to_numpy().sum(axis=1), np.ones(4), rtol=1e-8, atol=1e-8)

    n_markers = out["n_markers"]
    assert isinstance(n_markers, np.ndarray)
    np.testing.assert_array_equal(n_markers, np.array([1, 1]))
    assert out["gamma"] == 1.0


def test_references_mode_returns_only_mixture_rows() -> None:
    """Test that deconvolut returns estimates only for the rows in the mixture when using references and markers."""
    references = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.2, 3.0, 0.0],
        ],
        dtype=float,
    )
    mixture = np.array(
        [
            [2.5, 1.4, 0.0],
            [0.5, 2.7, 0.1],
        ],
        dtype=float,
    )

    out = deconvolut(mixture, references=references, markers=[[0], [1]], n_markers=1)
    estimates = out["estimates"]

    assert isinstance(estimates, pd.DataFrame)
    assert estimates.shape == (2, 2)
    assert list(estimates.index) == ["sample_0", "sample_1"]
    np.testing.assert_allclose(estimates.to_numpy().sum(axis=1), np.ones(2), rtol=1e-8, atol=1e-8)


def test_anndata_mode_writes_to_obsm_and_uns_inplace() -> None:
    """Test that deconvolut writes results to obsm and uns in-place when using an AnnData object."""
    y = _toy_matrix()
    adata = AnnData(X=y)
    adata.obs_names = ["s0", "s1", "s2", "s3"]
    adata.var_names = ["g0", "g1", "g2"]

    result = deconvolut(
        adata,
        pure_samples={"A": ["s0", "s1"], "B": ["s2", "s3"]},
        markers={"A": ["g0"], "B": ["g1"]},
        n_markers=1,
        key_added="deconv",
    )

    assert result is None
    assert "deconv" in adata.obsm
    assert "deconv" in adata.uns
    assert list(adata.obsm["deconv"].index) == ["s0", "s1", "s2", "s3"]
    assert list(adata.obsm["deconv"].columns) == ["A", "B"]
    np.testing.assert_array_equal(adata.uns["deconv"]["n_markers"], np.array([1, 1]))


def test_anndata_mode_supports_pure_samples_col_mapping() -> None:
    """Test that pure_samples can be resolved from an obs column."""
    y = _toy_matrix()
    adata = AnnData(X=y)
    adata.obs_names = ["s0", "s1", "s2", "s3"]
    adata.var_names = ["g0", "g1", "g2"]
    adata.obs["accession"] = ["L1", "L2", "B1", "B2"]

    result = deconvolut(
        adata,
        pure_samples_col="accession",
        pure_samples={"A": ["L1", "L2"], "B": ["B1", "B2"]},
        markers={"A": ["g0"], "B": ["g1"]},
        n_markers=1,
        key_added="by_obs",
    )

    assert result is None
    assert "by_obs" in adata.obsm
    assert list(adata.obsm["by_obs"].columns) == ["A", "B"]


def test_deconvolut_raises_when_pure_samples_col_missing() -> None:
    """Test that deconvolut raises when pure_samples_col is not in obs."""
    adata = AnnData(X=_toy_matrix())
    with pytest.raises(KeyError, match="AnnData .obs has no 'accession' column"):
        deconvolut(
            adata,
            pure_samples_col="accession",
            pure_samples={"A": ["L1"], "B": ["B1"]},
            markers={"A": [0], "B": [1]},
            n_markers=1,
        )


def test_anndata_copy_mode_returns_new_object() -> None:
    """Test that deconvolut returns a new AnnData object when copy=True."""
    y = _toy_matrix()
    adata = AnnData(X=y)
    adata.obs_names = ["s0", "s1", "s2", "s3"]
    adata.var_names = ["g0", "g1", "g2"]

    out = deconvolut(
        adata,
        pure_samples={"A": [0, 1], "B": [2, 3]},
        markers={"A": [0], "B": [1]},
        n_markers=1,
        copy=True,
        key_added="dt",
    )

    assert isinstance(out, AnnData)
    assert out is not adata
    assert "dt" in out.obsm
    assert "dt" in out.uns
    assert "dt" not in adata.obsm
    assert "dt" not in adata.uns


def test_deconvolut_raises_for_invalid_gamma() -> None:
    """Test that deconvolut raises a ValueError when gamma is invalid."""
    with pytest.raises(ValueError, match="gamma must be > 0"):
        deconvolut(_toy_matrix(), pure_samples={"A": [0, 1], "B": [2, 3]}, markers={"A": [0], "B": [1]}, gamma=0)


def test_deconvolut_raises_for_pure_samples_col_with_array_input() -> None:
    """Test that deconvolut requires AnnData when pure_samples_col is provided."""
    with pytest.raises(TypeError, match="Y must be AnnData when pure_samples_col is set"):
        deconvolut(
            _toy_matrix(),
            pure_samples_col="accession",
            pure_samples={"A": ["L1"], "B": ["B1"]},
            markers={"A": [0], "B": [1]},
            n_markers=1,
        )


def test_deconvolut_raises_when_no_references_or_pure_samples() -> None:
    """Test that deconvolut raises a ValueError when neither references nor pure_samples are provided."""
    with pytest.raises(ValueError, match="Either references or pure_samples must be provided"):
        deconvolut(_toy_matrix())


def test_deconvolut_raises_for_unknown_data_type() -> None:
    """Test that deconvolut raises a ValueError when an unknown data_type is provided."""
    with pytest.raises(ValueError, match="Unknown data_type"):
        deconvolut(
            _toy_matrix(),
            pure_samples={"A": [0, 1], "B": [2, 3]},
            markers={"A": [0], "B": [1]},
            data_type="not-a-real-type",
        )


def test_deconvolut_raises_for_unknown_marker_gene_name() -> None:
    """Test that deconvolut raises a ValueError when a marker gene name is not found in the data."""
    y = _toy_matrix()
    adata = AnnData(X=y)
    adata.obs_names = ["s0", "s1", "s2", "s3"]
    adata.var_names = ["g0", "g1", "g2"]

    with pytest.raises(ValueError, match="Unknown marker gene"):
        deconvolut(
            adata,
            pure_samples={"A": ["s0", "s1"], "B": ["s2", "s3"]},
            markers={"A": ["g0"], "B": ["missing_gene"]},
            n_markers=1,
        )
