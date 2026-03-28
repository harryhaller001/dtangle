from __future__ import annotations

import numpy as np
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


def _make_reference_adata() -> AnnData:
    reference = AnnData(X=_toy_matrix())
    reference.obs_names = ["r0", "r1", "r2", "r3"]
    reference.var_names = ["g0", "g1", "g2"]
    reference.obs["cell_type"] = ["A", "A", "B", "B"]
    return reference


def _make_mixture_adata() -> AnnData:
    mixture = AnnData(
        X=np.array(
            [
                [2.5, 1.4, 0.0],
                [0.5, 2.7, 0.1],
            ],
            dtype=float,
        )
    )
    mixture.obs_names = ["m0", "m1"]
    mixture.var_names = ["g0", "g1", "g2"]
    return mixture


def test_deconvolut_writes_to_obsm_and_uns_inplace() -> None:
    """Test that deconvolut writes results in-place for AnnData inputs."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()

    result = deconvolut(
        mixture,
        reference,
        "cell_type",
        markers={"A": ["g0"], "B": ["g1"]},
        n_markers=1,
        key_added="deconv",
    )

    assert result is None
    assert "deconv" in mixture.obsm
    assert "deconv" in mixture.uns
    assert list(mixture.obsm["deconv"].index) == ["m0", "m1"]
    assert list(mixture.obsm["deconv"].columns) == ["A", "B"]
    np.testing.assert_array_equal(mixture.uns["deconv"]["n_markers"], np.array([1, 1]))
    np.testing.assert_allclose(mixture.obsm["deconv"].to_numpy().sum(axis=1), np.ones(2), rtol=1e-8, atol=1e-8)


def test_deconvolut_groups_reference_samples_by_obs_key() -> None:
    """Test that reference samples are grouped by the provided annotation key."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()
    reference.obs["alt_label"] = ["L", "L", "B", "B"]

    result = deconvolut(
        mixture,
        reference,
        "alt_label",
        markers={"L": ["g0"], "B": ["g1"]},
        n_markers=1,
        key_added="by_label",
    )

    assert result is None
    assert "by_label" in mixture.obsm
    assert list(mixture.obsm["by_label"].columns) == ["L", "B"]


def test_deconvolut_raises_when_reference_annotation_missing() -> None:
    """Test that deconvolut raises when the reference annotation column is absent."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()
    with pytest.raises(KeyError, match="AnnData .obs has no 'accession' column"):
        deconvolut(mixture, reference, "accession", markers={"A": ["g0"], "B": ["g1"]}, n_markers=1)


def test_deconvolut_raises_when_reference_annotation_has_missing_values() -> None:
    """Test that deconvolut raises when the reference annotation contains NA values."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()
    reference.obs["cell_type"] = ["A", "A", None, "B"]
    with pytest.raises(ValueError, match="contains missing values"):
        deconvolut(mixture, reference, "cell_type", markers={"A": ["g0"], "B": ["g1"]}, n_markers=1)


def test_anndata_copy_mode_returns_new_object() -> None:
    """Test that deconvolut returns a new AnnData object when copy=True."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()

    out = deconvolut(
        mixture,
        reference,
        "cell_type",
        markers={"A": ["g0"], "B": ["g1"]},
        n_markers=1,
        copy=True,
        key_added="dt",
    )

    assert isinstance(out, AnnData)
    assert out is not mixture
    assert "dt" in out.obsm
    assert "dt" in out.uns
    assert "dt" not in mixture.obsm
    assert "dt" not in mixture.uns


def test_deconvolut_raises_for_invalid_gamma() -> None:
    """Test that deconvolut raises a ValueError when gamma is invalid."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()
    with pytest.raises(ValueError, match="gamma must be > 0"):
        deconvolut(mixture, reference, "cell_type", markers={"A": ["g0"], "B": ["g1"]}, gamma=0)


def test_deconvolut_raises_for_non_anndata_inputs() -> None:
    """Test that deconvolut requires AnnData objects for both Y and references."""
    reference = _make_reference_adata()
    mixture = _make_mixture_adata()

    with pytest.raises(TypeError, match="Y must be AnnData"):
        deconvolut(_toy_matrix(), reference, "cell_type", markers={"A": ["g0"], "B": ["g1"]}, n_markers=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="references must be AnnData"):
        deconvolut(mixture, _toy_matrix(), "cell_type", markers={"A": ["g0"], "B": ["g1"]}, n_markers=1)  # type: ignore[arg-type]


def test_deconvolut_raises_for_unknown_data_type() -> None:
    """Test that deconvolut raises a ValueError when an unknown data_type is provided."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()
    with pytest.raises(ValueError, match="Unknown data_type"):
        deconvolut(
            mixture,
            reference,
            "cell_type",
            markers={"A": ["g0"], "B": ["g1"]},
            data_type="not-a-real-type",
        )


def test_deconvolut_raises_for_unknown_marker_gene_name() -> None:
    """Test that deconvolut raises a ValueError when a marker gene name is not found in the data."""
    mixture = _make_mixture_adata()
    reference = _make_reference_adata()

    with pytest.raises(ValueError, match="Unknown marker gene"):
        deconvolut(
            mixture,
            reference,
            "cell_type",
            markers={"A": ["g0"], "B": ["missing_gene"]},
            n_markers=1,
        )
