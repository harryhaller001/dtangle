"""Internal data containers for dtangle deconvolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MatrixInput:
    matrix: np.ndarray
    sample_names: pd.Index
    gene_names: pd.Index


@dataclass(slots=True)
class PreparedInput:
    y: np.ndarray
    pure_samples: list[np.ndarray]
    cell_types: list[str]
    gene_names: pd.Index
