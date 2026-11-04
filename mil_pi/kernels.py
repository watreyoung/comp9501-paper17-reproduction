"""Kernel and MATLAB-utility replacements used by the MIL-PI programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class KernelSpec:
    """Description of a kernel matrix.

    Samples are represented by columns, matching the MATLAB code in the archive.
    ``gamma`` is the coefficient in ``exp(-gamma * ||x-y||^2)`` for an RBF kernel.
    """

    kernel_type: str = "linear"
    gamma: float | None = None


def as_feature_matrix(features: np.ndarray, *, name: str = "features") -> np.ndarray:
    """Return a finite 2-D float matrix with samples in columns."""

    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def l2_normalize_columns(features: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Normalize each sample column to unit L2 norm.

    Zero columns stay zero.  This is the Python equivalent of the missing
    ``L2_normalization`` utility referenced by the MATLAB demos.
    """

    array = as_feature_matrix(features)
    norms = np.linalg.norm(array, axis=0, keepdims=True)
    return array / np.maximum(norms, eps)


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=0)[:, None]
    y2 = np.sum(y * y, axis=0)[None, :]
    distances = x2 + y2 - 2.0 * (x.T @ y)
    return np.maximum(distances, 0.0)


def _default_rbf_gamma(features: np.ndarray) -> float:
    """Choose a data-dependent RBF gamma when the old utility omitted it.

    The original archive does not contain ``getKernel.m``.  We use the standard
    median-distance heuristic, ignoring zero distances.  Callers can pass an
    explicit ``gamma`` for exact reproducibility within a project.
    """

    distances = _squared_distances(features, features)
    nonzero = distances[distances > 1e-15]
    if nonzero.size == 0:
        return 1.0
    return float(1.0 / np.median(nonzero))


def get_kernel(
    features: np.ndarray,
    other: np.ndarray | None = None,
    spec: KernelSpec | dict | None = None,
) -> tuple[np.ndarray, KernelSpec] | np.ndarray:
    """Compute a linear or Gaussian kernel.

    The one-input form returns ``(kernel, resolved_spec)`` like the MATLAB
    ``[K, kernel_param] = getKernel(X, param)`` call.  With ``other`` and a
    resolved :class:`KernelSpec`, it returns only the cross-kernel matrix.
    """

    x = as_feature_matrix(features, name="features")
    y = x if other is None else as_feature_matrix(other, name="other")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"feature dimensions do not match: {x.shape[0]} and {y.shape[0]}"
        )

    if spec is None:
        resolved = KernelSpec("linear")
    elif isinstance(spec, dict):
        resolved = KernelSpec(
            kernel_type=str(spec.get("kernel_type", spec.get("type", "linear"))),
            gamma=spec.get("gamma"),
        )
    else:
        resolved = spec

    kind = resolved.kernel_type.lower().replace("-", "_")
    if kind in {"linear", "lin"}:
        kernel = x.T @ y
        resolved = KernelSpec("linear", None)
    elif kind in {"gaussian", "rbf"}:
        gamma = resolved.gamma
        if gamma is None:
            gamma = _default_rbf_gamma(x if other is None else np.concatenate([x, y], axis=1))
        if gamma <= 0:
            raise ValueError("RBF gamma must be positive")
        kernel = np.exp(-float(gamma) * _squared_distances(x, y))
        resolved = KernelSpec("gaussian", float(gamma))
    else:
        raise ValueError(f"unsupported kernel type: {resolved.kernel_type!r}")

    if other is None:
        return kernel, resolved
    return kernel


def _as_counts(counts: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=int).reshape(-1)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("bag counts must be non-empty positive integers")
    return values


def shrink_kernel(
    kernel: np.ndarray,
    positive_counts: Sequence[int] | np.ndarray,
    row_mode: int = 1,
    col_mode: int = 1,
) -> np.ndarray:
    """Collapse the positive-instance rows/columns into positive bags.

    ``shrink_kernel`` is referenced by the original ``main_sMIL_PI`` files but
    absent from the archive.  The intended operation is block averaging: every
    positive bag becomes one sample while negative instances remain individual
    samples.  ``row_mode`` and ``col_mode`` are retained for source-level API
    compatibility; mode ``1`` means averaging.
    """

    if row_mode != 1 or col_mode != 1:
        raise NotImplementedError("only the averaging mode (1, 1) is supported")
    matrix = np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be a square 2-D matrix")
    counts = _as_counts(positive_counts)
    pos_total = int(counts.sum())
    if pos_total > matrix.shape[0]:
        raise ValueError("positive bag counts exceed kernel size")
    neg_count = matrix.shape[0] - pos_total
    output_size = counts.size + neg_count
    result = np.empty((output_size, output_size), dtype=float)

    starts = np.cumsum(np.r_[0, counts[:-1]])
    for i, (start_i, count_i) in enumerate(zip(starts, counts)):
        stop_i = start_i + count_i
        for j, (start_j, count_j) in enumerate(zip(starts, counts)):
            stop_j = start_j + count_j
            result[i, j] = matrix[start_i:stop_i, start_j:stop_j].mean()
        if neg_count:
            result[i, counts.size :] = matrix[start_i:stop_i, pos_total:].mean(axis=0)
            result[counts.size :, i] = matrix[pos_total:, start_i:stop_i].mean(axis=1)

    if neg_count:
        result[counts.size :, counts.size :] = matrix[pos_total:, pos_total:]
    return result


def shrink_row(
    matrix: np.ndarray,
    positive_counts: Sequence[int] | np.ndarray,
    row_mode: int = 1,
) -> np.ndarray:
    """Collapse positive-instance rows into positive-bag rows by averaging."""

    if row_mode != 1:
        raise NotImplementedError("only the averaging mode (1) is supported")
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2-D")
    counts = _as_counts(positive_counts)
    pos_total = int(counts.sum())
    if pos_total > values.shape[0]:
        raise ValueError("positive bag counts exceed matrix rows")
    neg_count = values.shape[0] - pos_total
    result = np.empty((counts.size + neg_count, values.shape[1]), dtype=float)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    for i, (start, count) in enumerate(zip(starts, counts)):
        result[i] = values[start : start + count].mean(axis=0)
    if neg_count:
        result[counts.size :] = values[pos_total:]
    return result


def average_precision(labels: Iterable[float], scores: Iterable[float]) -> float:
    """Threshold-aggregated AP, matching sklearn including tied scores."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("labels and scores must be finite")
    if not y.size or not np.any(y > 0):
        return 0.0
    return float(average_precision_score(y > 0, s))
