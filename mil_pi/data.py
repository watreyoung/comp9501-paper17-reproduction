"""MATLAB ``.mat`` data loading for the original MIL-PI demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


@dataclass
class BagDataset:
    """Feature bags and held-out samples in the archive's layout.

    Each bag is a ``(feature_dim, instances_in_bag)`` matrix, i.e. instances are
    columns exactly as they are concatenated by the MATLAB demos.
    """

    positive_visual_bags: list[np.ndarray]
    negative_visual_bags: list[np.ndarray]
    positive_text_bags: list[np.ndarray]
    negative_text_bags: list[np.ndarray]
    test_features: np.ndarray
    test_labels: np.ndarray

    @property
    def bag_size(self) -> int:
        sizes = {bag.shape[1] for bag in self.positive_visual_bags + self.negative_visual_bags}
        if len(sizes) != 1:
            raise ValueError(f"all bags must have the same size for this implementation: {sizes}")
        return sizes.pop()

    @property
    def bag_count(self) -> int:
        if len(self.positive_visual_bags) != len(self.negative_visual_bags):
            raise ValueError("positive and negative bag counts must match")
        return len(self.positive_visual_bags)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        if name in value:
            return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    raise KeyError(f"MATLAB struct has no field {name!r}")


def _flatten_struct_array(value: Any) -> list[Any]:
    """Flatten scipy's several representations of MATLAB struct arrays."""

    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_struct_array(item))
        return result
    if isinstance(value, np.ndarray):
        if value.dtype == object or value.dtype.names is not None:
            result = []
            for item in value.reshape(-1):
                result.extend(_flatten_struct_array(item))
            return result
        # A single numeric feature matrix is still one bag.
        return [value]
    return [value]


def _extract_bags(mat: dict[str, Any], variable: str) -> list[np.ndarray]:
    if variable not in mat:
        raise KeyError(f"missing variable {variable!r} in .mat file")
    raw = mat[variable]
    bags: list[np.ndarray] = []
    for item in _flatten_struct_array(raw):
        try:
            features = _field(item, "features")
        except KeyError:
            # Some scipy configurations represent a one-element struct as a
            # 0-D object array; unwrap once before retrying.
            if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
                features = _field(item.reshape(-1)[0], "features")
            else:
                raise
        array = np.asarray(features, dtype=float)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(
                f"{variable} contains a bag with non-2-D features: {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{variable} contains NaN or infinity")
        bags.append(array)
    if not bags:
        raise ValueError(f"{variable} is empty")
    return bags


def _load_single_variable(path: Path, variable: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = loadmat(path, squeeze_me=False, struct_as_record=False)
    if variable not in loaded:
        raise KeyError(f"missing variable {variable!r} in {path}")
    return loaded[variable]


def _coerce_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def load_mat_dataset(
    data_home: str | Path,
    *,
    max_bag_num: int | None = 60,
    text_dim: int = 2000,
) -> BagDataset:
    """Load the four files required by ``README.txt`` in the archive.

    Required files and variables:

    * ``visual_bags.mat``: ``pos_bags``, ``neg_bags`` with ``features`` fields;
    * ``text_bags.mat``: the same structure;
    * ``test_features.mat``: ``test_features``;
    * ``test_labels.mat``: ``test_labels``.
    """

    root = Path(data_home).expanduser().resolve()
    visual = loadmat(root / "visual_bags.mat", squeeze_me=False, struct_as_record=False)
    text = loadmat(root / "text_bags.mat", squeeze_me=False, struct_as_record=False)
    test_features_mat = loadmat(root / "test_features.mat", squeeze_me=False, struct_as_record=False)
    test_labels_mat = loadmat(root / "test_labels.mat", squeeze_me=False, struct_as_record=False)

    positive_visual = _extract_bags(visual, "pos_bags")
    negative_visual = _extract_bags(visual, "neg_bags")
    positive_text = _extract_bags(text, "pos_bags")
    negative_text = _extract_bags(text, "neg_bags")

    if len(positive_visual) != len(negative_visual):
        raise ValueError("visual positive/negative bag counts differ")
    if len(positive_text) != len(negative_text):
        raise ValueError("text positive/negative bag counts differ")
    if len(positive_visual) != len(positive_text) or len(negative_visual) != len(negative_text):
        raise ValueError("visual and text bag counts differ")

    for sign, visuals, texts in [("positive", positive_visual, positive_text),
                                  ("negative", negative_visual, negative_text)]:
        for index, (v, t) in enumerate(zip(visuals, texts)):
            if v.shape[1] != t.shape[1]:
                raise ValueError(f"{sign} bag {index}: visual/text instance counts differ")
    text_dims = {b.shape[0] for b in positive_text + negative_text}
    if len(text_dims) != 1:
        raise ValueError("text feature dimensions are inconsistent")

    limit = len(positive_visual) if max_bag_num is None else min(len(positive_visual), max_bag_num)
    if limit <= 0:
        raise ValueError("no bags selected")
    positive_visual = positive_visual[:limit]
    negative_visual = negative_visual[:limit]
    positive_text = positive_text[:limit]
    negative_text = negative_text[:limit]

    visual_dim = positive_visual[0].shape[0]
    for name, bags in (
        ("positive_visual", positive_visual),
        ("negative_visual", negative_visual),
    ):
        for bag in bags:
            if bag.shape[0] != visual_dim:
                raise ValueError(f"{name} feature dimensions are inconsistent")

    text_dim_available = positive_text[0].shape[0]
    if text_dim <= 0:
        raise ValueError("text_dim must be positive")
    use_text_dim = min(text_dim, text_dim_available)
    positive_text = [bag[:use_text_dim] for bag in positive_text]
    negative_text = [bag[:use_text_dim] for bag in negative_text]

    test_features = _coerce_matrix(test_features_mat.get("test_features"), name="test_features")
    test_labels = np.asarray(test_labels_mat.get("test_labels"), dtype=float).reshape(-1)
    if not np.isfinite(test_labels).all():
        raise ValueError("test_labels contains NaN or infinity")
    if test_features.shape[0] != visual_dim and test_features.shape[1] == visual_dim:
        test_features = test_features.T
    if test_features.shape[0] != visual_dim:
        raise ValueError(
            "test_features has the wrong feature dimension: "
            f"expected {visual_dim}, got {test_features.shape}"
        )
    if test_features.shape[1] != test_labels.size:
        raise ValueError(
            f"test feature count {test_features.shape[1]} does not match "
            f"test label count {test_labels.size}"
        )

    return BagDataset(
        positive_visual_bags=positive_visual,
        negative_visual_bags=negative_visual,
        positive_text_bags=positive_text,
        negative_text_bags=negative_text,
        test_features=test_features,
        test_labels=test_labels,
    )
