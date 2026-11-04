"""Deterministic weak-label bags; instance truth stays outside model inputs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.io import savemat
from mil_pi.data import BagDataset


def make_bag_indices(labels, train_indices, class_id, *, seed, noise, n_bags=10, bag_size=5):
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    n_bad = int(round(noise * bag_size))
    if not np.isclose(n_bad / bag_size, noise) or not 0 <= n_bad < bag_size:
        raise ValueError("Noise must be an exact number of contaminants per bag")
    rng = np.random.default_rng(seed)
    positive = rng.permutation(train_indices[labels[train_indices] == class_id])
    negative = rng.permutation(train_indices[labels[train_indices] != class_id])
    needed = n_bags * bag_size
    if positive.size < needed or negative.size < needed + n_bags*n_bad:
        raise ValueError("Not enough distinct source videos to construct bags")
    positive = positive[:needed].reshape(n_bags, bag_size).copy()
    negative_bags = negative[:needed].reshape(n_bags, bag_size).copy()
    if n_bad:
        positive[:, -n_bad:] = negative[needed:needed+n_bags*n_bad].reshape(n_bags, n_bad)
    for bag in positive:
        rng.shuffle(bag)
    return positive, negative_bags


def make_task(cache, class_id, *, seed, noise, n_bags=10, bag_size=5, evaluation_split="test"):
    train = np.flatnonzero(cache["splits"] == "train")
    evaluation = np.flatnonzero(cache["splits"] == evaluation_split)
    if not evaluation.size:
        raise ValueError(f"Empty evaluation split: {evaluation_split}")
    positive, negative = make_bag_indices(cache["labels"], train, class_id,
        seed=seed, noise=noise, n_bags=n_bags, bag_size=bag_size)
    dataset = BagDataset(
        positive_visual_bags=[cache["visual"][b].T for b in positive],
        negative_visual_bags=[cache["visual"][b].T for b in negative],
        positive_text_bags=[cache["text"][b].T for b in positive],
        negative_text_bags=[cache["text"][b].T for b in negative],
        test_features=cache["visual"][evaluation].T,
        test_labels=np.where(cache["labels"][evaluation] == class_id, 1, -1))
    manifest = {
        "class_id": int(class_id), "seed": int(seed), "noise": float(noise),
        "positive_video_ids": cache["video_ids"][positive].tolist(),
        "negative_video_ids": cache["video_ids"][negative].tolist(),
        "positive_bag_true_positive_counts": (cache["labels"][positive] == class_id).sum(axis=1).tolist(),
        "evaluation_video_ids": cache["video_ids"][evaluation].tolist(),
        "evaluation_split": evaluation_split,
        "training_instance_truth_used_by_model": False,
    }
    return dataset, manifest


def export_mat_task(dataset: BagDataset, manifest, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def structs(bags):
        array = np.empty((1, len(bags)), dtype=[("features", "O")])
        for i, b in enumerate(bags):
            array[0, i]["features"] = b
        return array
    for name, pos, neg in [
        ("visual_bags", dataset.positive_visual_bags, dataset.negative_visual_bags),
        ("text_bags", dataset.positive_text_bags, dataset.negative_text_bags)]:
        savemat(root / f"{name}.mat", {"pos_bags": structs(pos), "neg_bags": structs(neg)}, do_compression=True)
    savemat(root / "test_features.mat", {"test_features": dataset.test_features}, do_compression=True)
    savemat(root / "test_labels.mat", {"test_labels": dataset.test_labels}, do_compression=True)
    (root / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
