"""Generate a tiny data directory with the same variables as the assignment.

Usage:
    python examples/make_synthetic_mat_data.py /tmp/mil_pi_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=5)
    parser.add_argument("--visual-dim", type=int, default=8)
    parser.add_argument("--text-dim", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=12)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    args.output.mkdir(parents=True, exist_ok=True)
    pos_visual = np.empty((args.bags,), dtype=[("features", "O")])
    neg_visual = np.empty((args.bags,), dtype=[("features", "O")])
    pos_text = np.empty((args.bags,), dtype=[("features", "O")])
    neg_text = np.empty((args.bags,), dtype=[("features", "O")])
    for index in range(args.bags):
        pos_visual[index]["features"] = rng.normal(1.2, 0.8, (args.visual_dim, args.bag_size))
        neg_visual[index]["features"] = rng.normal(-0.2, 0.8, (args.visual_dim, args.bag_size))
        pos_text[index]["features"] = rng.normal(1.0, 0.5, (args.text_dim, args.bag_size))
        neg_text[index]["features"] = rng.normal(-0.1, 0.5, (args.text_dim, args.bag_size))
    test_features = np.concatenate(
        [
            rng.normal(1.2, 0.8, (args.visual_dim, args.test_size // 2)),
            rng.normal(-0.2, 0.8, (args.visual_dim, args.test_size - args.test_size // 2)),
        ],
        axis=1,
    )
    test_labels = np.concatenate(
        [np.ones(args.test_size // 2), -np.ones(args.test_size - args.test_size // 2)]
    )
    savemat(args.output / "visual_bags.mat", {"pos_bags": pos_visual, "neg_bags": neg_visual})
    savemat(args.output / "text_bags.mat", {"pos_bags": pos_text, "neg_bags": neg_text})
    savemat(args.output / "test_features.mat", {"test_features": test_features})
    savemat(args.output / "test_labels.mat", {"test_labels": test_labels})
    print(f"wrote synthetic assignment data to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
