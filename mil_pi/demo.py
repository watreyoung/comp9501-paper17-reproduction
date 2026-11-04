"""Command-line runner for the six converted MIL-PI demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .algorithms import (
    KernelModel,
    mil_cpb_pi,
    mil_cpb_pi_da,
    mi_svm_pi,
    mi_svm_pi_da,
    s_mil_pi,
    s_mil_pi_da,
)
from .data import BagDataset, load_mat_dataset
from .kernels import KernelSpec, average_precision, get_kernel, l2_normalize_columns


METHODS = (
    "sMIL-PI",
    "sMIL-PI-DA",
    "MI-SVM-PI",
    "MI-SVM-PI-DA",
    "MIL-CPB-PI",
    "MIL-CPB-PI-DA",
)


def _prepare_problem(data: BagDataset, *, rbf_gamma: float | None = None):
    visual = np.concatenate(data.positive_visual_bags + data.negative_visual_bags, axis=1)
    text = np.concatenate(data.positive_text_bags + data.negative_text_bags, axis=1)
    visual = l2_normalize_columns(visual)
    test_features = l2_normalize_columns(data.test_features)
    visual_spec = KernelSpec("gaussian", rbf_gamma)
    K, visual_spec = get_kernel(visual, spec=visual_spec)
    test_kernel = get_kernel(test_features, visual, visual_spec)
    text_kernel, _ = get_kernel(text, spec=KernelSpec("linear"))
    instance_labels = np.concatenate(
        [np.ones(visual.shape[1] // 2), np.zeros(visual.shape[1] // 2)]
    )
    tK = text_kernel * np.outer(instance_labels, instance_labels)
    Kst = get_kernel(visual, test_features, visual_spec)
    bag_size = data.bag_size
    pos_count = len(data.positive_visual_bags)
    bags = []
    cursor = 0
    for _ in range(pos_count):
        bags.append(np.arange(cursor, cursor + bag_size, dtype=int))
        cursor += bag_size
    for _ in range(pos_count):
        bags.append(np.arange(cursor, cursor + bag_size, dtype=int))
        cursor += bag_size
    bag_labels = np.concatenate([np.ones(pos_count), -np.ones(pos_count)])
    return K, tK, test_kernel, Kst, bags, bag_labels, bag_size


def default_parameters(method: str, bag_size: int) -> dict:
    """Return the numeric defaults from the corresponding MATLAB demo."""

    if method == "sMIL-PI":
        return dict(bag_size=bag_size, tdim=2000, svm_C=100.0, gamma=100.0, rho=0.6)
    if method == "sMIL-PI-DA":
        return dict(
            bag_size=bag_size,
            tdim=2000,
            svm_C1=100.0,
            svm_C2=1e5,
            gamma1=100.0,
            gamma2=1e6,
            rho=0.6,
            B=1e10,
            epsilon=0.0,
        )
    if method == "MI-SVM-PI":
        return dict(
            bag_size=bag_size,
            tdim=2000,
            svm_C=0.01,
            gamma=10.0,
            pos_ratio=0.8,
        )
    if method == "MI-SVM-PI-DA":
        return dict(
            bag_size=bag_size,
            tdim=2000,
            svm_C1=0.01,
            svm_C2=1e-5,
            gamma1=10.0,
            gamma2=100.0,
            B=1e10,
            epsilon=0.0,
            pos_ratio=0.2,
            max_iter=3,
        )
    if method == "MIL-CPB-PI":
        return dict(
            bag_size=bag_size,
            tdim=2000,
            svm_C=0.01,
            svm_C_star=1.0,
            gamma=10.0,
            pos_ratio=0.6,
            max_iter=100,
        )
    if method == "MIL-CPB-PI-DA":
        return dict(
            bag_size=bag_size,
            tdim=2000,
            svm_C1=0.01,
            svm_C2=1e-5,
            gamma1=10.0,
            gamma2=100.0,
            B=1e10,
            epsilon=0.0,
            pos_ratio=0.2,
            max_iter=100,
        )
    raise ValueError(f"unknown method: {method}")


def run_method(
    method: str,
    data: BagDataset,
    *,
    rbf_gamma: float | None = None,
    overrides: dict | None = None,
) -> tuple[KernelModel, np.ndarray, float]:
    K, tK, test_kernel, Kst, bags, bag_labels, bag_size = _prepare_problem(
        data, rbf_gamma=rbf_gamma
    )
    param = default_parameters(method, bag_size)
    if overrides:
        param.update(overrides)
    positive_bags = [bag_size] * len(data.positive_visual_bags)
    negative_bags = [bag_size] * len(data.negative_visual_bags)

    if method == "sMIL-PI":
        model = s_mil_pi(K, tK, positive_bags, negative_bags, param)
        scores = model.decision_function(test_kernel)
    elif method == "sMIL-PI-DA":
        model = s_mil_pi_da(K, tK, K, Kst, positive_bags, negative_bags, param)
        scores = model.decision_function(test_kernel)
    elif method == "MI-SVM-PI":
        model = mi_svm_pi(K, tK, positive_bags, negative_bags, param)
        scores = model.decision_function(test_kernel)
    elif method == "MI-SVM-PI-DA":
        model = mi_svm_pi_da(K, tK, K, Kst, positive_bags, negative_bags, param)
        scores = model.decision_function(test_kernel)
    elif method == "MIL-CPB-PI":
        model = mil_cpb_pi(K, tK, bags, bag_labels, param)
        scores = model.decision_function(test_kernel)
    elif method == "MIL-CPB-PI-DA":
        model = mil_cpb_pi_da(K, tK, K, Kst, bags, bag_labels, param)
        scores = model.decision_function(test_kernel)
    else:
        raise ValueError(method)
    return model, scores, average_precision(data.test_labels, scores)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Python conversion of the six MATLAB MIL-PI demos."
    )
    parser.add_argument("--data-home", required=True, help="directory containing the four .mat files")
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--max-bag-num", type=int, default=60)
    parser.add_argument("--text-dim", type=int, default=2000)
    parser.add_argument(
        "--rbf-gamma",
        type=float,
        default=None,
        help="explicit visual RBF gamma; default uses the median-distance heuristic",
    )
    parser.add_argument(
        "--solver-max-iter",
        type=int,
        default=2000,
        help="maximum SLSQP iterations per QP",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mat_dataset(args.data_home, max_bag_num=args.max_bag_num, text_dim=args.text_dim)
    methods = METHODS if args.method == "all" else (args.method,)
    results = {}
    for method in methods:
        print(f"[{method}] loading/training...")
        model, scores, ap = run_method(
            method,
            data,
            rbf_gamma=args.rbf_gamma,
            overrides={"solver_max_iter": args.solver_max_iter},
        )
        results[method] = {
            "average_precision": ap,
            "support_vectors": int(model.support_indices.size),
            "objective": model.objective,
            "objective_iterations": max(1, len(model.objective_history)),
            "outer_iterations": model.outer_iterations,
            "stop_reason": model.stop_reason,
            "label_changes": model.label_changes,
            "solver_message": model.solver_message,
        }
        print(f"[{method}] AP = {ap:.6f}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
