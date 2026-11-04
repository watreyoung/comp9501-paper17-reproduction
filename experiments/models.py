"""Shared kernels, original six methods, and explicit visual-only baselines."""
from __future__ import annotations

import math
import time
import numpy as np
from mil_pi.algorithms import (KernelModel, s_mil_pi, s_mil_pi_da, mi_svm_pi,
                              mi_svm_pi_da, mil_cpb_pi, mil_cpb_pi_da)
from mil_pi.demo import default_parameters
from mil_pi.kernels import KernelSpec, get_kernel, l2_normalize_columns, shrink_kernel
from mil_pi.qp import qp_session, solve_qp

METHODS = ["SVM", "sMIL", "sMIL-PI", "MI-SVM-PI", "MIL-CPB-PI",
           "sMIL-PI-DA", "MI-SVM-PI-DA", "MIL-CPB-PI-DA", "sMIL-PI-shuffled"]


def prepare_problem(data, *, rbf_gamma=None):
    visual = l2_normalize_columns(np.concatenate(data.positive_visual_bags + data.negative_visual_bags, axis=1))
    text = np.concatenate(data.positive_text_bags + data.negative_text_bags, axis=1)
    target = l2_normalize_columns(data.test_features)
    kernel, spec = get_kernel(visual, spec=KernelSpec("gaussian", rbf_gamma))
    target_kernel = get_kernel(target, visual, spec)
    text_kernel, _ = get_kernel(text, spec=KernelSpec("linear"))
    n_bags, bag_size = len(data.positive_visual_bags), data.bag_size
    n_pos = n_bags * bag_size
    mask = np.r_[np.ones(n_pos), np.zeros(n_pos)]
    text_kernel *= np.outer(mask, mask)
    return dict(K=kernel, tK=text_kernel, test_kernel=target_kernel, Kst=target_kernel.T,
                n_bags=n_bags, bag_size=bag_size,
                bags=[np.arange(i*bag_size, (i+1)*bag_size) for i in range(2*n_bags)],
                bag_labels=np.r_[np.ones(n_bags), -np.ones(n_bags)],
                kernel_gamma=spec.gamma)


def _visual_svm(K, n_positive, *, margins=None, C=100, max_iter=100000):
    n = K.shape[0]
    y = np.r_[np.ones(n_positive), -np.ones(n-n_positive)]
    p = np.ones(n) if margins is None else margins
    upper = C * np.r_[np.ones(n_positive)/n_positive, np.ones(n-n_positive)/(n-n_positive)]
    result = solve_qp(K*np.outer(y, y), -p, A_eq=y[None], b_eq=np.zeros(1),
                      lb=np.zeros(n), ub=upper, max_iter=max_iter, name="visual SVM dual")
    alpha = result.x
    coef = alpha*y
    g = K@coef
    tolerance = 1e-6
    free = (alpha > tolerance) & (alpha < upper-tolerance)
    if np.any(free):
        bias = float(np.mean((y*p-g)[free]))
    else:
        lower = (np.r_[p[:n_positive]-g[:n_positive], -p[n_positive:]-g[n_positive:]])[
            ((y > 0) & (alpha <= tolerance)) | ((y < 0) & (alpha >= upper-tolerance))]
        higher = (y*p-g)[((y > 0) & (alpha >= upper-tolerance)) | ((y < 0) & (alpha <= tolerance))]
        low = np.max(lower, initial=-np.inf)
        high = np.min(higher, initial=np.inf)
        bias = float((low+high)/2) if np.isfinite(low+high) else 0.0
    return KernelModel(coef=coef, bias=bias, alpha=alpha, objective=result.objective,
                       solver_message=result.message)


def fit_predict(method, problem, *, parameters=None, backend="osqp", seed=0):
    if method not in METHODS:
        raise ValueError(method)
    b, m = problem["n_bags"], problem["bag_size"]
    k, tk = problem["K"], problem["tK"]
    base = "sMIL-PI" if method == "sMIL-PI-shuffled" else method
    p = {} if base in {"SVM", "sMIL"} else default_parameters(base, m)
    p.update(rho=0.6, pos_ratio=0.6, max_iter=100, mkl_max_iter=100, solver_max_iter=100000)
    p.update(parameters or {})
    p["pos_num"] = math.ceil(p["pos_ratio"]*m)
    started = time.perf_counter()
    if method.endswith("-shuffled"):
        order = np.r_[np.random.default_rng(seed).permutation(b*m), np.arange(b*m, 2*b*m)]
        tk = tk[np.ix_(order, order)]
    with qp_session(backend) as trace:
        if base == "SVM":
            model = _visual_svm(k, b*m, C=p.get("baseline_C", 100), max_iter=p["solver_max_iter"])
        elif base == "sMIL":
            small = shrink_kernel(k, [m]*b)
            margins = np.r_[np.full(b, 2*p["rho"]-1), np.ones(b*m)]
            model = _visual_svm(small, b, margins=margins, C=p.get("baseline_C", 100), max_iter=p["solver_max_iter"])
            model.coef = np.r_[np.repeat(model.coef[:b]/m, m), model.coef[b:]]
        elif base == "sMIL-PI":
            model = s_mil_pi(k, tk, [m]*b, [m]*b, p)
        elif base == "MI-SVM-PI":
            model = mi_svm_pi(k, tk, [m]*b, [m]*b, p)
        elif base == "MIL-CPB-PI":
            model = mil_cpb_pi(k, tk, problem["bags"], problem["bag_labels"], p)
        elif base == "sMIL-PI-DA":
            model = s_mil_pi_da(k, tk, k, problem["Kst"], [m]*b, [m]*b, p)
        elif base == "MI-SVM-PI-DA":
            model = mi_svm_pi_da(k, tk, k, problem["Kst"], [m]*b, [m]*b, p)
        elif base == "MIL-CPB-PI-DA":
            model = mil_cpb_pi_da(k, tk, k, problem["Kst"], problem["bags"], problem["bag_labels"], p)
    scores = model.decision_function(problem["test_kernel"])
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite held-out predictions")
    diagnostics = dict(seconds=time.perf_counter()-started, qp_solves=len(trace),
        max_constraint_violation=max((r["constraint_violation"] or 0 for r in trace), default=0),
        max_primal_residual=max((r["primal_residual"] or 0 for r in trace), default=0),
        max_dual_residual=max((r["dual_residual"] or 0 for r in trace), default=0),
        solver_iterations=sum(r["iterations"] for r in trace),
        objective=model.objective, outer_objective_history=model.objective_history,
        outer_iterations=model.outer_iterations, stop_reason=model.stop_reason,
        label_changes=model.label_changes,
        solver_message=model.solver_message, parameters=p,
        qp_diagnostics=trace)
    return scores, diagnostics
