"""Python implementations of the MIL-PI algorithms in the supplied MATLAB files.

The public functions mirror the six demo entry points:

* :func:`s_mil_pi` and :func:`s_mil_pi_da`;
* :func:`mi_svm_pi` and :func:`mi_svm_pi_da`;
* :func:`mil_cpb_pi` and :func:`mil_cpb_pi_da`.

All kernels use samples as rows at the public prediction boundary, while the
low-level helpers keep the source code's MATLAB convention where kernels are
indexed by sample and feature matrices are represented by columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .kernels import shrink_kernel, shrink_row
from .qp import QPError, QPResult, solve_qp


_EPS = 1e-10


@dataclass
class KernelModel:
    """A kernel decision function plus source-compatible training fields."""

    coef: np.ndarray
    bias: float = 0.0
    support_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    sv_coef: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    rho: float = 0.0
    alpha: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    d: np.ndarray | None = None
    y_set: np.ndarray | None = None
    effective_labels: np.ndarray | None = None
    objective: float | None = None
    objective_history: list[float] = field(default_factory=list)
    outer_iterations: int = 0
    stop_reason: str = "not_recorded"
    label_changes: list[int] = field(default_factory=list)
    solver_message: str = ""

    @property
    def SVs(self) -> np.ndarray:
        """MATLAB/libsvm-style one-based support indices are exposed as zero-based."""

        return self.support_indices

    def decision_function(self, kernel_to_model: np.ndarray) -> np.ndarray:
        matrix = np.asarray(kernel_to_model, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("kernel_to_model must be a 2-D matrix")
        if self.support_indices.size:
            scores = matrix[:, self.support_indices] @ self.sv_coef
        else:
            scores = matrix @ self.coef
        return np.asarray(scores, dtype=float).reshape(-1) + float(self.bias)


def _positive_negative_counts(
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
) -> tuple[np.ndarray, int]:
    def size_of(bag: int | object) -> int:
        if isinstance(bag, (int, np.integer)):
            value = int(bag)
        elif hasattr(bag, "bag_size"):
            value = int(getattr(bag, "bag_size"))
        elif hasattr(bag, "shape"):
            value = int(np.asarray(getattr(bag, "shape"))[1])
        else:
            raise TypeError("bags must be sizes, arrays, or objects with bag_size")
        if value <= 0:
            raise ValueError("bag sizes must be positive")
        return value

    positive_counts = np.asarray([size_of(bag) for bag in positive_bags], dtype=int)
    negative_count = int(sum(size_of(bag) for bag in negative_bags))
    if positive_counts.size == 0 or negative_count == 0:
        raise ValueError("both positive and negative bags are required")
    return positive_counts, negative_count


def _expand_positive_coefficients(
    bag_coefficients: np.ndarray,
    positive_counts: np.ndarray,
    negative_count: int,
) -> np.ndarray:
    bag_coefficients = np.asarray(bag_coefficients, dtype=float).reshape(-1)
    pos_bags = []
    for index, count in enumerate(positive_counts):
        pos_bags.append(np.full(int(count), bag_coefficients[index] / float(count)))
    negative = bag_coefficients[positive_counts.size :]
    if negative.size != negative_count:
        raise ValueError("negative coefficient count does not match bag sizes")
    return np.concatenate([*pos_bags, negative])


def _initial_feasible_smil(
    n: int,
    positive_count: int,
    c_positive: np.ndarray,
    *,
    da: bool = False,
    c2: float | None = None,
) -> np.ndarray:
    if not da:
        x0 = np.zeros(2 * n, dtype=float)
        x0[n : n + positive_count] = c_positive
        return x0
    assert c2 is not None
    x0 = np.zeros(3 * n, dtype=float)
    x0[n : n + positive_count] = c_positive
    x0[2 * n :] = c2
    return x0


def _bias_from_decision_margin(
    dec_2: np.ndarray,
    alpha: np.ndarray,
    y: np.ndarray,
) -> float:
    """Recover the bias using the same support/non-support rule as MATLAB."""

    interior = alpha > _EPS
    if np.any(interior):
        return float(np.mean(dec_2[interior]))
    positive = dec_2[y > 0]
    negative = dec_2[y < 0]
    if positive.size and negative.size:
        return float((np.max(positive) + np.min(negative)) / 2.0)
    return float(np.mean(dec_2))


def _solve_smil_pi_qp(
    K: np.ndarray,
    tK: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    svm_C: np.ndarray,
    gamma: float,
    pos_bn: int,
    neg_fn: int,
    *,
    solver_max_iter: int = 2000,
) -> tuple[np.ndarray, QPResult]:
    n = tK.shape[0]
    if pos_bn + neg_fn != n:
        raise ValueError("positive bag and negative instance counts do not match kernel size")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    tK = np.asarray(tK, dtype=float)
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    c = np.asarray(svm_C, dtype=float).reshape(-1)
    if any(value.size != n for value in (y, p, c)):
        raise ValueError("sMIL QP vectors must have length n")

    top = np.concatenate([tK, tK], axis=1)
    bottom = np.concatenate([tK, tK], axis=1)
    hessian = np.concatenate([top, bottom], axis=0) / gamma
    hessian[:n, :n] += K * np.outer(y, y)
    hessian += 1e-6 * np.eye(2 * n)
    tmp = -(tK @ c) / gamma
    linear = np.concatenate([-p + tmp, tmp])

    equality = np.zeros((2, 2 * n), dtype=float)
    equality[0, :n] = y
    equality[1, :pos_bn] = 1.0
    equality[1, n : n + pos_bn] = 1.0
    equality_rhs = np.array([0.0, float(np.sum(c[:pos_bn]))])

    lower = np.zeros(2 * n, dtype=float)
    upper = np.full(2 * n, np.inf, dtype=float)
    upper[pos_bn:n] = 1.0
    x0 = _initial_feasible_smil(n, pos_bn, c[:pos_bn])
    result = solve_qp(
        hessian,
        linear,
        A_eq=equality,
        b_eq=equality_rhs,
        lb=lower,
        ub=upper,
        x0=x0,
        max_iter=solver_max_iter,
        name="sMIL-PI QP",
    )
    return result.x, result


def train_smil_pi(
    K: np.ndarray,
    tK: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    svm_C: np.ndarray,
    gamma: float,
    pos_bn: int,
    neg_fn: int,
    *,
    solver_max_iter: int = 2000,
) -> KernelModel:
    """Low-level equivalent of ``train_sMIL_PI.m``."""

    x, result = _solve_smil_pi_qp(
        K, tK, y, p, svm_C, gamma, pos_bn, neg_fn, solver_max_iter=solver_max_iter
    )
    n = tK.shape[0]
    alpha = x[:n]
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    c = np.asarray(svm_C, dtype=float).reshape(-1)
    zeta = x[n : 2 * n]
    tilde_dec = (np.asarray(tK, dtype=float) @ (alpha + zeta - c)) / float(gamma)
    tilde_idx = zeta > _EPS
    tilde_b = float(np.mean(-tilde_dec[tilde_idx])) if np.any(tilde_idx) else float(np.max(-tilde_dec))
    dec = np.asarray(K, dtype=float) @ (alpha * y)
    dec_2 = -dec + y * (1.0 - tilde_dec - tilde_b)
    b = _bias_from_decision_margin(dec_2, alpha, y)
    coef = alpha * y
    return KernelModel(
        coef=coef,
        bias=b,
        support_indices=np.flatnonzero(alpha > _EPS),
        sv_coef=coef[alpha > _EPS],
        rho=-b,
        alpha=alpha,
        x=x,
        y=y,
        objective=result.objective,
        solver_message=result.message,
    )


def _solve_smil_pi_da_qp(
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    svm_C1: np.ndarray,
    svm_C2: np.ndarray,
    gamma1: float,
    gamma2: float,
    B: float,
    epsilon: float,
    pos_bn: int,
    neg_fn: int,
    *,
    solver_max_iter: int = 3000,
) -> tuple[np.ndarray, QPResult]:
    n = tK.shape[0]
    if pos_bn + neg_fn != n or np.asarray(Kst).shape[0] != n:
        raise ValueError("sMIL-PI-DA dimensions do not match")
    if gamma1 <= 0 or gamma2 <= 0 or B <= 0:
        raise ValueError("gamma1, gamma2, and B must be positive")
    c1 = np.asarray(svm_C1, dtype=float).reshape(-1)
    c2 = np.asarray(svm_C2, dtype=float).reshape(-1)
    if c1.size != n or c2.size != n:
        raise ValueError("C vectors must have length n")
    # The MATLAB code treats svm_C2 as a vector in the first sMIL DA solver,
    # but the supplied demos pass a class-weight vector.  The constraint uses
    # each element consistently, while the objective uses elementwise products.
    hessian = np.block(
        [
            [K * np.outer(y, y) + tK / gamma1 + Kss / gamma2, tK / gamma1, Kss / gamma2],
            [tK / gamma1, tK / gamma1, np.zeros_like(K)],
            [Kss / gamma2, np.zeros_like(K), Kss / gamma2],
        ]
    )
    hessian += 1e-6 * np.eye(3 * n)
    ns, nt = np.asarray(Kst).shape
    if nt <= 0:
        raise ValueError("Kst must have at least one target sample")
    ratio = ns / float(nt)
    sum_kst = np.sum(Kst, axis=1)
    f1 = -p - (tK @ c1) / gamma1 - ratio * sum_kst * c2 / gamma2
    f2 = -(tK @ c1) / gamma1
    f3 = -ratio * sum_kst * c2 / gamma2
    linear = np.concatenate([f1, f2, f3])

    inv_c2 = 1.0 / c2
    inequality = np.vstack(
        [
            np.concatenate([-inv_c2, np.zeros(n), -inv_c2]),
            np.concatenate([inv_c2, np.zeros(n), inv_c2]),
            np.concatenate([inv_c2, np.zeros(n), inv_c2]),
        ]
    )
    inequality_rhs = np.array([ns * epsilon - n, n + ns * epsilon, B * n], dtype=float)
    equality = np.zeros((2, 3 * n), dtype=float)
    equality[0, :n] = y
    equality[1, :pos_bn] = 1.0
    equality[1, n : n + pos_bn] = 1.0
    equality_rhs = np.array([0.0, float(np.sum(c1[:pos_bn]))])
    lower = np.zeros(3 * n, dtype=float)
    upper = np.full(3 * n, np.inf, dtype=float)
    upper[pos_bn:n] = 1.0
    x0 = _initial_feasible_smil(n, pos_bn, c1[:pos_bn], da=True, c2=float(np.mean(c2)))
    # For vector C2, setting every zeta2 element to its own C2 is the exact
    # feasible point; the helper uses a scalar to preserve a compact API.
    x0[2 * n :] = c2
    result = solve_qp(
        hessian,
        linear,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        lb=lower,
        ub=upper,
        x0=x0,
        max_iter=solver_max_iter,
        name="sMIL-PI-DA QP",
    )
    return result.x, result


def train_smil_pi_da(
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    svm_C1: np.ndarray,
    svm_C2: np.ndarray,
    gamma1: float,
    gamma2: float,
    B: float,
    epsilon: float,
    pos_bn: int,
    neg_fn: int,
    *,
    solver_max_iter: int = 3000,
) -> KernelModel:
    """Low-level equivalent of ``train_sMIL_PI_DA.m``."""

    x, result = _solve_smil_pi_da_qp(
        K,
        tK,
        Kss,
        Kst,
        y,
        p,
        svm_C1,
        svm_C2,
        gamma1,
        gamma2,
        B,
        epsilon,
        pos_bn,
        neg_fn,
        solver_max_iter=solver_max_iter,
    )
    n = tK.shape[0]
    alpha = x[:n]
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    c1 = np.asarray(svm_C1, dtype=float).reshape(-1)
    c2 = np.asarray(svm_C2, dtype=float).reshape(-1)
    zeta1 = x[n : 2 * n]
    zeta2 = x[2 * n : 3 * n]
    tilde_dec = (np.asarray(tK, dtype=float) @ (alpha + zeta1 - c1)) / float(gamma1)
    tilde_idx = zeta1 > _EPS
    tilde_b = float(np.mean(-tilde_dec[tilde_idx])) if np.any(tilde_idx) else float(np.max(-tilde_dec))
    ns, nt = np.asarray(Kst).shape
    beta2 = alpha + zeta2 - c2
    tilde_dec2 = (np.asarray(tK, dtype=float) @ beta2) / float(gamma2)
    tilde_dec2 += (c2 / float(gamma2)) * (np.sum(np.asarray(Kss), axis=1) * beta2)
    tilde_dec2 -= (ns / float(nt)) * (c2 / float(gamma2)) * (np.sum(np.asarray(Kst), axis=1) * beta2)
    tilde_idx2 = zeta2 > _EPS
    tilde_b2 = float(np.mean(-tilde_dec2[tilde_idx2])) if np.any(tilde_idx2) else float(np.max(-tilde_dec2))
    dec = np.asarray(K, dtype=float) @ (alpha * y)
    dec_2 = -dec + y * (1.0 - tilde_dec - tilde_b) + y * (1.0 - tilde_dec2 - tilde_b2)
    b = _bias_from_decision_margin(dec_2, alpha, y)
    coef = alpha * y
    return KernelModel(
        coef=coef,
        bias=b,
        support_indices=np.flatnonzero(alpha > _EPS),
        sv_coef=coef[alpha > _EPS],
        rho=-b,
        alpha=alpha,
        x=x,
        y=y,
        objective=result.objective,
        solver_message=result.message,
    )


def s_mil_pi(
    K: np.ndarray,
    tK: np.ndarray,
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
    param: dict,
) -> KernelModel:
    """Train standard sMIL with privileged information."""

    pos_counts, neg_fn = _positive_negative_counts(positive_bags, negative_bags)
    pos_bn = pos_counts.size
    K_small = shrink_kernel(K, pos_counts)
    tK_small = shrink_kernel(tK, pos_counts)
    labels = np.concatenate([np.ones(pos_bn), -np.ones(neg_fn)])
    svm_C = float(param["svm_C"]) * np.concatenate(
        [np.ones(pos_bn) / pos_bn, np.ones(neg_fn) / neg_fn]
    )
    rho = float(param.get("rho", 0.6))
    p = np.concatenate([np.full(pos_bn, 2.0 * rho - 1.0), np.ones(neg_fn)])
    bag_model = train_smil_pi(
        K_small,
        tK_small,
        labels,
        p,
        svm_C,
        float(param["gamma"]),
        pos_bn,
        neg_fn,
        solver_max_iter=int(param.get("solver_max_iter", 2000)),
    )
    coef = _expand_positive_coefficients(bag_model.coef, pos_counts, neg_fn)
    return KernelModel(
        coef=coef,
        bias=bag_model.bias,
        support_indices=np.flatnonzero(np.abs(coef) > _EPS),
        sv_coef=coef[np.abs(coef) > _EPS],
        rho=-bag_model.bias,
        alpha=coef,
        x=bag_model.x,
        y=np.concatenate([np.ones(pos_counts.sum()), -np.ones(neg_fn)]),
        objective=bag_model.objective,
        solver_message=bag_model.solver_message,
    )


def s_mil_pi_da(
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
    param: dict,
) -> KernelModel:
    """Train sMIL-PI with domain adaptation."""

    pos_counts, neg_fn = _positive_negative_counts(positive_bags, negative_bags)
    pos_bn = pos_counts.size
    K_small = shrink_kernel(K, pos_counts)
    tK_small = shrink_kernel(tK, pos_counts)
    Kss_small = shrink_kernel(Kss, pos_counts)
    Kst_small = shrink_row(Kst, pos_counts)
    labels = np.concatenate([np.ones(pos_bn), -np.ones(neg_fn)])
    c1 = float(param["svm_C1"]) * np.concatenate(
        [np.ones(pos_bn) / pos_bn, np.ones(neg_fn) / neg_fn]
    )
    c2 = float(param["svm_C2"]) * np.concatenate(
        [np.ones(pos_bn) / pos_bn, np.ones(neg_fn) / neg_fn]
    )
    rho = float(param.get("rho", 0.6))
    p = np.concatenate([np.full(pos_bn, 2.0 * rho - 1.0), np.ones(neg_fn)])
    bag_model = train_smil_pi_da(
        K_small,
        tK_small,
        Kss_small,
        Kst_small,
        labels,
        p,
        c1,
        c2,
        float(param["gamma1"]),
        float(param["gamma2"]),
        float(param.get("B", 1e10)),
        float(param.get("epsilon", 0.0)),
        pos_bn,
        neg_fn,
        solver_max_iter=int(param.get("solver_max_iter", 3000)),
    )
    coef = _expand_positive_coefficients(bag_model.coef, pos_counts, neg_fn)
    return KernelModel(
        coef=coef,
        bias=bag_model.bias,
        support_indices=np.flatnonzero(np.abs(coef) > _EPS),
        sv_coef=coef[np.abs(coef) > _EPS],
        rho=-bag_model.bias,
        alpha=coef,
        y=np.concatenate([np.ones(pos_counts.sum()), -np.ones(neg_fn)]),
        objective=bag_model.objective,
        solver_message=bag_model.solver_message,
    )


def _solve_mi_pi_qp(
    labels: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    svm_C: float,
    gamma: float,
    *,
    solver_max_iter: int = 2000,
) -> tuple[np.ndarray, QPResult, np.ndarray]:
    labels = np.asarray(labels, dtype=float).reshape(-1)
    n = labels.size
    if n % 2:
        raise ValueError("the supplied MATLAB formulation requires an even number of instances")
    y = np.where(labels == 1, 1.0, -1.0)
    hessian = np.block(
        [
            [K * np.outer(y, y) + tK / gamma, tK / gamma],
            [tK / gamma, tK / gamma],
        ]
    )
    hessian += 1e-6 * np.eye(2 * n)
    tmp = -(svm_C / gamma) * np.sum(tK, axis=1)
    linear = np.concatenate([-np.ones(n) + tmp, tmp])
    equality = np.zeros((2, 2 * n), dtype=float)
    equality[0, :n] = y
    equality[1, : n // 2] = 1.0
    equality[1, n : n + n // 2] = 1.0
    equality_rhs = np.array([0.0, svm_C * n / 2.0])
    lower = np.zeros(2 * n)
    upper = np.full(2 * n, np.inf)
    upper[n // 2 : n] = 1.0
    x0 = np.zeros(2 * n)
    x0[n : n + n // 2] = svm_C
    result = solve_qp(
        hessian,
        linear,
        A_eq=equality,
        b_eq=equality_rhs,
        lb=lower,
        ub=upper,
        x0=x0,
        max_iter=solver_max_iter,
        name="MI-SVM-PI QP",
    )
    return result.x, result, y


def _make_svm_model_from_mi_solution(
    x: np.ndarray,
    result: QPResult,
    K: np.ndarray,
    tK: np.ndarray,
    labels: np.ndarray,
    svm_C: float,
    gamma: float,
    *,
    da: bool = False,
    Kss: np.ndarray | None = None,
    Kst: np.ndarray | None = None,
    svm_C2: float | None = None,
    gamma2: float | None = None,
) -> KernelModel:
    n = len(labels)
    alpha = x[:n]
    y = np.where(np.asarray(labels).reshape(-1) == 1, 1.0, -1.0)
    zeta = x[n : 2 * n]
    tilde_dec = (tK @ (alpha + zeta - svm_C)) / gamma
    tilde_idx = zeta > _EPS
    tilde_b = float(np.mean(-tilde_dec[tilde_idx])) if np.any(tilde_idx) else float(np.max(-tilde_dec))

    if da:
        assert Kss is not None and Kst is not None and svm_C2 is not None
        zeta2 = x[2 * n : 3 * n]
        ns, nt = Kst.shape
        beta2 = alpha + zeta2 - svm_C2
        gamma2_value = float(gamma2 if gamma2 is not None else gamma)
        tilde_dec2 = (tK @ beta2) / gamma2_value
        tilde_dec2 += (svm_C2 / gamma2_value) * np.sum(Kss, axis=1) * beta2
        tilde_dec2 -= (ns / float(nt)) * (svm_C2 / gamma2_value) * np.sum(Kst, axis=1) * beta2
        tilde_idx2 = zeta2 > _EPS
        tilde_b2 = float(np.mean(-tilde_dec2[tilde_idx2])) if np.any(tilde_idx2) else float(np.max(-tilde_dec2))
    else:
        tilde_dec2 = 0.0
        tilde_b2 = 0.0

    dec = K @ (alpha * y)
    dec_2 = -dec + y * (1.0 - tilde_dec - tilde_b)
    if da:
        dec_2 = dec_2 + y * (1.0 - tilde_dec2 - tilde_b2)
    active = alpha > _EPS
    if np.any(active):
        b = float(np.mean(dec_2[active]))
    else:
        positive = dec_2[y > 0]
        negative = dec_2[y < 0]
        b = float((np.max(positive) + np.min(negative)) / 2.0)
    if not np.isfinite(b):
        raise QPError("MI-SVM-PI bias is NaN/inf")
    rho = -b
    coef = alpha * y
    support = np.flatnonzero(alpha > _EPS)
    return KernelModel(
        coef=coef,
        bias=b,
        support_indices=support,
        sv_coef=coef[support],
        rho=rho,
        alpha=alpha,
        x=x,
        y=np.asarray(labels, dtype=float).reshape(-1).copy(),
        objective=result.objective,
        solver_message=result.message,
    )


def train_mi_svm_pi(
    labels: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    param: dict,
) -> KernelModel:
    """Low-level equivalent of ``train_mi_SVM_PI.m``."""

    c = float(param["svm_C"])
    gamma = float(param["gamma"])
    x, result, _ = _solve_mi_pi_qp(
        labels, K, tK, c, gamma, solver_max_iter=int(param.get("solver_max_iter", 2000))
    )
    return _make_svm_model_from_mi_solution(x, result, K, tK, labels, c, gamma)


def _solve_mi_pi_da_qp(
    labels: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    svm_C1: float,
    svm_C2: float,
    gamma1: float,
    gamma2: float,
    B: float,
    epsilon: float,
    *,
    solver_max_iter: int = 3000,
) -> tuple[np.ndarray, QPResult, np.ndarray]:
    labels = np.asarray(labels, dtype=float).reshape(-1)
    n = labels.size
    if n % 2:
        raise ValueError("the supplied MATLAB formulation requires an even number of instances")
    y = np.where(labels == 1, 1.0, -1.0)
    ns, nt = np.asarray(Kst).shape
    if ns != n or nt <= 0:
        raise ValueError("Kst has incompatible dimensions")
    hessian = np.block(
        [
            [K * np.outer(y, y) + tK / gamma1 + Kss / gamma2, tK / gamma1, Kss / gamma2],
            [tK / gamma1, tK / gamma1, np.zeros_like(K)],
            [Kss / gamma2, np.zeros_like(K), Kss / gamma2],
        ]
    )
    hessian += 1e-6 * np.eye(3 * n)
    ratio = ns / float(nt)
    f1 = -np.ones(n) - (svm_C1 / gamma1) * np.sum(tK, axis=1)
    f1 -= (svm_C2 / gamma2) * ratio * np.sum(Kst, axis=1)
    f2 = -(svm_C1 / gamma1) * np.sum(tK, axis=1)
    f3 = -(svm_C2 / gamma2) * ratio * np.sum(Kst, axis=1)
    linear = np.concatenate([f1, f2, f3])
    inv_c2 = 1.0 / svm_C2
    inequality = np.vstack(
        [
            np.concatenate([-np.full(n, inv_c2), np.zeros(n), -np.full(n, inv_c2)]),
            np.concatenate([np.full(n, inv_c2), np.zeros(n), np.full(n, inv_c2)]),
            np.concatenate([np.full(n, inv_c2), np.zeros(n), np.full(n, inv_c2)]),
        ]
    )
    inequality_rhs = np.array([ns * epsilon - n, n + ns * epsilon, B * n], dtype=float)
    equality = np.zeros((2, 3 * n), dtype=float)
    equality[0, :n] = y
    equality[1, : n // 2] = 1.0
    equality[1, n : n + n // 2] = 1.0
    equality_rhs = np.array([0.0, svm_C1 * n / 2.0])
    lower = np.zeros(3 * n)
    upper = np.full(3 * n, np.inf)
    upper[n // 2 : n] = 1.0
    x0 = np.zeros(3 * n)
    x0[n : n + n // 2] = svm_C1
    x0[2 * n :] = svm_C2
    result = solve_qp(
        hessian,
        linear,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        lb=lower,
        ub=upper,
        x0=x0,
        max_iter=solver_max_iter,
        name="MI-SVM-PI-DA QP",
    )
    return result.x, result, y


def train_mi_svm_pi_da(
    labels: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    param: dict,
) -> KernelModel:
    """Low-level equivalent of ``train_mi_svm_PI_DA.m``."""

    c1 = float(param["svm_C1"])
    c2 = float(param["svm_C2"])
    x, result, _ = _solve_mi_pi_da_qp(
        labels,
        K,
        tK,
        Kss,
        Kst,
        c1,
        c2,
        float(param["gamma1"]),
        float(param["gamma2"]),
        float(param.get("B", 1e10)),
        float(param.get("epsilon", 0.0)),
        solver_max_iter=int(param.get("solver_max_iter", 3000)),
    )
    return _make_svm_model_from_mi_solution(
        x,
        result,
        K,
        tK,
        labels,
        c1,
        float(param["gamma1"]),
        da=True,
        Kss=Kss,
        Kst=Kst,
        svm_C2=c2,
        gamma2=float(param["gamma2"]),
    )


def _instance_counts_from_bags(
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
) -> tuple[np.ndarray, np.ndarray, int]:
    pos_counts, neg_fn = _positive_negative_counts(positive_bags, negative_bags)

    def size_of(bag: int | object) -> int:
        if isinstance(bag, (int, np.integer)):
            return int(bag)
        if hasattr(bag, "bag_size"):
            return int(getattr(bag, "bag_size"))
        array = np.asarray(bag)
        return int(array.shape[1])

    neg_counts = np.asarray([size_of(bag) for bag in negative_bags], dtype=int)
    return pos_counts, neg_counts, neg_fn


def _positive_instance_indices(counts: Sequence[int], offset: int = 0) -> list[np.ndarray]:
    result = []
    cursor = int(offset)
    for count in counts:
        result.append(np.arange(cursor, cursor + int(count), dtype=int))
        cursor += int(count)
    return result


def _update_positive_labels(
    labels: np.ndarray,
    decisions: np.ndarray,
    positive_counts: np.ndarray,
    minimum_positive: np.ndarray,
) -> np.ndarray:
    updated = np.asarray(labels, dtype=float).copy()
    cursor = 0
    for count, required in zip(positive_counts, minimum_positive):
        count = int(count)
        required = max(1, min(count, int(required)))
        segment = decisions[cursor : cursor + count]
        # Preserve the global range-threshold labels from the outer loop.
        # MATLAB only flips the highest-scoring required instances to +1 here.
        order = np.argsort(-segment, kind="mergesort")
        updated[cursor + order[:required]] = 1.0
        if not np.any(updated[cursor : cursor + count] > 0):
            updated[cursor + int(np.argmax(segment))] = 1.0
        cursor += count
    return updated


def mi_svm_pi(
    K: np.ndarray,
    tK: np.ndarray,
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
    param: dict,
) -> KernelModel:
    """Train MI-SVM-PI with the positive-bag label refinement loop."""

    pos_counts, neg_counts, neg_fn = _instance_counts_from_bags(positive_bags, negative_bags)
    pos_fn = int(pos_counts.sum())
    n = pos_fn + neg_fn
    if K.shape[0] != n or tK.shape[0] != n:
        raise ValueError("kernel dimensions do not match bag instances")
    labels = np.concatenate([np.ones(pos_fn), -np.ones(neg_fn)])
    ratio = float(param.get("pos_ratio", 0.8))
    if "pos_num" in param:
        lower = np.full(pos_counts.size, int(param["pos_num"]), dtype=int)
    else:
        lower = np.ceil(pos_counts * ratio).astype(int)
    max_iter = int(param.get("max_iter", 100))
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    history, changes = [], []
    stop_reason = "max_iter"
    for iteration in range(max_iter):
        model = train_mi_svm_pi(labels, K, tK, param)
        history.append(float(model.objective))
        changes.append(0 if iteration == 0 else int(np.count_nonzero(labels != old_labels)))
        if iteration > 0 and np.array_equal(labels, old_labels):
            stop_reason = "labels_stable" if changes[-1] == 0 else "label_change_tolerance"
            break
        old_labels = labels.copy()
        decisions = model.decision_function(K[:pos_fn]) if model.support_indices.size else np.zeros(pos_fn)
        if decisions.size:
            threshold = (np.max(decisions) - np.min(decisions)) * 0.2 + np.min(decisions)
            labels[:pos_fn] = np.where(decisions > threshold, 1.0, -1.0)
        labels[:pos_fn] = _update_positive_labels(labels[:pos_fn], decisions, pos_counts, lower)
    model.objective_history = history
    model.outer_iterations = len(history)
    model.stop_reason = stop_reason
    model.label_changes = changes
    return model


def mi_svm_pi_da(
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    positive_bags: Sequence[int] | Sequence[object],
    negative_bags: Sequence[int] | Sequence[object],
    param: dict,
) -> KernelModel:
    """Train MI-SVM-PI with domain adaptation."""

    pos_counts, neg_counts, neg_fn = _instance_counts_from_bags(positive_bags, negative_bags)
    pos_fn = int(pos_counts.sum())
    n = pos_fn + neg_fn
    if K.shape[0] != n or tK.shape[0] != n or Kst.shape[0] != n:
        raise ValueError("kernel dimensions do not match bag instances")
    labels = np.concatenate([np.ones(pos_fn), -np.ones(neg_fn)])
    ratio = float(param.get("pos_ratio", 0.2))
    if "pos_num" in param:
        lower = np.full(pos_counts.size, int(param["pos_num"]), dtype=int)
    else:
        lower = np.ceil(pos_counts * ratio).astype(int)
    max_iter = int(param.get("max_iter", 3))
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    history, changes = [], []
    stop_reason = "max_iter"
    for iteration in range(max_iter):
        model = train_mi_svm_pi_da(labels, K, tK, Kss, Kst, param)
        history.append(float(model.objective))
        changes.append(0 if iteration == 0 else int(np.count_nonzero(labels != old_labels)))
        if model.sv_coef.size == 0:
            stop_reason = "no_support_vectors"
            break
        if iteration > 0 and np.linalg.norm(labels - old_labels) < np.sqrt(pos_counts.size):
            stop_reason = "labels_stable" if changes[-1] == 0 else "label_change_tolerance"
            break
        old_labels = labels.copy()
        decisions = model.decision_function(K[:pos_fn]) if model.support_indices.size else np.zeros(pos_fn)
        if decisions.size:
            threshold = (np.max(decisions) - np.min(decisions)) * 0.4 + np.min(decisions)
            labels[:pos_fn] = np.where(decisions > threshold, 1.0, -1.0)
        labels[:pos_fn] = _update_positive_labels(labels[:pos_fn], decisions, pos_counts, lower)
    model.objective_history = history
    model.outer_iterations = len(history)
    model.stop_reason = stop_reason
    model.label_changes = changes
    return model


def _sum_y_kernels(yset: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    weighted = np.asarray(yset, dtype=float) * np.sqrt(np.asarray(coefficients, dtype=float))[None, :]
    return weighted @ weighted.T


def _initial_mkl_coefficients(m: int, param: dict) -> np.ndarray:
    degree = float(param.get("degree", 1.0))
    d_norm = float(param.get("d_norm", 1.0))
    if "d" in param:
        raw = np.asarray(param["d"], dtype=float).reshape(-1)
        if np.any(raw < 0) or raw.size > m:
            raise ValueError("param['d'] must be nonnegative and no longer than y_set columns")
        norm = np.sum(raw**degree) ** (1.0 / degree)
        coefficients = np.zeros(m, dtype=float)
        if norm > 0:
            coefficients[: raw.size] = d_norm * raw / norm
        return coefficients
    return np.full(m, d_norm * (1.0 / m) ** (1.0 / degree), dtype=float)


def _svm_plus_train_all(
    tmp_y: np.ndarray,
    labels: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    param: dict,
    *,
    Kss: np.ndarray | None = None,
    Kst: np.ndarray | None = None,
    da: bool = False,
) -> tuple[KernelModel, QPResult]:
    n = len(labels)
    if n % 2:
        raise ValueError("the original CPB formulation requires an even number of instances")
    svm_C1 = float(param["svm_C1"] if da else param["svm_C"])
    gamma1 = float(param["gamma1"] if da else param["gamma"])
    if not da:
        svm_C_star = float(param["svm_C_star"])
        hessian = np.block(
            [
                [K + tK / gamma1, tK / gamma1],
                [tK / gamma1, tK / gamma1],
            ]
        )
        hessian += 1e-6 * np.eye(2 * n)
        tmp = -(svm_C1 / gamma1) * np.sum(tK, axis=1)
        linear = np.concatenate([-np.ones(n) + tmp, tmp])
        equality = np.zeros((2, 2 * n))
        equality[0, :n] = tmp_y
        equality[1, : n // 2] = 1.0
        equality[1, n : n + n // 2] = 1.0
        equality_rhs = np.array([0.0, svm_C1 * n / 2.0])
        lower = np.zeros(2 * n)
        upper = np.full(2 * n, np.inf)
        upper[n // 2 : n] = svm_C_star
        x0 = np.zeros(2 * n)
        x0[n : n + n // 2] = svm_C1
        result = solve_qp(
            hessian,
            linear,
            A_eq=equality,
            b_eq=equality_rhs,
            lb=lower,
            ub=upper,
            x0=x0,
            max_iter=int(param.get("solver_max_iter", 2500)),
            name="MIL-CPB-PI QP",
        )
        x = result.x
        alpha = x[:n]
        support = np.flatnonzero(alpha > _EPS)
        coef = alpha[support]
        return (
            KernelModel(
                coef=alpha,
                bias=0.0,
                support_indices=support,
                sv_coef=coef,
                rho=0.0,
                alpha=alpha,
                x=x,
                y=np.ones(n),
                objective=result.objective,
                solver_message=result.message,
            ),
            result,
        )

    assert Kss is not None and Kst is not None
    svm_C2 = float(param["svm_C2"])
    gamma2 = float(param["gamma2"])
    B = float(param.get("B", 1e10))
    epsilon = float(param.get("epsilon", 0.0))
    ns, nt = Kst.shape
    if ns != n or nt <= 0:
        raise ValueError("Kst has incompatible dimensions")
    hessian = np.block(
        [
            [K + tK / gamma1 + Kss / gamma2, tK / gamma1, Kss / gamma2],
            [tK / gamma1, tK / gamma1, np.zeros_like(K)],
            [Kss / gamma2, np.zeros_like(K), Kss / gamma2],
        ]
    )
    hessian += 1e-6 * np.eye(3 * n)
    ratio = ns / float(nt)
    f1 = -np.ones(n) - (svm_C1 / gamma1) * np.sum(tK, axis=1)
    f1 -= (svm_C2 / gamma2) * ratio * np.sum(Kst, axis=1)
    f2 = -(svm_C1 / gamma1) * np.sum(tK, axis=1)
    f3 = -(svm_C2 / gamma2) * ratio * np.sum(Kst, axis=1)
    linear = np.concatenate([f1, f2, f3])
    inv_c2 = 1.0 / svm_C2
    inequality = np.vstack(
        [
            np.concatenate([-np.full(n, inv_c2), np.zeros(n), -np.full(n, inv_c2)]),
            np.concatenate([np.full(n, inv_c2), np.zeros(n), np.full(n, inv_c2)]),
            np.concatenate([np.full(n, inv_c2), np.zeros(n), np.full(n, inv_c2)]),
        ]
    )
    inequality_rhs = np.array([ns * epsilon - n, n + ns * epsilon, B * n])
    equality = np.zeros((2, 3 * n))
    equality[0, :n] = tmp_y
    equality[1, : n // 2] = 1.0
    equality[1, n : n + n // 2] = 1.0
    equality_rhs = np.array([0.0, svm_C1 * n / 2.0])
    lower = np.zeros(3 * n)
    upper = np.full(3 * n, np.inf)
    upper[n // 2 : n] = 1.0
    x0 = np.zeros(3 * n)
    x0[n : n + n // 2] = svm_C1
    x0[2 * n :] = svm_C2
    result = solve_qp(
        hessian,
        linear,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        lb=lower,
        ub=upper,
        x0=x0,
        max_iter=int(param.get("solver_max_iter", 3000)),
        name="MIL-CPB-PI-DA QP",
    )
    x = result.x
    alpha = x[:n]
    support = np.flatnonzero(alpha > _EPS)
    coef = alpha[support]
    return (
        KernelModel(
            coef=alpha,
            bias=0.0,
            support_indices=support,
            sv_coef=coef,
            rho=0.0,
            alpha=alpha,
            x=x,
            y=np.ones(n),
            objective=result.objective,
            solver_message=result.message,
        ),
        result,
    )


def _return_alpha_cpb(
    K: np.ndarray,
    tK: np.ndarray,
    yset: np.ndarray,
    coefficients: np.ndarray,
    param: dict,
    *,
    Kss: np.ndarray | None = None,
    Kst: np.ndarray | None = None,
    da: bool = False,
) -> tuple[KernelModel, float, np.ndarray]:
    n, m = yset.shape
    yK = _sum_y_kernels(yset, coefficients)
    tmp_y = yset @ coefficients
    kernel = K * yK
    inner, _result = _svm_plus_train_all(
        tmp_y,
        np.ones(n),
        kernel,
        tK,
        param,
        Kss=Kss,
        Kst=Kst,
        da=da,
    )
    alpha = inner.x[:n]
    if da:
        assert Kss is not None and Kst is not None
        c1 = float(param["svm_C1"])
        c2 = float(param["svm_C2"])
        gamma1 = float(param["gamma1"])
        gamma2 = float(param["gamma2"])
        zeta1 = inner.x[n : 2 * n]
        zeta2 = inner.x[2 * n : 3 * n]
        beta1 = alpha + zeta1 - c1
        beta2 = alpha + zeta2 - c2
    else:
        c1 = float(param["svm_C"])
        gamma1 = float(param["gamma"])
        zeta1 = inner.x[n : 2 * n]
        beta1 = alpha + zeta1 - c1
        beta2 = None
        gamma2 = 1.0
    aka = np.outer(alpha, alpha) * K
    q = np.asarray([yset[:, i] @ aka @ yset[:, i] for i in range(m)], dtype=float)
    objective = float(np.sum(alpha) - 0.5 * coefficients @ q)
    objective -= float(0.5 / gamma1 * beta1 @ tK @ beta1)
    if da:
        assert beta2 is not None and Kss is not None and Kst is not None
        ns, nt = Kst.shape
        objective -= float(0.5 / gamma2 * beta2 @ Kss @ beta2)
        objective -= float(c2 / gamma2 * beta2 @ np.sum(Kss, axis=1))
        objective += float(c2 / gamma2 * (ns / float(nt)) * beta2 @ np.sum(Kst, axis=1))
    wn = coefficients * np.sqrt(np.maximum(q, 0.0))
    return inner, objective, wn


def train_mil_cpb_pi(
    yset: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    param: dict,
) -> KernelModel:
    """Train one CPB-PI MKL iteration and return its final model."""

    yset = np.asarray(yset, dtype=float)
    if yset.ndim == 1:
        yset = yset[:, None]
    n, m = yset.shape
    if K.shape != (n, n) or tK.shape != (n, n):
        raise ValueError("CPB kernel dimensions do not match yset")
    coefficients = _initial_mkl_coefficients(m, param)
    degree = float(param.get("degree", 1.0))
    d_norm = float(param.get("d_norm", 1.0))
    max_iter = int(param.get("mkl_max_iter", 100))
    tau = float(param.get("mkl_tau", 0.001))
    history: list[float] = []
    final_model: KernelModel | None = None
    for _ in range(max_iter):
        final_model, objective, wn = _return_alpha_cpb(K, tK, yset, coefficients, param)
        history.append(objective)
        if m == 1 or len(history) == max_iter or (len(history) > 1 and objective > history[-2] - tau * abs(objective)):
            break
        wnp = wn ** (2.0 / (degree + 1.0))
        eta = np.sum(wnp**degree) ** (1.0 / degree)
        if eta <= _EPS:
            break
        coefficients = d_norm * wnp / eta
    assert final_model is not None
    effective_labels = yset @ coefficients
    coef_full = np.zeros(n)
    coef_full[final_model.support_indices] = final_model.sv_coef * effective_labels[final_model.support_indices]
    final_model.coef = coef_full
    final_model.sv_coef = coef_full[final_model.support_indices]
    final_model.y = effective_labels
    final_model.effective_labels = effective_labels
    final_model.d = coefficients
    final_model.objective = history[-1]
    final_model.objective_history = history
    return final_model


def train_mil_cpb_pi_da(
    yset: np.ndarray,
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    param: dict,
) -> KernelModel:
    """Train one CPB-PI-DA MKL iteration and return its final model."""

    yset = np.asarray(yset, dtype=float)
    if yset.ndim == 1:
        yset = yset[:, None]
    n, m = yset.shape
    if K.shape != (n, n) or tK.shape != (n, n) or Kss.shape != (n, n) or Kst.shape[0] != n:
        raise ValueError("CPB-PI-DA kernel dimensions do not match yset")
    coefficients = _initial_mkl_coefficients(m, param)
    degree = float(param.get("degree", 1.0))
    d_norm = float(param.get("d_norm", 1.0))
    max_iter = int(param.get("mkl_max_iter", 100))
    tau = float(param.get("mkl_tau", 0.001))
    history: list[float] = []
    final_model: KernelModel | None = None
    for _ in range(max_iter):
        final_model, objective, wn = _return_alpha_cpb(
            K, tK, yset, coefficients, param, Kss=Kss, Kst=Kst, da=True
        )
        history.append(objective)
        if m == 1 or len(history) == max_iter or (len(history) > 1 and objective > history[-2] - tau * abs(objective)):
            break
        wnp = wn ** (2.0 / (degree + 1.0))
        eta = np.sum(wnp**degree) ** (1.0 / degree)
        if eta <= _EPS:
            break
        coefficients = d_norm * wnp / eta
    assert final_model is not None
    effective_labels = yset @ coefficients
    coef_full = np.zeros(n)
    coef_full[final_model.support_indices] = final_model.sv_coef * effective_labels[final_model.support_indices]
    final_model.coef = coef_full
    final_model.sv_coef = coef_full[final_model.support_indices]
    final_model.y = effective_labels
    final_model.effective_labels = effective_labels
    final_model.d = coefficients
    final_model.objective = history[-1]
    final_model.objective_history = history
    return final_model


def _enumerate_positive_patterns(bag_size: int, minimum_positive: int) -> np.ndarray:
    if bag_size <= 0:
        raise ValueError("bag_size must be positive")
    minimum_positive = max(1, min(bag_size, int(minimum_positive)))
    patterns = []
    for mask in range(1 << bag_size):
        bits = np.array([(mask >> i) & 1 for i in range(bag_size)], dtype=float)
        if bits.sum() >= minimum_positive:
            patterns.append(np.where(bits > 0, 1.0, -1.0))
    return np.asarray(patterns, dtype=float).T


def infer_y_enum(
    y_pos: np.ndarray,
    model: KernelModel,
    bags: Sequence[np.ndarray],
    labels: np.ndarray,
    Q: np.ndarray,
    y_set: np.ndarray | None = None,
) -> np.ndarray:
    """Enumerate feasible positive-bag labelings and maximize the CPB score.

    This is the missing ``infer_y_enum`` utility.  For each positive bag it
    enumerates all allowed bit patterns and performs coordinate ascent on
    ``y.T @ ((Q .* (alpha alpha.T)) @ y)``.  With the original bag size of five,
    each coordinate has at most 32 candidates.
    """

    del y_set  # retained in the signature because the MATLAB caller passes it
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    alpha = np.asarray(model.alpha, dtype=float).reshape(-1)
    if alpha.size != n:
        raise ValueError("model alpha and Q dimensions do not match")
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(bags) != labels.size:
        raise ValueError("bags and bag labels must have the same length")
    score_matrix = Q * np.outer(alpha, alpha)
    y = np.full(n, -1.0, dtype=float)
    positive_bags = []
    for index, label in zip(bags, labels):
        index = np.asarray(index, dtype=int).reshape(-1)
        if label > 0:
            positive_bags.append(index)
            y[index] = 1.0
    if not positive_bags:
        return y
    for _ in range(20):
        changed = False
        for index in positive_bags:
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[index] = False
            rest = np.flatnonzero(rest_mask)
            current = y[rest]
            block = score_matrix[np.ix_(index, index)]
            cross = score_matrix[np.ix_(index, rest)] @ current
            patterns = y_pos if y_pos.shape[0] == index.size else _enumerate_positive_patterns(index.size, 1)
            scores = np.einsum("ik,ij,jk->k", patterns, block, patterns) + 2.0 * (patterns.T @ cross)
            best = patterns[:, int(np.argmax(scores))]
            if not np.array_equal(y[index], best):
                changed = True
                y[index] = best
        if not changed:
            break
    return y


def _build_initial_bag_state(
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    positive_indices = _positive_instance_indices(positive_counts)
    negative_indices = _positive_instance_indices(negative_counts, int(positive_counts.sum()))
    bags = positive_indices + negative_indices
    labels = np.concatenate([np.ones(len(positive_indices)), -np.ones(len(negative_indices))])
    initial = np.concatenate(
        [np.ones(int(positive_counts.sum())), -np.ones(int(negative_counts.sum()))]
    )
    return bags, labels, initial


def mil_cpb_pi(
    K: np.ndarray,
    tK: np.ndarray,
    bags: Sequence[np.ndarray],
    bag_labels: np.ndarray,
    param: dict,
) -> KernelModel:
    """Train MIL-CPB-PI with cutting-plane label inference."""

    K = np.asarray(K, dtype=float)
    tK = np.asarray(tK, dtype=float)
    bag_labels = np.asarray(bag_labels, dtype=float).reshape(-1)
    bag_indices = [np.asarray(index, dtype=int).reshape(-1) for index in bags]
    n = K.shape[0]
    if K.shape != tK.shape or K.shape[0] != n:
        raise ValueError("K and tK must be matching square matrices")
    if len(bag_indices) != bag_labels.size:
        raise ValueError("bags and bag_labels must have matching lengths")
    if len({int(index.size) for index in bag_indices}) != 1:
        raise ValueError("MIL-CPB uses one common bag size")
    bag_size = bag_indices[0].size
    initial = np.zeros(n)
    for index, label in zip(bag_indices, bag_labels):
        initial[index] = 1.0 if label > 0 else -1.0
    min_positive = int(np.ceil(bag_size * float(param.get("pos_ratio", 0.6))))
    y_pos = _enumerate_positive_patterns(bag_size, min_positive)
    yset = initial[:, None]
    max_iter = int(param.get("max_iter", 100))
    history: list[float] = []
    model: KernelModel | None = None
    for _ in range(max_iter):
        model = train_mil_cpb_pi(yset, K, tK, param)
        model.y_set = yset.copy()
        history.append(float(model.objective))
        if len(history) > 1 and history[-1] > history[-2] - 0.01 * abs(history[-2]):
            break
        if len(history) == max_iter:
            break
        inferred = infer_y_enum(y_pos, model, bag_indices, bag_labels, K + 1.0, yset)
        yset = np.column_stack([yset, inferred])
    assert model is not None
    model.y_set = yset
    model.objective_history = history
    return model


def mil_cpb_pi_da(
    K: np.ndarray,
    tK: np.ndarray,
    Kss: np.ndarray,
    Kst: np.ndarray,
    bags: Sequence[np.ndarray],
    bag_labels: np.ndarray,
    param: dict,
) -> KernelModel:
    """Train MIL-CPB-PI with domain adaptation."""

    K = np.asarray(K, dtype=float)
    tK = np.asarray(tK, dtype=float)
    Kss = np.asarray(Kss, dtype=float)
    Kst = np.asarray(Kst, dtype=float)
    bag_labels = np.asarray(bag_labels, dtype=float).reshape(-1)
    bag_indices = [np.asarray(index, dtype=int).reshape(-1) for index in bags]
    n = K.shape[0]
    if K.shape != tK.shape or Kss.shape != (n, n) or Kst.shape[0] != n:
        raise ValueError("CPB-PI-DA kernel dimensions do not match")
    if len(bag_indices) != bag_labels.size:
        raise ValueError("bags and bag_labels must have matching lengths")
    if len({int(index.size) for index in bag_indices}) != 1:
        raise ValueError("MIL-CPB uses one common bag size")
    bag_size = bag_indices[0].size
    initial = np.zeros(n)
    for index, label in zip(bag_indices, bag_labels):
        initial[index] = 1.0 if label > 0 else -1.0
    min_positive = int(np.ceil(bag_size * float(param.get("pos_ratio", 0.2))))
    y_pos = _enumerate_positive_patterns(bag_size, min_positive)
    yset = initial[:, None]
    max_iter = int(param.get("max_iter", 100))
    history: list[float] = []
    model: KernelModel | None = None
    for _ in range(max_iter):
        model = train_mil_cpb_pi_da(yset, K, tK, Kss, Kst, param)
        model.y_set = yset.copy()
        history.append(float(model.objective))
        if len(history) > 1 and history[-1] > history[-2] - 0.001 * abs(history[-1]):
            break
        if len(history) == max_iter:
            break
        inferred = infer_y_enum(y_pos, model, bag_indices, bag_labels, K + 1.0, yset)
        yset = np.column_stack([yset, inferred])
    assert model is not None
    model.y_set = yset
    model.objective_history = history
    return model
