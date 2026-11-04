"""Small convex-QP adapter replacing MATLAB ``quadprog``/MOSEK calls."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass
class QPResult:
    x: np.ndarray
    success: bool
    status: int
    message: str
    objective: float
    iterations: int
    backend: str = "slsqp"
    primal_residual: float | None = None
    dual_residual: float | None = None
    constraint_violation: float | None = None
    equality_multipliers: np.ndarray | None = None


_BACKEND: ContextVar[str] = ContextVar("mil_pi_qp_backend", default="slsqp")
_TRACE: ContextVar[list | None] = ContextVar("mil_pi_qp_trace", default=None)


@contextmanager
def qp_session(backend: str = "osqp"):
    """Select a solver for one experiment and collect every QP diagnostic."""
    if backend not in {"osqp", "slsqp"}:
        raise ValueError(f"unknown QP backend: {backend}")
    trace: list[dict] = []
    backend_token = _BACKEND.set(backend)
    trace_token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _BACKEND.reset(backend_token)
        _TRACE.reset(trace_token)


class QPError(RuntimeError):
    """Raised when the fallback QP solver cannot produce a feasible solution."""


def _matrix(value: np.ndarray | None, rows: int, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    array = array.reshape(-1, rows)
    return array


def _vector(value: np.ndarray | None, size: int, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} has length {array.size}, expected {size}")
    return array


def _scale_rows(matrix: np.ndarray | None, rhs: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if matrix is None:
        return None, rhs
    scales = np.maximum(1.0, np.max(np.abs(matrix), axis=1))
    return matrix / scales[:, None], None if rhs is None else rhs / scales


def _constraint_residuals(
    x: np.ndarray,
    A_ub: np.ndarray | None,
    b_ub: np.ndarray | None,
    A_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float, float]:
    inequality = 0.0
    equality = 0.0
    bounds = 0.0
    if A_ub is not None and b_ub is not None:
        inequality = float(np.max(np.maximum(A_ub @ x - b_ub, 0.0), initial=0.0))
    if A_eq is not None and b_eq is not None:
        equality = float(np.max(np.abs(A_eq @ x - b_eq), initial=0.0))
    bounds = float(
        max(
            np.max(np.maximum(lb - x, 0.0), initial=0.0),
            np.max(np.maximum(x - ub, 0.0), initial=0.0),
        )
    )
    return inequality, equality, bounds


def _solve_slsqp(
    H: np.ndarray,
    f: np.ndarray,
    *,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
    name: str = "quadratic program",
) -> QPResult:
    """Solve a dense convex quadratic program with scipy's SLSQP backend.

    The archive relies on MATLAB ``quadprog`` and, for some demos, MOSEK.  This
    project intentionally has no proprietary dependency, so we solve the same
    problem locally with analytic objective gradients and linear constraints.
    Row scaling is applied only to constraints and does not change the feasible
    set; it matters for the very small ``svm_C2`` values in the DA variants.
    """

    hessian = np.asarray(H, dtype=float)
    linear = np.asarray(f, dtype=float).reshape(-1)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("H must be a square matrix")
    n = hessian.shape[0]
    if linear.size != n:
        raise ValueError("f and H dimensions do not match")
    hessian = 0.5 * (hessian + hessian.T)

    if lb is None:
        lower = np.full(n, -np.inf)
    else:
        lower = np.asarray(lb, dtype=float).reshape(-1)
    if ub is None:
        upper = np.full(n, np.inf)
    else:
        upper = np.asarray(ub, dtype=float).reshape(-1)
    if lower.size != n or upper.size != n:
        raise ValueError("bound dimensions do not match H")
    if np.any(lower > upper):
        raise ValueError("lower bounds exceed upper bounds")

    inequality = None if A_ub is None else np.asarray(A_ub, dtype=float)
    if inequality is not None:
        if inequality.ndim != 2 or inequality.shape[1] != n:
            raise ValueError("A_ub has incompatible shape")
        ub_rhs = _vector(b_ub, inequality.shape[0], name="b_ub")
    else:
        ub_rhs = None

    equality = None if A_eq is None else np.asarray(A_eq, dtype=float)
    if equality is not None:
        if equality.ndim != 2 or equality.shape[1] != n:
            raise ValueError("A_eq has incompatible shape")
        eq_rhs = _vector(b_eq, equality.shape[0], name="b_eq")
    else:
        eq_rhs = None

    # Scaling rows is safe for both equality and inequality constraints.
    inequality, ub_rhs = _scale_rows(inequality, ub_rhs)
    equality, eq_rhs = _scale_rows(equality, eq_rhs)

    if x0 is None:
        start = np.zeros(n, dtype=float)
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        start[finite_lower] = np.maximum(start[finite_lower], lower[finite_lower])
        start[finite_upper] = np.minimum(start[finite_upper], upper[finite_upper])
    else:
        start = np.asarray(x0, dtype=float).reshape(-1).copy()
        if start.size != n:
            raise ValueError("x0 and H dimensions do not match")
    start = np.maximum(start, np.where(np.isfinite(lower), lower, -1e100))
    start = np.minimum(start, np.where(np.isfinite(upper), upper, 1e100))

    constraints: list[dict[str, Callable[[np.ndarray], np.ndarray]]] = []
    if inequality is not None and ub_rhs is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, a=inequality, b=ub_rhs: b - a @ x,
                "jac": lambda _x, a=inequality: -a,
            }
        )
    if equality is not None and eq_rhs is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, a=equality, b=eq_rhs: a @ x - b,
                "jac": lambda _x, a=equality: a,
            }
        )

    def objective(x: np.ndarray) -> float:
        return float(0.5 * x @ hessian @ x + linear @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return hessian @ x + linear

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )

    # Successful termination and feasibility in the original constraints
    # are both required, including calls outside a qp_session.
    raw_ineq = None if A_ub is None else np.asarray(A_ub, dtype=float)
    raw_eq = None if A_eq is None else np.asarray(A_eq, dtype=float)
    raw_b_ub = None if b_ub is None else np.asarray(b_ub, dtype=float).reshape(-1)
    raw_b_eq = None if b_eq is None else np.asarray(b_eq, dtype=float).reshape(-1)
    x = np.asarray(result.x, dtype=float)
    violation = _constraint_residuals(x, raw_ineq, raw_b_ub, raw_eq, raw_b_eq, lower, upper)
    feasible = max(violation) <= max(1e-6, 100.0 * tol)
    success = bool(result.success and feasible)
    if not success:
        raise QPError(
            f"{name} failed: {result.message}; constraint residuals="
            f"(ineq={violation[0]:.3g}, eq={violation[1]:.3g}, bounds={violation[2]:.3g})"
        )

    message = str(result.message)
    return QPResult(
        x=x,
        success=success,
        status=int(result.status),
        message=message,
        objective=objective(x),
        iterations=int(getattr(result, "nit", 0) or 0),
        constraint_violation=max(violation),
    )


def _solve_osqp(H, f, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                lb=None, ub=None, x0=None, max_iter=2000, tol=1e-8,
                name="quadratic program"):
    import osqp
    from scipy import sparse

    h = np.asarray(H, dtype=float)
    q = np.asarray(f, dtype=float).reshape(-1)
    n = q.size
    if h.shape != (n, n) or not np.isfinite(h).all() or not np.isfinite(q).all():
        raise ValueError("QP objective must have finite, matching dimensions")
    h = 0.5 * (h + h.T)
    lower = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    upper = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float).reshape(-1)
    if lower.size != n or upper.size != n or np.any(lower > upper):
        raise ValueError("invalid variable bounds")
    blocks = [sparse.eye(n, format="csc")]
    lows = [lower]
    highs = [upper]
    for matrix, rhs, equality in [(A_ub, b_ub, False), (A_eq, b_eq, True)]:
        if matrix is None:
            continue
        matrix = np.asarray(matrix, dtype=float)
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if matrix.ndim != 2 or matrix.shape != (rhs.size, n):
            raise ValueError("constraint dimensions do not match objective")
        matrix, rhs = _scale_rows(matrix, rhs)
        blocks.append(sparse.csc_matrix(matrix))
        lows.append(rhs if equality else np.full(rhs.size, -np.inf))
        highs.append(rhs)
    a = sparse.vstack(blocks, format="csc")
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(h)), q=q, A=a,
                 l=np.concatenate(lows), u=np.concatenate(highs),
                 eps_abs=min(tol, 1e-11), eps_rel=min(tol, 1e-11), max_iter=int(max_iter),
                 polishing=True, verbose=False)
    if x0 is not None:
        solver.warm_start(x=np.asarray(x0, dtype=float).reshape(-1))
    result = solver.solve(raise_error=False)
    info = result.info
    if result.x is None or info.status_val != 1:
        raise QPError(f"{name}: OSQP {info.status}; primal={info.prim_res:.3g}, dual={info.dual_res:.3g}")
    x = np.asarray(result.x, dtype=float)
    violation = max(_constraint_residuals(x, A_ub, b_ub, A_eq, b_eq, lower, upper))
    if not np.isfinite(x).all() or violation > max(1e-6, 100 * tol):
        raise QPError(f"{name}: OSQP returned invalid constraints, residual={violation:.3g}")
    equality_multipliers = None
    if A_eq is not None:
        start_dual = n + (0 if A_ub is None else len(b_ub))
        scales = np.maximum(1.0, np.max(np.abs(np.asarray(A_eq)), axis=1))
        equality_multipliers = np.asarray(result.y[start_dual:]) / scales
    return QPResult(x=x, success=True, status=int(info.status_val),
                    message=f"OSQP {info.status}", objective=float(0.5*x@h@x+q@x),
                    iterations=int(info.iter), backend="osqp",
                    primal_residual=float(info.prim_res), dual_residual=float(info.dual_res),
                    constraint_violation=float(violation),
                    equality_multipliers=equality_multipliers)


def solve_qp(H, f, **kwargs) -> QPResult:
    """Use the original solver by default; experiments can select OSQP explicitly.

    OSQP solves the same quadratic objective and linear constraints. It rejects
    unsuccessful termination instead of accepting a merely feasible point.
    """
    backend = _BACKEND.get()
    result = (_solve_osqp if backend == "osqp" else _solve_slsqp)(H, f, **kwargs)
    trace = _TRACE.get()
    if trace is not None:
        trace.append({"backend": backend, "success": result.success,
                      "message": result.message, "iterations": result.iterations,
                      "objective": result.objective,
                      "constraint_violation": result.constraint_violation,
                      "primal_residual": result.primal_residual,
                      "dual_residual": result.dual_residual})
        if not result.success:
            raise QPError(f"{kwargs.get('name', 'QP')}: {result.message}")
    return result
