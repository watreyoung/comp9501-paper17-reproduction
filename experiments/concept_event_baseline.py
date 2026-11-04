"""Core-cost event SVM with serializable coefficients for concept features."""
import numpy as np
from mil_pi.paper_models import dual_problem
from mil_pi.qp import qp_session


def fit_concept_event(problem, *, scale=1.):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite validation scale required')
    K, cross = problem['K'], problem['test_kernel']
    npos = problem['n_bags']*problem['bag_size']
    if K.shape != (2*npos, 2*npos):
        raise ValueError('Shared balanced event-bag protocol required')
    y = np.r_[np.ones(npos), -np.ones(npos)]
    costs = np.r_[np.full(npos, 100*scale), np.ones(npos)]
    with qp_session('osqp') as trace:
        model = dual_problem(K, problem['full_tK'], y, np.ones(len(K)), costs, np.zeros(len(K), bool), gamma=100.)
    scores = model.predict(cross)
    violation = max((r['constraint_violation'] for r in trace), default=0.)
    if not np.isfinite(scores).all() or violation > 1e-6:
        raise ArithmeticError('Nonfinite prediction or infeasible concept SVM')
    diag = dict(stop_reason=model.stop_reason, objective=model.objective,
                max_constraint_violation=violation, qp_diagnostics=trace,
                positive_cost=100*scale, negative_cost=1.,
                kernel_gamma=problem['kernel_gamma'], target_labels_used=False)
    return scores, diag, dict(coef=model.coef, bias=np.array(model.bias), kernel_gamma=np.array(problem['kernel_gamma']))
