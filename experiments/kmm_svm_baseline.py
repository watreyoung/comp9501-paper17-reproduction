"""KMM importance-weighted event SVM, complementing frozen KMM+MIL-PI.

Same equality-constrained KMM and box B=10 as existing controls; Gaussian
kernels/median bandwidth supplied by shared event protocol. Target labels
are never inputs. Costs remain positive C1 / negative 1 before weighting.
"""
import numpy as np
from .alignment_models import kmm_weights
from mil_pi.paper_models import dual_problem
from mil_pi.qp import qp_session


def fit_kmm_svm(K,Kst,y,costs,*,bound=10.):
    K=np.asarray(K,float);Kst=np.asarray(Kst,float);y=np.asarray(y,float);costs=np.asarray(costs,float)
    n=len(y)
    if K.shape!=(n,n) or Kst.ndim!=2 or Kst.shape[0]!=n or not Kst.shape[1] or costs.shape!=(n,) or set(np.unique(y))!={-1.,1.}:raise ValueError('Matching kernels, two-class labels and per-instance costs required')
    if not np.isfinite(K).all() or not np.isfinite(Kst).all() or not np.isfinite(costs).all() or np.any(costs<=0) or not np.isfinite(bound) or bound<1:raise ValueError('Finite kernels, positive costs and feasible KMM bound required')
    with qp_session('osqp') as trace:
        weights=kmm_weights(K,Kst,bound=bound)
        model=dual_problem(K,np.zeros_like(K),y,np.ones(n),costs*weights,np.zeros(n,bool))
    diagnostics=dict(weights=weights.tolist(),weight_sum=float(weights.sum()),bound=bound,
                     max_constraint_violation=max(r['constraint_violation'] for r in trace),qp_diagnostics=trace,
                     classifier_objective=model.objective,stop_reason=model.stop_reason)
    return model,weights,diagnostics
