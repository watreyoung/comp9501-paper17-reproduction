"""Binary hinge-SVM pair coordinate ascent with an on-demand kernel oracle.

Solve the original dual (no low-rank approximation). O(n) optimizer state;
no n*n Gram allocation. KKT gap and cap remain explicit. Basic max-gap
working set favors auditability; large-scale performance still needs testing.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ColumnSVM:
    alpha: np.ndarray
    labels: np.ndarray
    bias: float
    dual_objective: float
    kkt_gap: float
    iterations: int
    stop_reason: str
    equality_residual: float

    @property
    def coefficients(self):return self.alpha*self.labels

    def decision_function(self,cross_kernel):return np.asarray(cross_kernel)@self.coefficients+self.bias


def fit_column_svm(labels,costs,column,*,diagonal=None,tolerance=1e-6,max_iter=100000,backend="numpy"):
    """column(i) -> K[:,i]; PSD kernel and accurate columns are required.

    Pair updates maintain y'alpha=0. Stop when max(y-f) over upward
    beta=alpha*y directions minus min(y-f) over downward directions <=tol.
    """
    y=np.asarray(labels,float);cost=np.asarray(costs,float);n=len(y)
    if y.ndim!=1 or cost.shape!=(n,) or set(np.unique(y))!={-1.,1.} or not np.isfinite(cost).all() or np.any(cost<=0) or tolerance<=0 or max_iter<1:raise ValueError('Two-class labels, positive finite costs and solver limits required')
    diag=np.asarray(diagonal,float) if diagonal is not None else np.array([column(i)[i] for i in range(n)])
    if diag.shape!=(n,) or not np.isfinite(diag).all():raise ValueError('Finite kernel diagonal required')
    if backend not in ['numpy','numba']:raise ValueError('Unknown SMO scan backend')
    if backend=='numba':
        from .column_svm_numba import select_pair,update_scores
    alpha=np.zeros(n);f=np.zeros(n);reason='max_iter';updates=0
    def working_set():
        residual=y-f
        up=((y>0)&(alpha<cost))|((y<0)&(alpha>0))
        low=((y>0)&(alpha>0))|((y<0)&(alpha<cost))
        i=int(np.argmax(np.where(up,residual,-np.inf)))
        j=int(np.argmin(np.where(low,residual,np.inf)))
        return i,j,float(residual[i]-residual[j]),residual,up,low
    for iteration in range(max_iter):
        if backend=='numba':i,j,gap=select_pair(y,alpha,cost,f)
        else:i,j,gap,_,_,_=working_set()
        if gap<=tolerance:reason='kkt_tolerance';break
        ki=np.asarray(column(i),float);kj=np.asarray(column(j),float)
        if ki.shape!=(n,) or kj.shape!=(n,) or not np.isfinite(ki).all() or not np.isfinite(kj).all():raise ValueError('Oracle must return finite kernel columns')
        curvature=float(diag[i]+diag[j]-2*ki[j])
        if curvature<-1e-9:raise ArithmeticError('Kernel has negative pair curvature')
        allowance_i=cost[i]-alpha[i] if y[i]>0 else alpha[i]
        allowance_j=alpha[j] if y[j]>0 else cost[j]-alpha[j]
        limit=min(allowance_i,allowance_j)
        step=min(gap/curvature,limit) if curvature>1e-15 else limit
        if step<=0:raise ArithmeticError('Nonpositive SMO step despite KKT violation')
        alpha[i]+=y[i]*step;alpha[j]-=y[j]*step
        # Only snap an endpoint actually reached by this pair step. Absolute
        # tolerances can incorrectly snap both bounds when C is very small.
        if step==allowance_i:alpha[i]=cost[i] if y[i]>0 else 0.
        if step==allowance_j:alpha[j]=0. if y[j]>0 else cost[j]
        if backend=='numba':update_scores(f,ki,kj,step)
        else:f+=step*(ki-kj)
        updates+=1
    _,_,gap,residual,up,low=working_set()
    if gap<=tolerance:reason='kkt_tolerance'
    lower=float(residual[up].max());upper=float(residual[low].min())
    free=(alpha>0)&(alpha<cost)
    bias=float(np.mean(residual[free])) if free.any() else (lower+upper)/2
    return ColumnSVM(alpha,y,bias,float(alpha.sum()-.5*(alpha*y)@f),gap,updates,reason,float(abs(y@alpha)))
