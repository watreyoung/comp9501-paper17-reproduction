"""Selective Transfer Machine, Chu et al. CVPR 2013, Eq.3/6, Algorithm 1.

Use quadratic hinge (p=2) and the nonlinear representer form with no extra
free intercept, exactly as Eq.3. The caller supplies the classification and
MMD Gram matrix. lambda_ multiplies the unnormalized QP form of Eq.6;
it absorbs the n²/2 constant relative to raw empirical MMD² in Eq.4.
"""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from mil_pi.qp import qp_session,solve_qp
from .multiview_baselines import kernel_features


def quadratic_kernel_svm(K,y,cost):
    """Independent feature-space convex primal, including exact zero costs."""
    F,R=kernel_features(K);n=len(y);w=cp.Variable(F.shape[1]);slack=cp.Variable(n,nonneg=True)
    constraints=[cp.multiply(y,F@w)>=1-slack]
    problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(w)+cp.sum(cp.multiply(cost,cp.square(slack)))),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=300)
    if problem.status!='optimal':raise ArithmeticError('STM classifier solve failed: '+problem.status)
    violation=max(float(np.max(np.abs(c.violation()))) for c in constraints)
    if violation>1e-6:raise ArithmeticError('STM classifier constraint violation')
    coef=R@w.value;scores=K@coef;loss=np.maximum(1-y*scores,0)**2
    return coef,loss,float(.5*coef@K@coef+cost@loss),violation


def selective_weights(K,kappa,loss,cost,lambda_,bound,epsilon):
    n=len(K);H=K+1e-8*np.eye(n);f=cost*loss/lambda_-kappa
    kwargs=dict(lb=np.zeros(n),ub=np.full(n,bound),max_iter=200000,name='STM selective weights Eq6')
    if epsilon==0:kwargs.update(A_eq=np.ones((1,n)),b_eq=[float(n)])
    else:kwargs.update(A_ub=np.r_[np.ones((1,n)),-np.ones((1,n))],b_ub=[n*(1+epsilon),-n*(1-epsilon)])
    result=solve_qp(H,f,**kwargs)
    return np.clip(result.x,0,bound),result


@dataclass
class SelectiveTransferMachine:
    coef: np.ndarray
    weights: np.ndarray
    objective: float
    history: list
    stop_reason: str
    qp_diagnostics: list
    max_constraint_violation: float

    def predict(self,cross_kernel):return np.asarray(cross_kernel)@self.coef


def fit_stm(K,Kst,y,*,C=1.,lambda_=1.,bound=10.,epsilon=0.,max_iter=100,tolerance=1e-7):
    K=np.asarray(K,float);Kst=np.asarray(Kst,float);y=np.asarray(y,float);n=len(K)
    cost=np.broadcast_to(np.asarray(C,float),(n,)).copy()
    if K.shape!=(n,n) or Kst.ndim!=2 or Kst.shape[0]!=n or not Kst.shape[1]:raise ValueError('Matching source and source-target kernels required')
    if not np.isin(y,[-1,1]).all() or y.shape!=(n,) or len(np.unique(y))!=2:raise ValueError('Binary source labels required')
    if np.any(cost<=0) or lambda_<=0 or bound<1 or not 0<=epsilon<1 or max_iter<1:raise ValueError('Invalid STM parameters')
    kappa=n*Kst.mean(axis=1);H=K+1e-8*np.eye(n);loss=np.zeros(n)
    history=[];reason='max_iter';max_violation=0.
    with qp_session('osqp') as trace:
        for iteration in range(max_iter):
            weights,result=selective_weights(K,kappa,loss,cost,lambda_,bound,epsilon)
            coef,loss,classifier_objective,violation=quadratic_kernel_svm(K,y,cost*weights)
            objective=classifier_objective+lambda_*(.5*weights@H@weights-kappa@weights)
            max_violation=max(max_violation,violation,result.constraint_violation or 0.)
            history.append(dict(iteration=iteration+1,objective=float(objective),classifier_objective=classifier_objective,source_weight_sum=float(weights.sum()),max_weight=float(weights.max()),min_weight=float(weights.min()),classifier_constraint_violation=violation))
            if len(history)>1:
                change=history[-2]['objective']-objective;scale=max(1.,abs(objective))
                if change< -1e-6*scale:raise ArithmeticError('STM alternating objective increased beyond numerical tolerance')
                if abs(change)<=tolerance*scale:reason='objective_tolerance';break
    return SelectiveTransferMachine(coef,weights,float(objective),history,reason,trace,max_violation)
