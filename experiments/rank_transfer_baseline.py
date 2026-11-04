"""Sharmanska et al. ICCV 2013 Rank Transfer, Eq.6–11 / Algorithm 1.

Both representations are linear, as required by the target IJCV paper's RT
baseline. Teacher pairs have yi>yj. Student pairs explicitly include yi==yj
(Eq.9 uses >=), and use the published practical rho>0.1 cutoff.
"""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp


def rank_svm_primal(X,i,j,margins,C):
    """Solve the linear hinge-ranking primal in its exact data row space."""
    X=np.asarray(X,float);i=np.asarray(i,int);j=np.asarray(j,int);margins=np.asarray(margins,float)
    if C<=0:raise ValueError('Positive ranking cost required')
    if not len(i):return np.zeros(X.shape[1]),0.,0.
    U,s,Vt=np.linalg.svd(X,full_matrices=False)
    rank=int(np.count_nonzero(s>max(X.shape)*np.finfo(float).eps*s[0])) if len(s) and s[0]>0 else 0
    if rank==0:return np.zeros(X.shape[1]),float(C*margins.sum()),0.
    F=U[:,:rank]*s[:rank];difference=F[i]-F[j]
    w=cp.Variable(rank);slack=cp.Variable(len(i),nonneg=True)
    constraints=[difference@w>=margins-slack]
    p=cp.Problem(cp.Minimize(.5*cp.sum_squares(w)+C*cp.sum(slack)),constraints)
    p.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9,max_iter=300)
    if p.status!='optimal':raise ArithmeticError('RT ranking primal solve failed: '+p.status)
    violation=max(float(np.max(np.abs(c.violation()))) for c in constraints)
    if violation>1e-6:raise ArithmeticError('RT ranking primal infeasible')
    weights=Vt[:rank].T@w.value
    hinge=np.maximum(margins-(X[i]-X[j])@weights,0)
    objective=float(.5*weights@weights+C*hinge.sum())
    return weights,objective,violation


@dataclass
class RankTransfer:
    visual_weights: np.ndarray
    privileged_weights: np.ndarray
    teacher_pairs: np.ndarray
    student_pairs: np.ndarray
    student_margins: np.ndarray
    teacher_objective: float
    student_objective: float
    max_constraint_violation: float
    margin_cutoff: float

    def predict(self,visual):return np.asarray(visual)@self.visual_weights


def fit_rank_transfer(visual,privileged,y,*,teacher_C=1.,student_C=1.,margin_cutoff=.1):
    X=np.asarray(visual,float);T=np.asarray(privileged,float);y=np.asarray(y,float)
    if X.ndim!=2 or T.ndim!=2 or len(X)!=len(T) or len(X)!=len(y):raise ValueError('Paired source representations required')
    if not np.isfinite(X).all() or not np.isfinite(T).all():raise ValueError('Finite source features required')
    if not np.isin(y,[-1,1]).all() or len(np.unique(y))!=2 or margin_cutoff<0:raise ValueError('Binary labels and nonnegative cutoff required')
    i,j=np.where(y[:,None]>y[None,:]);teacher_pairs=np.c_[i,j]
    tw,to,tv=rank_svm_primal(T,i,j,np.ones(len(i)),teacher_C)
    scores=T@tw;rho=scores[:,None]-scores[None,:]
    eligible=(y[:,None]>=y[None,:])&(rho>margin_cutoff)
    si,sj=np.where(eligible);student_pairs=np.c_[si,sj];margins=rho[si,sj]
    vw,vo,vv=rank_svm_primal(X,si,sj,margins,student_C)
    return RankTransfer(vw,tw,teacher_pairs,student_pairs,margins,to,vo,max(tv,vv),margin_cutoff)
