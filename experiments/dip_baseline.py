"""DIP Eq.4/7 (ICCV 2013), Gaussian MMD on a Grassmann manifold.

No class-clustering regularizer (DIP-CC is a distinct variant). Features are
rows; bandwidth is the median squared distance of source samples before
projection. PCA preserves the numerical rank of joint unlabeled data.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist,cdist


def gaussian_mmd_value_gradient(X,W,n_source,sigma):
    """Biased empirical MMD² and its ambient derivative w.r.t. W."""
    X=np.asarray(X,float);Z=X@W;n=len(X)
    e=np.r_[np.full(n_source,1/n_source),np.full(n-n_source,-1/(n-n_source))]
    D=cdist(Z,Z,'sqeuclidean');Q=np.exp(-D/sigma)*np.outer(e,e)
    value=float(Q.sum())
    # sum_ij Qij (xi-xj)(xi-xj)'W = 2 X'(diag(Q1)-Q) XW.
    gradient=(-4/sigma)*X.T@(Q.sum(axis=1)[:,None]*Z-Q@Z)
    return value,gradient


def grassmann_step(W,H,step):
    """Exact Grassmann exponential and parallel transport along its geodesic."""
    U,s,Vt=np.linalg.svd(H,full_matrices=False);V=Vt.T
    cosine=np.cos(step*s);sine=np.sin(step*s);WV=W@V
    point=(WV*cosine+U*sine)@Vt
    correction=-WV*sine+U*(cosine-1)
    def transport(Z):return Z+correction@(U.T@Z)
    return point,transport


@dataclass
class DomainInvariantProjection:
    joint_mean: np.ndarray
    pca_basis: np.ndarray
    projection: np.ndarray
    sigma: float
    history: list
    stop_reason: str
    orthogonality_error: float

    def transform(self,X):
        return (np.asarray(X)-self.joint_mean)@self.pca_basis@self.projection


def fit_dip(source,target,dimension,*,max_iter=100,gradient_tolerance=1e-6,relative_tolerance=1e-9):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1]:raise ValueError('Shared feature dimension required')
    if min(len(source),len(target))<2:raise ValueError('Both domains require at least two samples')
    distances=pdist(source,'sqeuclidean');sigma=float(np.median(distances))
    if not sigma>0:raise ValueError('Nonzero source distance bandwidth required')
    X=np.r_[source,target];mean=X.mean(axis=0);X=X-mean
    _,s,Vt=np.linalg.svd(X,full_matrices=False)
    rank=int(np.count_nonzero(s>max(X.shape)*np.finfo(float).eps*s[0]))
    if not 1<=dimension<=rank:raise ValueError('DIP dimension exceeds joint PCA rank')
    basis=Vt[:rank].T;X=X@basis;W=np.eye(rank,dimension)
    value,ambient=gaussian_mmd_value_gradient(X,W,len(source),sigma)
    gradient=ambient-W@(W.T@ambient);H=-gradient;history=[];reason='max_iter'
    for iteration in range(max_iter):
        norm=float(np.linalg.norm(gradient))
        if norm<=gradient_tolerance:
            reason='gradient_tolerance';break
        derivative=float(np.sum(gradient*H))
        if derivative>=0:H=-gradient;derivative=-norm**2
        step=1/max(float(np.linalg.norm(H)),1e-12);accepted=False
        for backtrack in range(40):
            candidate,transport=grassmann_step(W,H,step)
            newvalue,newambient=gaussian_mmd_value_gradient(X,candidate,len(source),sigma)
            if newvalue<=value+1e-4*step*derivative:
                accepted=True;break
            step*=.5
        if not accepted:
            reason='line_search_failed';break
        newgradient=newambient-candidate@(candidate.T@newambient)
        transported_gradient=transport(gradient);transported_H=transport(H)
        beta=max(0.,float(np.sum(newgradient*(newgradient-transported_gradient)))/norm**2)
        newH=-newgradient+beta*transported_H
        change=value-newvalue
        history.append(dict(iteration=iteration+1,objective=newvalue,previous_objective=value,gradient_norm=float(np.linalg.norm(newgradient)),step=step,backtracks=backtrack,beta=beta))
        W=candidate;value=newvalue;gradient=newgradient;H=newH
        if change<=relative_tolerance*max(1.,abs(value)):
            reason='objective_tolerance';break
    error=float(np.max(np.abs(W.T@W-np.eye(dimension))))
    if error>1e-8:raise ArithmeticError('DIP orthogonality lost')
    return DomainInvariantProjection(mean,basis,W,sigma,history,reason,error)
