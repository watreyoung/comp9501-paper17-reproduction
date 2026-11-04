"""Feature adaptation primitives, separate from the frozen core run.

SA: Fernando et al., ICCV 2013, M = P_s.T P_t.
TCA: Pan et al., IEEE TNN 2011, kernel trace minimization with
A.T K H K A = I. No class labels enter either estimator.
Rows are samples throughout this module.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator,eigsh


def row_zscore(X):
    X=np.asarray(X,float)
    centered=X-X.mean(axis=1,keepdims=True)
    scale=centered.std(axis=1,keepdims=True)
    return centered/np.where(scale>0,scale,1.)


@dataclass
class SubspaceAlignment:
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_basis: np.ndarray
    target_basis: np.ndarray
    alignment: np.ndarray

    def transform_source(self, X):
        return row_zscore(X)@self.source_basis@self.alignment

    def transform_target(self, X):
        return row_zscore(X)@self.target_basis


def fit_sa(source, target, dimension):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1]:
        raise ValueError('Source and target must share feature dimension')
    source=row_zscore(source);target=row_zscore(target)
    sm=source.mean(axis=0);tm=target.mean(axis=0)
    _,ss,vs=np.linalg.svd(source-sm,full_matrices=False)
    _,st,vt=np.linalg.svd(target-tm,full_matrices=False)
    rank=lambda s: int(np.count_nonzero(s>max(source.shape+target.shape)*np.finfo(float).eps*s[0])) if len(s) and s[0]>0 else 0
    if not 1<=dimension<=min(rank(ss),rank(st)):
        raise ValueError('Requested dimension exceeds centered domain rank')
    ps=vs[:dimension].T;pt=vt[:dimension].T
    return SubspaceAlignment(sm,tm,ps,pt,ps.T@pt)


@dataclass
class TransferComponents:
    coefficients: np.ndarray
    eigenvalues: np.ndarray
    constraint_error: float
    stationarity_error: float

    def transform_kernel(self, cross_kernel):
        """cross_kernel has rows=new samples, columns=fit source+target."""
        return np.asarray(cross_kernel)@self.coefficients


def fit_tca_dense_reference(kernel, n_source, dimension, mu=1., rank_tolerance=1e-10):
    """Solve TCA exactly on the nonzero range of K H K.

    Eliminating its nullspace through a Schur complement preserves the
    full regularized objective. Simply truncating to the range changes
    the optimum when the objective couples range and nullspace.
    """
    K=np.asarray(kernel,float);n=len(K)
    if K.shape!=(n,n) or not np.isfinite(K).all() or not np.allclose(K,K.T,atol=1e-10):
        raise ValueError('A finite symmetric kernel is required')
    if not 0<n_source<n or mu<=0:raise ValueError('Both domains and positive mu required')
    e=np.r_[np.full(n_source,1/n_source),np.full(n-n_source,-1/(n-n_source))]
    v=K@e;A=mu*np.outer(v,v)+np.eye(n)
    KH=K-K.mean(axis=1,keepdims=True);B=KH@KH.T
    vals,U=eigh(B);keep=vals>rank_tolerance*max(1.,vals[-1])
    if not 1<=dimension<=int(keep.sum()):raise ValueError('Requested dimension exceeds constraint rank')
    R=U[:,keep]/np.sqrt(vals[keep]);N=U[:,~keep]
    # A is SPD. Minimize over the unconstrained nullspace coordinates.
    if N.shape[1]:
        R=R-N@np.linalg.solve(N.T@A@N,N.T@A@R)
    ev,V=eigh(R.T@A@R,subset_by_index=(0,dimension-1))
    W=R@V
    constraint=float(np.max(np.abs(W.T@B@W-np.eye(dimension))))
    residual=A@W-(B@W)*ev
    stationarity=float(np.linalg.norm(residual)/max(1.,np.linalg.norm(A@W)))
    if constraint>1e-6 or stationarity>1e-6:raise ArithmeticError('TCA eigenproblem residual exceeds tolerance')
    return TransferComponents(W,ev,constraint,stationarity)


def fit_tca(kernel,n_source,dimension,mu=1.,rank_tolerance=1e-10):
    """Same full TCA pencil, solved by deterministic residual-checked ARPACK.

    A=I+mu vv.T is SPD and its inverse square root is a rank-one update.
    Leading eigenpairs of A^-1/2 B A^-1/2 avoid factoring singular B.
    All source and target samples and the full kernel remain present.
    """
    K=np.asarray(kernel,float);n=len(K)
    if K.shape!=(n,n) or not np.isfinite(K).all() or not np.allclose(K,K.T,atol=1e-10):
        raise ValueError('A finite symmetric kernel is required')
    if not 0<n_source<n or mu<=0 or not 1<=dimension<n:
        raise ValueError('Both domains, positive mu and dimension below n required')
    e=np.r_[np.full(n_source,1/n_source),np.full(n-n_source,-1/(n-n_source))]
    v=K@e;norm=float(v@v)
    a=(1-1/np.sqrt(1+mu*norm))/norm if norm>0 else 0.
    def inverse_root(x):
        return x-a*np.outer(v,v@x) if x.ndim==2 else x-a*v*(v@x)
    def constraint_product(x):
        z=K@x
        return K@(z-z.mean(axis=0))
    def product(x):return inverse_root(constraint_product(inverse_root(x)))
    op=LinearOperator((n,n),matvec=product,matmat=product,dtype=float)
    rho,U=eigsh(op,k=dimension,which='LA',tol=1e-11,maxiter=10*n,v0=np.linspace(-1,1,n))
    order=np.argsort(rho)[::-1];rho=rho[order];U=U[:,order]
    if rho[-1]<=rank_tolerance*max(1.,rho[0]):
        raise ValueError('Requested dimension exceeds constraint rank')
    W=inverse_root(U)/np.sqrt(rho);ev=1/rho;BW=constraint_product(W)
    AW=W+mu*np.outer(v,v@W)
    constraint=float(np.max(np.abs(W.T@BW-np.eye(dimension))))
    stationarity=float(np.linalg.norm(AW-BW*ev)/max(1.,np.linalg.norm(AW)))
    if constraint>1e-6 or stationarity>1e-6:
        raise ArithmeticError('TCA eigenproblem residual exceeds tolerance')
    return TransferComponents(W,ev,constraint,stationarity)
