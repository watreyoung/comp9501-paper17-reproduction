"""Two-view kernel baseline primitives; not yet a frozen experiment protocol."""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp


def kernel_features(K):
    K=np.asarray(K,float)
    if K.ndim!=2 or K.shape[0]!=K.shape[1] or not np.allclose(K,K.T,atol=1e-10):
        raise ValueError('Symmetric square Gram matrix required')
    v,U=np.linalg.eigh(K);tol=1e-10*max(1.,v[-1])
    if v[0]<-tol:raise ValueError('Kernel is not positive semidefinite')
    keep=v>tol
    if not keep.any():raise ValueError('Zero-rank kernel')
    return U[:,keep]*np.sqrt(v[keep]),U[:,keep]/np.sqrt(v[keep])


@dataclass
class TwoViewSVM:
    visual_coef: np.ndarray
    visual_bias: float
    text_coef: np.ndarray
    text_bias: float
    objective: float
    agreement_loss: float
    constraint_violation: float

    def predict_visual(self,cross_kernel):
        return np.asarray(cross_kernel)@self.visual_coef+self.visual_bias


def fit_svm2k(K_visual,K_text,y,c_visual,c_text,*,coupling=1.,epsilon=.1):
    """RKHS primal: two SVM losses plus epsilon-insensitive agreement.

    min .5(||w_v||²+||w_t||²)+c_v'xi_v+c_t'xi_t+D sum eta,
    y f_v>=1-xi_v, y f_t>=1-xi_t, |f_v-f_t|<=epsilon+eta.
    Only the visual classifier is used for held-out prediction.
    """
    if coupling<0 or epsilon<0:raise ValueError('Nonnegative agreement parameters required')
    y=np.asarray(y,float);n=len(y)
    if not np.isin(y,[-1,1]).all() or len(np.unique(y))!=2:raise ValueError('Binary labels required')
    Fv,Rv=kernel_features(K_visual);Ft,Rt=kernel_features(K_text)
    if len(Fv)!=n or len(Ft)!=n:raise ValueError('Unpaired training kernels')
    cv=np.broadcast_to(np.asarray(c_visual,float),(n,));ct=np.broadcast_to(np.asarray(c_text,float),(n,))
    if np.any(cv<=0) or np.any(ct<=0):raise ValueError('Positive instance costs required')
    wv=cp.Variable(Fv.shape[1]);wt=cp.Variable(Ft.shape[1]);bv=cp.Variable();bt=cp.Variable()
    xv=cp.Variable(n,nonneg=True);xt=cp.Variable(n,nonneg=True);eta=cp.Variable(n,nonneg=True)
    fv=Fv@wv+bv;ft=Ft@wt+bt
    constraints=[cp.multiply(y,fv)>=1-xv,cp.multiply(y,ft)>=1-xt,
                 fv-ft<=epsilon+eta,ft-fv<=epsilon+eta]
    objective=.5*(cp.sum_squares(wv)+cp.sum_squares(wt))+cv@xv+ct@xt+coupling*cp.sum(eta)
    problem=cp.Problem(cp.Minimize(objective),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,max_iter=300)
    if problem.status!='optimal':raise ArithmeticError('SVM-2K solve failed: '+problem.status)
    violation=max(float(np.max(np.abs(c.violation()))) for c in constraints)
    if violation>1e-6:raise ArithmeticError('SVM-2K constraints violated')
    agreement=float(np.maximum(np.abs(fv.value-ft.value)-epsilon,0).sum())
    return TwoViewSVM(Rv@wv.value,float(bv.value),Rt@wt.value,float(bt.value),
                      float(problem.value),agreement,violation)


@dataclass
class KernelCCA:
    visual_coef: np.ndarray
    text_coef: np.ndarray
    visual_column_mean: np.ndarray
    visual_grand_mean: float
    canonical_correlations: np.ndarray

    def transform_visual(self,cross_kernel):
        K=np.asarray(cross_kernel,float)
        centered=K-K.mean(axis=1,keepdims=True)-self.visual_column_mean+self.visual_grand_mean
        return centered@self.visual_coef


def fit_kcca(K_visual,K_text,dimension,regularization=1.):
    """Hardoon et al. regularized KCCA: alpha'(K²+kappa K)alpha=1.

    Center both training kernels, use exact empirical RKHS features, then
    whiten the regularized within-view covariances and perform an SVD.
    This avoids inverting singular Gram matrices.
    """
    if regularization<=0:raise ValueError('Positive KCCA regularization required')
    Kv=np.asarray(K_visual,float);Kt=np.asarray(K_text,float)
    if Kv.shape!=Kt.shape:raise ValueError('Paired kernels required')
    def center(K):return K-K.mean(axis=0)-K.mean(axis=1,keepdims=True)+K.mean()
    Fv,Rv=kernel_features(center(Kv));Ft,Rt=kernel_features(center(Kt))
    if not 1<=dimension<=min(Fv.shape[1],Ft.shape[1]):raise ValueError('KCCA dimension exceeds view rank')
    def inverse_root(C):
        v,U=np.linalg.eigh(C)
        return (U/np.sqrt(v))@U.T
    Av=inverse_root(Fv.T@Fv+regularization*np.eye(Fv.shape[1]))
    At=inverse_root(Ft.T@Ft+regularization*np.eye(Ft.shape[1]))
    U,s,Vt=np.linalg.svd(Av@(Fv.T@Ft)@At,full_matrices=False)
    return KernelCCA(Rv@Av@U[:,:dimension],Rt@At@Vt[:dimension].T,
                     Kv.mean(axis=0),float(Kv.mean()),s[:dimension])
