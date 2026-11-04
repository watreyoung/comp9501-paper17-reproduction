"""Dispatch the frozen all-family paper controls, with a shared mathematical QP."""
import time
import numpy as np
from mil_pi.paper_models import dual_problem,train_mi,train_cpb
from mil_pi.kernels import shrink_kernel,shrink_row
from mil_pi.qp import qp_session
from .alignment_models import kmm_weights

FAMILIES=['sMIL','mi-SVM','MIL-CPB']
CORE=[f+s for f in FAMILIES for s in ['', '-PI','-PI-DA']]
CONTROLS=[f+s for f in FAMILIES for s in ['-PI-full','-PI-shuffled']]
KMM=['KMM+'+f+'-PI' for f in FAMILIES]
METHODS=['SVM','pSVM+']+CORE+CONTROLS+KMM


def family_of(method):
    for f in FAMILIES:
        if f in method:return f
    return method


def parameters(method,scale):
    f=family_of(method);instance=f in {'mi-SVM','MIL-CPB'}
    return dict(C1=(.01 if instance else 100.)*scale,gamma=10. if instance else 100.,
                C2=1e-5 if instance else 10.,lambda_=100.,sigma=.6,
                max_iter=100,mkl_max_iter=100)


def fit(method,p,params,*,seed=0):
    start=time.perf_counter();f=family_of(method)
    K,T=p['K'],p['full_tK'];b,m=p['n_bags'],p['bag_size'];nplus=b*m
    pi='full' if method.endswith('-full') else 'partial' if 'PI' in method or method=='pSVM+' else 'none'
    da=method.endswith('-DA')
    if method.endswith('-shuffled'):
        order=np.r_[np.random.default_rng(seed).permutation(nplus),np.arange(nplus,len(K))]
        T=T[np.ix_(order,order)]
    weights=np.ones(len(K))
    with qp_session('osqp') as trace:
        if method.startswith('KMM+'):
            weights=kmm_weights(K,p['Kst'])
        if f=='sMIL':
            kv=shrink_kernel(K,[m]*b);tk=shrink_kernel(T,[m]*b);kst=shrink_row(p['Kst'],[m]*b)
            y=np.r_[np.ones(b),-np.ones(nplus)]
            margins=np.r_[np.full(b,2*params['sigma']-1),np.ones(nplus)]
            c=np.r_[np.full(b,params['C1']),np.ones(nplus)]*shrink_row(weights[:,None],[m]*b).ravel()
            mask=np.ones(len(y),bool) if pi=='full' else y>0 if pi=='partial' else np.zeros(len(y),bool)
            model=dual_problem(kv,tk,y,margins,c,mask,gamma=params['gamma'],C2=params['C2'] if da else None,
                               lam=params['lambda_'],Kst=kst)
            model.coef=np.r_[np.repeat(model.coef[:b]/m,m),model.coef[b:]]
        else:
            y=np.r_[np.ones(nplus),-np.ones(nplus)]
            c=np.r_[np.full(nplus,params['C1']),np.ones(nplus)]*weights
            if f=='mi-SVM':
                model=train_mi(K,T,nplus,m,c=c,pi=pi,gamma=params['gamma'],sigma=params['sigma'],
                               C2=params['C2'] if da else None,lam=params['lambda_'],Kst=p['Kst'],max_iter=params['max_iter'])
            elif f=='MIL-CPB':
                model=train_cpb(K,T,nplus,m,c=c,pi=pi,gamma=params['gamma'],sigma=params['sigma'],
                               C2=params['C2'] if da else None,lam=params['lambda_'],Kst=p['Kst'],
                               max_iter=params['max_iter'],mkl_max_iter=params['mkl_max_iter'])
            else:
                mask=np.arange(len(y))<nplus if pi=='partial' else np.zeros(len(y),bool)
                model=dual_problem(K,T,y,np.ones(len(y)),c,mask,gamma=params['gamma'])
        scores=model.predict(p['test_kernel'])
    if not np.isfinite(scores).all():raise ValueError('nonfinite prediction')
    diagnostics=dict(seconds=time.perf_counter()-start,objective=model.objective,
        stop_reason=model.stop_reason,outer_iterations=len(model.history),history=model.history,
        model_diagnostics=model.diagnostics,qp_diagnostics=trace,qp_solves=len(trace),
        final_labels=None if model.labels is None else model.labels.tolist(),
        max_constraint_violation=max([r['constraint_violation'] for r in trace]+[r['constraint_violation'] for r in model.diagnostics.get('conic_diagnostics',[])],default=0.),
        kernel_gamma=p['kernel_gamma'])
    return scores,diagnostics
