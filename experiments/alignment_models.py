"""Explicit soft-margin / SVM+ ablations and KMM for paper-question alignment.

Focus on the sMIL family; do not claim to reproduce all original baselines.
The final intercept is omitted because all reported metrics rank each class.
"""
import numpy as np
from mil_pi.algorithms import KernelModel, s_mil_pi_da
from mil_pi.kernels import shrink_kernel, shrink_row
from mil_pi.qp import solve_qp, qp_session

METHODS=['SVM','sMIL','pSVM+','sMIL-PI','sMIL-PI-full','KMM+sMIL-PI','sMIL-PI-DA','sMIL-PI-shuffled']


def correcting_svm(K,T,y,margins,c,pi_mask,*,gamma=100.,negative_caps=None):
    """Dual of min .5||w||² + gamma/2||u||² + sum c_i*xi_i.

    PI subset P: xi_i=<u,t_i>+b*, xi_i>=0. Outside P: ordinary
    independent soft-margin slack, so 0<=alpha_i<=c_i. The correction
    intercept yields sum_{P}(alpha_i+beta_i-c_i)=0; y'alpha=0.
    """
    n=len(y);pi_mask=np.asarray(pi_mask,dtype=bool);idx=np.flatnonzero(pi_mask)
    if not idx.size:
        r=solve_qp(K*np.outer(y,y)+1e-6*np.eye(n),-margins,A_eq=y[None],b_eq=[0.],
                   lb=np.zeros(n),ub=c,max_iter=100000,name='aligned soft-margin SVM')
        return KernelModel(coef=r.x*y,objective=r.objective)
    # beta exists only on PI examples. Embedding S maps these into n rows.
    S=np.eye(n)[:,idx];TP=T*np.outer(pi_mask,pi_mask)
    A=np.c_[np.eye(n),S]
    H=A.T@TP@A/gamma
    H[:n,:n]+=K*np.outer(y,y)
    H+=1e-6*np.eye(n+len(idx))
    f=-A.T@TP@np.where(pi_mask,c,0.)/gamma
    f[:n]-=margins
    eq=np.zeros((2,n+len(idx)));eq[0,:n]=y
    eq[1,:n]=pi_mask;eq[1,n:]=1.
    upper=np.r_[np.where(pi_mask,np.inf,c),np.full(len(idx),np.inf)]
    x0=np.r_[np.zeros(n),c[idx]]
    r=solve_qp(H,f,A_eq=eq,b_eq=[0.,c[idx].sum()],lb=np.zeros(len(f)),ub=upper,
               x0=x0,max_iter=100000,name='aligned privileged SVM dual')
    return KernelModel(coef=r.x[:n]*y,objective=r.objective)


def kmm_weights(K,Kst,*,bound=10.):
    """KMM: min .5*w'K*w - ns/nt * 1'Kts*w, sum w=ns, 0<=w<=B."""
    n=len(K);r=solve_qp(K+1e-6*np.eye(n),-n*Kst.mean(axis=1),
        A_eq=np.ones((1,n)),b_eq=[float(n)],lb=np.zeros(n),ub=np.full(n,bound),
        x0=np.ones(n),max_iter=100000,name='KMM source weights')
    # OSQP may return tiny negative values within feasibility tolerance.
    # Project only those numerical bound violations before using loss costs.
    return np.clip(r.x,0.,bound)


def aligned_predict(method,problem,params,*,seed=0):
    b,m=problem['n_bags'],problem['bag_size'];n=b*m
    K,T=problem['K'],problem['full_tK']
    cvalue=params['C'];gamma=params['gamma'];sigma=params.get('sigma',.6)
    with qp_session('osqp') as trace:
        if method=='sMIL-PI-DA':
            p=dict(svm_C1=cvalue,svm_C2=params['C2'],gamma1=gamma,
                   gamma2=params['lambda'],rho=sigma,B=1e10,epsilon=0.,solver_max_iter=100000)
            model=s_mil_pi_da(K,problem['tK'],K,problem['Kst'],[m]*b,[m]*b,p)
        else:
            grouped=method not in {'SVM','pSVM+'}
            if grouped:
                kv=shrink_kernel(K,[m]*b);tt=shrink_kernel(T,[m]*b)
                y=np.r_[np.ones(b),-np.ones(n)]
                margins=np.r_[np.full(b,2*sigma-1),np.ones(n)]
                # Match the existing paper-style PI negative cost (=1), rather
                # than compare methods with different negative-loss scaling.
                c=np.r_[np.full(b,cvalue/b),np.ones(n)]
                if method=='KMM+sMIL-PI':
                    weights=kmm_weights(K,problem['Kst'])
                    bag_weights=shrink_row(weights[:,None],[m]*b).ravel()
                    c*=bag_weights
                mask=y>0 if method not in {'sMIL','sMIL-PI-full'} else (np.ones(len(y),bool) if method=='sMIL-PI-full' else np.zeros(len(y),bool))
                if method=='sMIL-PI-shuffled':
                    order=np.r_[np.random.default_rng(seed).permutation(n),np.arange(n,2*n)]
                    tt=shrink_kernel(T[np.ix_(order,order)],[m]*b)
            else:
                kv,tt=K,T;y=np.r_[np.ones(n),-np.ones(n)];margins=np.ones(2*n)
                c=np.r_[np.full(n,cvalue/n),np.ones(n)]
                mask=y>0 if method=='pSVM+' else np.zeros(2*n,bool)
            model=correcting_svm(kv,tt,y,margins,c,mask,gamma=gamma)
            if grouped:model.coef=np.r_[np.repeat(model.coef[:b]/m,m),model.coef[b:]]
        scores=model.decision_function(problem['test_kernel'])
    if not np.isfinite(scores).all():raise ValueError('nonfinite scores')
    return scores,dict(qp_diagnostics=trace,qp_solves=len(trace),objective=model.objective,
                       max_constraint_violation=max(r['constraint_violation'] for r in trace))
