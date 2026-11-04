"""Equation-based controls for IJCV 2016 §§3–4, independent of MATLAB demos.

sMIL: Eq.8, mi-SVM: Eq.9 + Algorithm1, CPB: Eq.15 + Algorithms2/3.
No-PI controls use independent hinge slack, not a zero text-kernel surrogate.
DA uses explicit alpha<=C2*theta with MMD mu=(C2*n)^2/lambda.
"""
from dataclasses import dataclass,field
import numpy as np
from scipy.optimize import minimize
from .qp import solve_qp

@dataclass
class PaperModel:
    coef: np.ndarray
    bias: float
    alpha: np.ndarray
    objective: float
    labels: np.ndarray|None=None
    history: list=field(default_factory=list)
    stop_reason: str='single_qp'
    diagnostics: dict=field(default_factory=dict)

    def predict(self,K):return np.asarray(K)@self.coef+self.bias


def dual_problem(K,T,y,margins,c,pi_mask,*,gamma=10.,C2=None,lam=100.,Kst=None,
                 label_kernel=None,regularized_bias=False):
    """Min negative dual; beta only on PI subset, r=C2*theta for DA.

    c_i is a per-example cost. Outside PI, alpha_i<=c_i. Inside PI,
    sum(alpha_P+beta-c_P)=0. For CPB label_kernel already includes Q=K+1.
    Returns the exact QP model and its equality multiplier (the free bias).
    """
    K=np.asarray(K,float);T=np.asarray(T,float);y=np.asarray(y,float)
    c=np.asarray(c,float);margins=np.asarray(margins,float);P=np.asarray(pi_mask,bool)
    n=len(y);idx=np.flatnonzero(P);nb=len(idx);da=C2 is not None
    if gamma<=0 or np.any(c<0) or (da and (C2<=0 or lam<=0)):raise ValueError('positive costs/regularizers required')
    S=np.eye(n)[:,idx];size=n+nb+(n if da else 0)
    H=np.zeros((size,size));f=np.zeros(size);f[:n]=-margins
    kernel=K*np.outer(y,y) if label_kernel is None else np.asarray(label_kernel)
    H[:n,:n]=kernel
    constant=0.
    if nb:
        A=np.c_[np.eye(n),S];TP=T*np.outer(P,P)
        H[:n+nb,:n+nb]+=A.T@TP@A/gamma
        f[:n+nb]-=A.T@TP@np.where(P,c,0.)/gamma
        constant=.5/gamma*np.where(P,c,0.)@TP@np.where(P,c,0.)
    eq=[];rhs=[]
    if not regularized_bias:
        e=np.zeros(size);e[:n]=y;eq.append(e);rhs.append(0.)
    if nb:
        e=np.zeros(size);e[:n]=P;e[n:n+nb]=1;eq.append(e);rhs.append(float(c[P].sum()))
    upper=np.r_[np.where(P,np.inf,c),np.full(nb,np.inf),np.full(n,np.inf) if da else []]
    ub=None;ubrhs=None
    if da:
        if Kst is None or Kst.shape[0]!=n:raise ValueError('matching source-target kernel required')
        H[-n:,-n:]=K/lam
        f[-n:]=-(n*C2/lam)*np.mean(Kst,axis=1)
        ub=np.zeros((n,size));ub[:,:n]=np.eye(n);ub[:,-n:]=-np.eye(n);ubrhs=np.zeros(n)
        e=np.zeros(size);e[-n:]=1;eq.append(e);rhs.append(n*C2)
    # Same declared tiny stabiliser in all no-PI/PI/DA controls.
    H+=1e-8*np.eye(size)
    r=solve_qp(H,f,A_eq=np.asarray(eq) if eq else None,b_eq=np.asarray(rhs) if eq else None,
               A_ub=ub,b_ub=ubrhs,lb=np.zeros(size),ub=upper,max_iter=200000,
               name='paper equation dual')
    bias=0. if regularized_bias else float(r.equality_multipliers[0])
    alpha=r.x[:n]
    model=PaperModel(coef=alpha*y,bias=bias,alpha=alpha,objective=-float(r.objective+constant))
    model.diagnostics=dict(source_weights=(r.x[-n:]/C2).tolist() if da else [],
                           dual_variables=r.x.tolist(),constraint_violation=r.constraint_violation)
    return model


def feasible_update(scores,n_positive,bag_size,sigma):
    """Algorithm1/4: predict by sign, repair only violating positive bags."""
    y=np.r_[np.where(np.asarray(scores[:n_positive])>0,1.,-1.),-np.ones(len(scores)-n_positive)]
    required=int(np.ceil(sigma*bag_size))
    for start in range(0,n_positive,bag_size):
        ix=np.arange(start,start+bag_size)
        if np.count_nonzero(y[ix]>0)<required:
            order=np.argsort(-np.asarray(scores)[ix],kind='stable');y[ix[order[:required]]]=1.
    return y


def train_mi(K,T,n_positive,bag_size,*,c,pi='none',gamma=10.,sigma=.6,C2=None,lam=100.,Kst=None,max_iter=100):
    n=len(K);y=np.r_[np.ones(n_positive),-np.ones(n-n_positive)]
    mask=np.zeros(n,bool) if pi=='none' else np.ones(n,bool) if pi=='full' else np.arange(n)<n_positive
    history=[];seen=set()
    for iteration in range(max_iter):
        model=dual_problem(K,T,y,np.ones(n),c,mask,gamma=gamma,C2=C2,lam=lam,Kst=Kst)
        model.labels=y.copy();new=feasible_update(model.predict(K),n_positive,bag_size,sigma)
        history.append(dict(iteration=iteration+1,objective=model.objective,label_changes=int(np.count_nonzero(new!=y))))
        if np.array_equal(new,y):reason='labels_stable';break
        key=new.tobytes()
        if key in seen:reason='label_cycle';break
        if iteration+1==max_iter:reason='max_iter';break
        seen.add(y.tobytes());y=new
    model.history=history;model.stop_reason=reason
    return model


def bag_patterns(bag_size,minimum):
    bits=((np.arange(2**bag_size)[:,None]>>np.arange(bag_size))&1)
    return (2*bits[bits.sum(axis=1)>=minimum]-1).T.astype(float)


def bag_oracle(Q,alpha,n_positive,bag_size,sigma,*,max_passes=100,tol=1e-12):
    """Paper Algorithm2, exact within each bag, approximate over bags.

    Strict score improvement preserves current labels on ties, avoiding cycles.
    """
    M=Q*np.outer(alpha,alpha);y=np.r_[np.ones(n_positive),-np.ones(len(Q)-n_positive)]
    patterns=bag_patterns(bag_size,int(np.ceil(sigma*bag_size)))
    reason='max_passes'
    for step in range(max_passes):
        changed=False
        for start in range(0,n_positive,bag_size):
            ix=np.arange(start,start+bag_size);other=np.setdiff1d(np.arange(len(Q)),ix)
            block=M[np.ix_(ix,ix)];cross=M[np.ix_(ix,other)]@y[other]
            scores=np.einsum('ik,ij,jk->k',patterns,block,patterns)+2*patterns.T@cross
            current=y[ix]@block@y[ix]+2*y[ix]@cross;best=int(np.argmax(scores))
            if scores[best]>current+tol*max(1.,abs(current)):
                y[ix]=patterns[:,best];changed=True
        if not changed:reason='coordinate_stable';break
    return y,dict(passes=step+1,stop_reason=reason,score=float(y@M@y),global_optimum_claimed=False)


def train_cpb_nested_reference(K,T,n_positive,bag_size,*,c,pi='none',gamma=10.,sigma=.6,C2=None,lam=100.,Kst=None,max_iter=100,mkl_max_iter=100,tol=1e-5):
    n=len(K);Q=K+1.;initial=np.r_[np.ones(n_positive),-np.ones(n-n_positive)]
    Y=initial[:,None];history=[];d=np.ones(1)
    mask=np.zeros(n,bool) if pi=='none' else np.ones(n,bool) if pi=='full' else np.arange(n)<n_positive
    for outer in range(max_iter):
        last={};evaluations=[]
        def evaluate(weights):
            if 'd' in last and np.array_equal(weights,last['d']):return last['value'],last['gradient']
            mixed=Q*((Y*weights)@Y.T)
            model=dual_problem(K,T,np.ones(n),np.ones(n),c,mask,gamma=gamma,C2=C2,lam=lam,Kst=Kst,
                               label_kernel=mixed,regularized_bias=True)
            scores=np.einsum('it,ij,jt->t',Y,(Q*np.outer(model.alpha,model.alpha)),Y)
            last.update(d=weights.copy(),model=model,value=model.objective,gradient=-.5*scores,q=scores)
            evaluations.append(model.objective)
            return model.objective,-.5*scores
        if Y.shape[1]>1:
            opt=minimize(evaluate,d,jac=True,method='SLSQP',bounds=[(0.,1.)]*len(d),
                         constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:np.ones(len(w)))],
                         options=dict(maxiter=mkl_max_iter,ftol=1e-9))
            if not opt.success:raise RuntimeError(f'MKL failed: {opt.message}')
            d=np.clip(opt.x,0.,1.);d/=d.sum();evaluate(d);mkl_reason='simplex_optimum';mkl_iterations=int(opt.nit)
        else:evaluate(d);mkl_reason='single_kernel';mkl_iterations=1
        model=last['model'];effective=Y@d;model.coef=model.alpha*effective;model.bias=float(model.coef.sum())
        new,oracle=bag_oracle(Q,model.alpha,n_positive,bag_size,sigma)
        qnew=oracle['score'];violation=.5*(qnew-max(last['q']))
        history.append(dict(iteration=outer+1,objective=model.objective,kernels=Y.shape[1],
                            violation=violation,mkl_iterations=mkl_iterations,mkl_stop_reason=mkl_reason,
                            oracle=oracle))
        if violation<=tol*max(1.,abs(max(last['q']))):reason='no_violated_coordinate_cut';break
        if any(np.array_equal(new,Y[:,j]) for j in range(Y.shape[1])):reason='duplicate_cut';break
        if outer+1==max_iter:reason='max_iter';break
        Y=np.c_[Y,new];d=np.r_[d,0.]
    model.labels=Y@d;model.history=history;model.stop_reason=reason
    model.diagnostics.update(kernel_weights=d.tolist(),label_vectors=Y.tolist(),
                             bias_convention='regularized intercept, Q=K+1; no alpha-y equality')
    return model


def restricted_cpb(Q,T,Y,c,mask,*,gamma=10.,C2=None,lam=100.,Ksource=None,Kst=None):
    """Eq.16 QCQP, with alpha/delta/r scaled to avoid tiny DA coefficients.

    delta=alpha_P+beta-c_P removes the large correcting-function offset.
    Cut multipliers sum to the objective scale; normalisation recovers d.
    """
    import cvxpy as cp
    n=len(Q);P=np.asarray(mask,bool);idx=np.flatnonzero(P)
    positive=c[:n//2][c[:n//2]>1e-12]
    unit=max(1e-8,min(1.,float(np.median(positive)) if positive.size else 1.,C2 if C2 is not None else 1.))
    a=cp.Variable(n,nonneg=True);tau=cp.Variable();constraints=[]
    cost=-cp.sum(a)+unit*tau+5e-9*unit*cp.sum_squares(a)
    if np.any(~P):constraints.append(a[~P]<=c[~P]/unit)
    delta=None;r=None
    if len(idx):
        delta=cp.Variable(len(idx))
        constraints += [delta>=a[idx]-c[idx]/unit,cp.sum(delta)==0]
        cost+=unit*cp.quad_form(delta,cp.psd_wrap(T[np.ix_(idx,idx)]))/(2*gamma)
        beta=c[idx]+unit*(delta-a[idx])
        cost+=5e-9/unit*cp.sum_squares(beta)
    if C2 is not None:
        r=cp.Variable(n,nonneg=True)
        constraints += [a<=r,cp.sum(r)==n*C2/unit]
        cost+=unit*cp.quad_form(r,cp.psd_wrap(Ksource))/(2*lam)-(n*C2/lam)*np.mean(Kst,axis=1)@r+5e-9*unit*cp.sum_squares(r)
    cuts=[]
    for j in range(Y.shape[1]):
        cut=.5*cp.quad_form(a,cp.psd_wrap(Q*np.outer(Y[:,j],Y[:,j])))<=tau
        cuts.append(cut);constraints.append(cut)
    problem=cp.Problem(cp.Minimize(cost),constraints)
    attempts=[]
    try:
        problem.solve(solver='CLARABEL',max_iter=300,tol_gap_abs=1e-7,tol_gap_rel=1e-7,tol_feas=1e-7)
        attempts.append(dict(solver='CLARABEL',status=problem.status))
    except cp.error.SolverError as error:
        attempts.append(dict(solver='CLARABEL',status='error',error=str(error)))
    if not attempts or attempts[-1]['status']!='optimal':
        problem.solve(solver='SCS',eps=1e-8,max_iters=200000,acceleration_lookback=10)
        attempts.append(dict(solver='SCS',status=problem.status))
    if problem.status!='optimal':raise RuntimeError('CPB conic solver: '+problem.status)
    alpha=unit*np.asarray(a.value).ravel();d=np.array([float(cut.dual_value)/unit for cut in cuts])
    if np.min(d)<-1e-7 or abs(d.sum()-1)>.01:raise RuntimeError(f'invalid MKL simplex duals: sum={d.sum()}, min={d.min()}')
    raw_simplex_sum=float(d.sum())
    d=np.maximum(d,0.);d/=d.sum()
    q=np.einsum('it,ij,jt->t',Y,Q*np.outer(alpha,alpha),Y)
    residual=max(float(np.max(-alpha)),float(np.max(.5*q-unit**2*float(tau.value))),0.)
    if np.any(~P):residual=max(residual,float(np.max(alpha[~P]-c[~P])))
    if len(idx):residual=max(residual,unit*abs(float(np.sum(delta.value))),float(np.max(alpha[idx]-c[idx]-unit*delta.value)))
    if r is not None:residual=max(residual,float(np.max(alpha-unit*r.value)),abs(float(unit*np.sum(r.value)-n*C2)))
    if residual>1e-6:raise RuntimeError(f'CPB conic constraints violated: {residual}')
    coef=alpha*(Y@d)
    model=PaperModel(coef=coef,bias=float(coef.sum()),alpha=alpha,objective=-float(problem.value)*unit)
    model.diagnostics=dict(source_weights=[] if r is None else (unit*np.asarray(r.value)/C2).tolist(),
        kernel_weights=d.tolist(),label_vectors=Y.tolist(),constraint_violation=residual,
        bias_convention='regularized intercept Q=K+1; no alpha-y equality')
    diagnostic=dict(solver=problem.solver_stats.solver_name,attempts=attempts,status=problem.status,iterations=int(problem.solver_stats.num_iters),
                    objective=model.objective,constraint_violation=residual,kernels=Y.shape[1],
                    simplex_sum=float(d.sum()),raw_simplex_sum=raw_simplex_sum,variable_unit=unit,
                    cut_complementarity_gap=float(.5*(max(q)-d@q)))
    return model,d,q,diagnostic


def train_cpb(K,T,n_positive,bag_size,*,c,pi='none',gamma=10.,sigma=.6,C2=None,lam=100.,Kst=None,max_iter=100,mkl_max_iter=100,tol=1e-5):
    n=len(K);Q=K+1.;Y=np.r_[np.ones(n_positive),-np.ones(n-n_positive)][:,None]
    mask=np.zeros(n,bool) if pi=='none' else np.ones(n,bool) if pi=='full' else np.arange(n)<n_positive
    history=[];conic=[]
    for iteration in range(max_iter):
        model,d,q,diag=restricted_cpb(Q,T,Y,c,mask,gamma=gamma,C2=C2,lam=lam,Ksource=K,Kst=Kst)
        conic.append(diag)
        new,oracle=bag_oracle(Q,model.alpha,n_positive,bag_size,sigma)
        violation=.5*(oracle['score']-max(q))
        history.append(dict(iteration=iteration+1,objective=model.objective,kernels=Y.shape[1],
                            violation=violation,mkl_iterations=diag['iterations'],mkl_stop_reason='conic_optimal',oracle=oracle))
        if len(history)>1 and history[-2]['objective']-model.objective <= .001*max(abs(history[-2]['objective']),1e-12):
            reason='objective_tolerance';break
        if violation<=tol*max(1.,abs(max(q))):reason='no_violated_coordinate_cut';break
        if any(np.array_equal(new,Y[:,j]) for j in range(Y.shape[1])):reason='duplicate_cut';break
        if iteration+1==max_iter:reason='max_iter';break
        Y=np.c_[Y,new]
    model.labels=Y@d;model.history=history;model.stop_reason=reason
    model.diagnostics['conic_diagnostics']=conic
    return model
