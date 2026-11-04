"""Common event-task interface for verified nonrepresentation baselines.

No metrics or target labels are read here. C conventions remain explicit;
SVM-uniform supplies the source-only comparator for uniform-cost DASVM/STM.
"""
import time
import numpy as np
from sklearn.svm import SVC
from mil_pi.kernels import l2_normalize_columns,get_kernel,KernelSpec
from .kmm_svm_baseline import fit_kmm_svm
from .stm_baseline import fit_stm
from .rank_transfer_baseline import fit_rank_transfer
from .dasvm_baseline import fit_dasvm

METHODS=('KMM+SVM','STM','RT','DASVM','SVM-uniform')


def fit_literature(method,data,problem,*,scale=1.):
    if method not in METHODS or not np.isfinite(scale) or scale<=0:raise ValueError('Known literature method and positive finite validation scale required')
    start=time.perf_counter()
    X=l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags,axis=1)).T
    target=l2_normalize_columns(data.test_features).T
    npos=sum(b.shape[1] for b in data.positive_visual_bags)
    y=np.r_[np.ones(npos),-np.ones(len(X)-npos)]
    params=dict(scale=scale);arrays={}
    if method=='KMM+SVM':
        cost=np.r_[np.full(npos,100*scale),np.ones(len(X)-npos)]
        model,weights,diag=fit_kmm_svm(problem['K'],problem['Kst'],y,cost)
        scores=model.predict(problem['test_kernel']);arrays=dict(weights=weights,coef=model.coef,bias=np.array(model.bias))
        params.update(positive_cost=100*scale,negative_cost=1.,bound=10.)
    elif method=='STM':
        params.update(C=scale,lambda_=100.,bound=10.,epsilon=0.)
        model=fit_stm(problem['K'],problem['Kst'],y,C=scale,lambda_=100.)
        scores=model.predict(problem['test_kernel']);arrays=dict(weights=model.weights,coef=model.coef)
        diag=dict(stop_reason=model.stop_reason,objective=model.objective,history=model.history,max_constraint_violation=model.max_constraint_violation)
    elif method=='RT':
        text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1).T
        params.update(teacher_C=scale,student_C=scale,margin_cutoff=.1)
        model=fit_rank_transfer(X,text,y,teacher_C=scale,student_C=scale)
        scores=model.predict(target);arrays=dict(visual_weights=model.visual_weights,privileged_weights=model.privileged_weights,teacher_pairs=model.teacher_pairs,student_pairs=model.student_pairs,student_margins=model.student_margins)
        diag=dict(stop_reason='two_convex_stages',teacher_objective=model.teacher_objective,student_objective=model.student_objective,max_constraint_violation=model.max_constraint_violation)
    elif method=='DASVM':
        params.update(C=scale,initial_target_cost=.01*scale,tau=10,quota=1,epsilon=.001,maximum_fraction=.5,max_iter=1000)
        K,_=get_kernel(np.r_[X,target].T,spec=KernelSpec('gaussian',problem['kernel_gamma']))
        model=fit_dasvm(K,y,C=scale,initial_target_cost=.01*scale)
        scores=model.decision_function(K[len(X):]);arrays=dict(training_indices=model.training_indices,pseudo_labels=model.pseudo_labels,target_ages=model.target_ages)
        diag=dict(stop_reason=model.stop_reason,history=model.history,final_source_count=len(model.history[-1]['source_remaining']),constant_solution=model.history[-1]['single_class_constant_solution'])
    else:
        params.update(C=scale,cost_convention='uniform per source instance')
        model=SVC(C=scale,kernel='precomputed',tol=1e-7,max_iter=200000).fit(problem['K'],y)
        if model.fit_status_!=0:raise ArithmeticError('Uniform SVM failed to converge')
        scores=model.decision_function(problem['test_kernel'])
        arrays=dict(support_indices=model.support_,dual_coef=model.dual_coef_,intercept=model.intercept_)
        diag=dict(stop_reason='libsvm_converged')
    if not np.isfinite(scores).all():raise ArithmeticError('Nonfinite literature predictions')
    diag.update(seconds=time.perf_counter()-start,parameters=params,kernel_gamma=problem['kernel_gamma'],target_labels_used=False)
    return scores,diag,arrays
