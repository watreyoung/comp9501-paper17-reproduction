"""Shared feature-adaptation dispatch and validation-only MIL-PI selection.

Adds missing heads without modifying previously frozen experiment sources.
SGF is intentionally absent until its original complete procedure is verified.
"""
import numpy as np
from .feature_adaptation import fit_sa,fit_tca
from .geodesic_baselines import fit_gfk
from .dip_baseline import fit_dip
from .baseline_extension import adapted_problem
from .core_models import fit,parameters
from mil_pi.kernels import l2_normalize_columns,get_kernel,KernelSpec

HEADS=('SVM','sMIL-PI','mi-SVM-PI','MIL-CPB-PI')


def feature_adaptation(name,data,problem,*,dimension=10,tca_mu=.1):
    X=l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags,axis=1)).T
    target=l2_normalize_columns(data.test_features).T
    diag=dict(method=name,dimension=dimension,uses_target_features=True,uses_target_labels=False)
    if name=='SA':
        model=fit_sa(X,target,dimension)
        source_new=model.transform_source(X);target_new=model.transform_target(target)
        arrays=dict(source_basis=model.source_basis,target_basis=model.target_basis,alignment=model.alignment)
    elif name=='TCA':
        Z=np.r_[X,target];K,_=get_kernel(Z.T,spec=KernelSpec('gaussian',problem['kernel_gamma']))
        model=fit_tca(K,len(X),dimension,tca_mu);z=model.transform_kernel(K)
        source_new=z[:len(X)];target_new=z[len(X):]
        diag.update(mu_internal=tca_mu,mu_paper=1/tca_mu,parameter_convention='I + mu_internal KLK; journal KLK + mu_paper I',constraint_error=model.constraint_error,stationarity_error=model.stationarity_error)
        arrays=dict(coefficients=model.coefficients,eigenvalues=model.eigenvalues)
    elif name=='GFK':
        model=fit_gfk(X,target,dimension)
        source_new=model.transform_source(X);target_new=model.transform_target(target)
        diag.update(preprocessing=model.preprocessing,dimension_policy='explicit fixed dimension; not automatic SDM')
        arrays=dict(factor=model.factor,source_basis=model.source_basis,target_basis=model.target_basis,source_location=model.source_location,target_location=model.target_location,source_scale=model.source_scale,target_scale=model.target_scale)
    elif name=='DIP':
        model=fit_dip(X,target,dimension)
        source_new=model.transform(X);target_new=model.transform(target)
        diag.update(stop_reason=model.stop_reason,history=model.history,orthogonality_error=model.orthogonality_error,sigma=model.sigma)
        arrays=dict(joint_mean=model.joint_mean,pca_basis=model.pca_basis,projection=model.projection)
    else:raise ValueError('Unverified or unknown feature adaptation: '+name)
    if not np.isfinite(source_new).all() or not np.isfinite(target_new).all():raise ArithmeticError('Nonfinite adaptation')
    adapted,gamma=adapted_problem(problem,source_new,target_new)
    diag['kernel_gamma']=gamma
    return adapted,diag,arrays


def classify_adaptation(head,problem,*,scale=1.,seed=0):
    if head not in HEADS:raise ValueError('Unknown event classifier head')
    return fit(head,problem,parameters(head,scale),seed=seed)


def select_validation_family(records,class_ids,scales):
    """Complete validation macro-AP only; deterministic head/scale tie order.

    Failed/incomplete candidates excluded, duplicate rows rejected. Test rows
    rejected rather than ignored, preventing accidental test-directed choice.
    SVM is a separate reported comparator, never a MIL-PI family candidate.
    """
    classes=set(class_ids)
    if not classes or len(classes)!=len(class_ids):raise ValueError('Unique nonempty validation classes required')
    if any(r['split']!='validate' for r in records):raise ValueError('Selection accepts validation records only')
    options=[]
    for head in HEADS[1:]:
        for scale in scales:
            rows=[r for r in records if r['head']==head and r['scale']==scale]
            ids=[r['class_id'] for r in rows]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate validation class candidate')
            if set(ids)!=classes or any(r['status']!='ok' for r in rows):continue
            values=np.array([r['average_precision'] for r in rows],float)
            if not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid validation AP')
            options.append(dict(head=head,scale=scale,validation_map=float(values.mean())))
    if not options:raise ValueError('No complete valid MIL-PI family candidate')
    return max(options,key=lambda r:r['validation_map'])
