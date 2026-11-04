"""Bruzzone/Marconcini PAMI2010 DASVM three-phase solver, Eq1-12.

Audited margin selection, pseudo-label ages and target-only convergence.
"""
import numpy as np


def select_margin_targets(scores,unlabeled,quota):
    """Eq2/3: inside margin, closest to +1/-1 first, separately by sign."""
    scores=np.asarray(scores,float);unlabeled=np.asarray(unlabeled,bool)
    if scores.ndim!=1 or scores.shape!=unlabeled.shape or not np.isfinite(scores).all() or not isinstance(quota,int) or quota<1:raise ValueError('Finite target scores, matching mask and positive quota required')
    upper=np.flatnonzero(unlabeled&(scores>=0)&(scores<=1))
    lower=np.flatnonzero(unlabeled&(scores<0)&(scores>=-1))
    return upper[np.argsort(-scores[upper],kind='stable')[:quota]],lower[np.argsort(scores[lower],kind='stable')[:quota]]


def select_source_deletions(scores,remaining,upper_added,lower_added,quota):
    """Eq6/7: farthest by predicted side, not source truth; empty H fallback."""
    scores=np.asarray(scores,float);remaining=np.asarray(remaining,bool)
    if scores.ndim!=1 or scores.shape!=remaining.shape or not np.isfinite(scores).all() or quota<1 or min(upper_added,lower_added)<0:raise ValueError('Valid source scores, remaining mask and quotas required')
    upper=np.flatnonzero(remaining&(scores>=0));lower=np.flatnonzero(remaining&(scores<0))
    a,b=(upper_added,lower_added) if upper_added+lower_added else (quota,quota)
    return np.r_[upper[np.argsort(-scores[upper],kind='stable')[:a]],lower[np.argsort(scores[lower],kind='stable')[:b]]]


def update_target_ages(previous_labels,ages,current_scores,newly_selected,tau):
    """Eq4/5: inconsistent pseudo labels return to unlabeled, no same-step readd.

    ages=0 means unlabeled; surviving ages grow to tau; newly selected age1.
    Return new label/age arrays and inconsistent indices for audit.
    """
    labels=np.asarray(previous_labels,float);ages=np.asarray(ages,int);scores=np.asarray(current_scores,float);new=np.asarray(newly_selected,int)
    if labels.ndim!=1 or labels.shape!=ages.shape or labels.shape!=scores.shape or tau<2 or np.any((ages<0)|(ages>tau)) or not np.isfinite(scores).all():raise ValueError('Valid pseudo-label state, finite scores and tau>=2 required')
    active=ages>0
    if not np.isin(labels[active],[-1,1]).all() or np.any(labels[~active]!=0):raise ValueError('Labels must match active pseudo-label mask')
    if len(np.unique(new))!=len(new) or np.any((new<0)|(new>=len(ages))) or np.any(active[new]):raise ValueError('New targets must be distinct previously unlabeled indices')
    predicted=np.where(scores>=0,1.,-1.)
    inconsistent=np.flatnonzero(active&(predicted!=labels));survive=active&(predicted==labels)
    result_ages=np.zeros_like(ages);result_labels=np.zeros_like(labels)
    result_ages[survive]=np.minimum(ages[survive]+1,tau);result_labels[survive]=labels[survive]
    result_ages[new]=1;result_labels[new]=predicted[new]
    return result_labels,result_ages,inconsistent


def source_cost(iteration,C,initial_target_cost,tau):
    """Eq10: quadratic source decay, floored at initial target cost."""
    if iteration<0 or C<=0 or not 0<initial_target_cost<=C or tau<2:raise ValueError('Valid iteration and positive cost schedule required')
    return max((initial_target_cost-C)*iteration**2/tau**2+C,initial_target_cost)


def target_costs(ages,C,initial_target_cost,tau,maximum_fraction=.5):
    """Eq9: quadratic growth by uninterrupted pseudo-label age."""
    ages=np.asarray(ages,int)
    if C<=0 or not 0<maximum_fraction<=1 or not 0<initial_target_cost<=maximum_fraction*C or tau<2 or np.any((ages<1)|(ages>tau)):raise ValueError('Valid active ages and target cost schedule required')
    return initial_target_cost+(maximum_fraction*C-initial_target_cost)*(ages-1)**2/(tau-1)**2


from dataclasses import dataclass
from sklearn.svm import SVC


@dataclass
class ConstantDASVMClassifier:
    label: float
    def decision_function(self,K):return np.full(len(K),self.label)


@dataclass
class DASVM:
    classifier: object
    training_indices: np.ndarray
    source_count: int
    pseudo_labels: np.ndarray
    target_ages: np.ndarray
    history: list
    stop_reason: str

    def decision_function(self,joint_cross_kernel):
        K=np.asarray(joint_cross_kernel,float)
        if K.ndim!=2 or K.shape[1]!=self.source_count+len(self.target_ages) or not np.isfinite(K).all():raise ValueError('Cross kernel must reference all fitted source and target samples')
        return self.classifier.decision_function(K[:,self.training_indices])


def fit_dasvm(joint_kernel,source_labels,*,C=1.,initial_target_cost=.01,tau=10,quota=1,epsilon=.001,maximum_fraction=.5,max_iter=1000):
    """Three-phase binary DASVM, Eq1-12, with audited transitions.

    joint_kernel ordering is source then unlabeled target; target labels are
    never inputs. Original uniform source cost C is used (not the core event
    positive-C1/negative-1 convention); formal comparisons must disclose it.
    Single-class training has exact zero-loss constant solution w=0,b=+/-1;
    report this degeneracy. Empty states raise; never revert to source model. Iteration cap is reported, never called convergence.
    """
    K=np.asarray(joint_kernel,float);y=np.asarray(source_labels,float);ns=len(y)
    if y.ndim!=1 or set(np.unique(y))!={-1.,1.} or K.ndim!=2 or K.shape[0]!=K.shape[1] or len(K)<=ns or not np.isfinite(K).all() or not np.allclose(K,K.T,atol=1e-9):raise ValueError('Finite symmetric joint kernel, two-class source and nonempty target required')
    if not 0<=epsilon<1 or max_iter<1 or not isinstance(quota,int) or quota<1:raise ValueError('Valid stopping tolerance, quota and iteration cap required')
    source_cost(0,C,initial_target_cost,tau);target_costs([1],C,initial_target_cost,tau,maximum_fraction)
    nt=len(K)-ns;remaining=np.ones(ns,bool);labels=np.zeros(nt);ages=np.zeros(nt,int);history=[]
    def train(indices,train_labels,cost):
        if not len(train_labels):raise ArithmeticError('DASVM encountered empty training state')
        if len(np.unique(train_labels))==1:return ConstantDASVMClassifier(float(train_labels[0]))
        classifier=SVC(C=1.,kernel='precomputed',tol=1e-7,max_iter=200000)
        classifier.fit(K[np.ix_(indices,indices)],train_labels,sample_weight=cost)
        if classifier.fit_status_!=0:raise ArithmeticError('DASVM inner SVM did not converge')
        return classifier
    indices=np.arange(ns);classifier=train(indices,y,np.full(ns,C));reason='max_iter'
    for iteration in range(1,max_iter+1):
        scores=classifier.decision_function(K[:,indices]);target_scores=scores[ns:]
        upper,lower=select_margin_targets(target_scores,ages==0,quota);new=np.r_[upper,lower]
        deleted=select_source_deletions(scores[:ns],remaining,len(upper),len(lower),quota)
        new_labels,new_ages,inconsistent=update_target_ages(labels,ages,target_scores,new,tau)
        remaining[deleted]=False;labels=new_labels;ages=new_ages
        active=np.flatnonzero(ages>0);src=np.flatnonzero(remaining)
        indices=np.r_[src,ns+active];cost=np.r_[np.full(len(src),source_cost(iteration,C,initial_target_cost,tau)),target_costs(ages[active],C,initial_target_cost,tau,maximum_fraction)]
        train_labels=np.r_[y[src],labels[active]]
        converged=not len(src) and len(new)<=int(np.ceil(epsilon*nt)) and len(inconsistent)<=int(np.ceil(epsilon*nt))
        history.append(dict(iteration=iteration,source_remaining=src.tolist(),targets_added=new.tolist(),source_deleted=deleted.tolist(),targets_inconsistent=inconsistent.tolist(),active_target_indices=active.tolist(),active_target_labels=labels[active].tolist(),active_target_ages=ages[active].tolist(),training_indices=indices.tolist(),training_costs=cost.tolist(),convergence_criterion=converged))
        classifier=train(indices,train_labels,cost)
        history[-1]['single_class_constant_solution']=isinstance(classifier,ConstantDASVMClassifier)
        if converged:reason='target_only_empirical_convergence';break
    return DASVM(classifier,indices,ns,labels,ages,history,reason)
