"""Full author-rule MIMLBoost using exact product-kernel SVM training.

Avoids conditional feature copies and full conditional Gram matrices. Bootstrap
sample count and every source word remain unchanged. Runtime at full scale is
not implied by small-data equivalence tests.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from .miml_conditional_kernel import ConditionalRBF
from .miml_bootstrap_svm import fit_conditional_bootstrap
from .mimlboost_baseline import exponential_step
from .miml_conditionals import conditional_layout


@dataclass
class IndexedWeakLearner:
    coefficients: np.ndarray
    bias: float

    def signs(self,visual_kernel,label_kernel):
        return np.where((visual_kernel@self.coefficients)@label_kernel+self.bias>0,1.,-1.)


@dataclass
class IndexedMIMLBoost:
    source_visual: np.ndarray
    label_kernel: np.ndarray
    gamma: float
    n_labels: int
    classifiers: list
    coefficients: np.ndarray
    history: list
    stop_reason: str
    final_weights: np.ndarray

    def decision_function(self,bags,*,batch_visual=64):
        arrays,lengths,_=conditional_layout(bags,self.n_labels)
        if batch_visual<1 or arrays[0].shape[1]!=self.source_visual.shape[1]:raise ValueError('Valid batch size and training feature dimension required')
        output=np.zeros((len(arrays),self.n_labels))
        if not self.classifiers:return output
        for b,bag in enumerate(arrays):
            for start in range(0,len(bag),batch_visual):
                K=np.exp(-self.gamma*cdist(bag[start:start+batch_visual],self.source_visual,'sqeuclidean'))
                for weak,c in zip(self.classifiers,self.coefficients):output[b]+=c*weak.signs(K,self.label_kernel).sum(axis=0)
        return output

    def predict_labels(self,bags):return self.decision_function(bags)>=0


def fit_indexed_mimlboost(bags,word_labels,*,rounds=100,C=1.,gamma=1.,seed=0,svm_tolerance=1e-6,svm_max_iter=100000):
    labels=np.asarray(word_labels,float)
    arrays,lengths,_=conditional_layout(bags,labels.shape[1] if labels.ndim==2 else 0)
    if labels.shape!=(len(arrays),labels.shape[1]) or not np.isin(labels,[0,1]).all() or rounds<1 or C<=0:raise ValueError('Matching binary bag-word labels and positive limits required')
    kernel=ConditionalRBF(arrays,labels.shape[1],gamma=gamma);visual=np.vstack(arrays)
    nwords=labels.shape[1];counts=np.repeat(lengths,nwords)
    weights=np.full(len(counts),1/len(counts));rng=np.random.default_rng(seed)
    models=[];coefficients=[];history=[];reason='max_rounds'
    for iteration in range(rounds):
        probability=np.repeat(weights/counts,counts)
        draws=rng.choice(kernel.total_instances,kernel.total_instances,replace=True,p=probability/probability.sum())
        _,_,group=kernel.decode(np.unique(draws));sample_labels=labels.ravel()[group]
        constant=len(np.unique(sample_labels))==1
        if constant:
            # A constant sign function has zero RKHS norm and free +/-1 bias.
            weak=IndexedWeakLearner(np.zeros((len(visual),nwords)),float(2*sample_labels[0]-1))
            solver=dict(stop_reason='single_class_constant',iterations=0,kkt_gap=0.,equality_residual=0.)
        else:
            model,ids=fit_conditional_bootstrap(kernel,labels,draws,C=C,tolerance=svm_tolerance,max_iter=svm_max_iter)
            solver=dict(stop_reason=model.stop_reason,iterations=model.iterations,kkt_gap=model.kkt_gap,equality_residual=model.equality_residual)
            if model.stop_reason!='kkt_tolerance' or model.equality_residual>1e-6:raise ArithmeticError('MIMLBoost weak SVM not converged: '+str(solver))
            vi,wi,_=kernel.decode(ids);coef=np.zeros((len(visual),nwords))
            np.add.at(coef,(vi,wi),model.coefficients);weak=IndexedWeakLearner(coef,model.bias)
        signs=weak.signs(kernel.visual_kernel,kernel.label_kernel)
        # Source bags are consecutive in visual, labels are bag-by-word.
        starts=np.r_[0,np.cumsum(lengths)[:-1]]
        positives=np.add.reduceat((signs>0).astype(float),starts,axis=0)/lengths[:,None]
        errors=np.where(labels>0,1-positives,positives).ravel()
        entry=dict(round=iteration+1,errors=errors.copy(),weights_before=weights.copy(),constant_bootstrap=constant,stored=False,solver=solver)
        history.append(entry)
        if np.all(errors<=.5):reason='all_errors_at_most_half_before_store';break
        c=exponential_step(weights,errors);entry['coefficient']=c
        if c<=0:reason='nonpositive_coefficient_before_store';break
        if not np.isfinite(c):raise ArithmeticError('Unexpected infinite coefficient after author stop')
        logweights=np.log(weights)+(2*errors-1)*c;weights=np.exp(logweights-logsumexp(logweights))
        models.append(weak);coefficients.append(c);entry['stored']=True
    return IndexedMIMLBoost(visual,kernel.label_kernel,gamma,nwords,models,np.asarray(coefficients),history,reason,weights)
