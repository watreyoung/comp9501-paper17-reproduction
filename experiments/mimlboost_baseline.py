"""MIMLBoost, Zhou/Zhang NIPS2006, checked against author June2008 code.

Joint scalar label augmentation and weighted bootstrap (not independent
per-label AdaBoost). Preserve author's all-errors<=.5 pre-storage stop.
Use deterministic RNG and stable convex 1D minimization instead of clock
seeding/fminunc. Inputs must already use a training-fitted normalization.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from sklearn.svm import SVC
from .miml_conditionals import conditional_layout,conditional_blocks


def conditional_instances(bags,n_labels):
    if n_labels<2 or not len(bags):
        raise ValueError('At least two labels and one bag required')
    arrays=[np.asarray(b,float) for b in bags]
    d=arrays[0].shape[1] if arrays[0].ndim==2 else -1
    if d<1 or any(b.ndim!=2 or not len(b) or b.shape[1]!=d or not np.isfinite(b).all() for b in arrays):
        raise ValueError('Finite nonempty bags of shared dimension required')
    er=1. if d+1<=50 else np.log(d+1)/np.log(50)
    counts=np.repeat([len(b) for b in arrays],n_labels)
    X=np.vstack([np.column_stack([b,np.full(len(b),er*j/(n_labels-1))])
                 for b in arrays for j in range(n_labels)])
    return X,counts


def exponential_step(weights,errors):
    """Minimize sum w exp((2e-1)c); return nonpositive-stop as c=0."""
    weights=np.asarray(weights,float);errors=np.asarray(errors,float)
    if weights.shape!=errors.shape or np.any(weights<=0) or not np.isfinite(weights).all() or np.any((errors<0)|(errors>1)) or not np.isfinite(errors).all():
        raise ValueError('Positive finite weights and errors in [0,1] required')
    a=2*errors-1;logw=np.log(weights)
    def derivative(c):
        z=logw+a*c;return float(np.dot(np.exp(z-logsumexp(z)),a))
    if derivative(0)>=0:return 0.
    if not np.any(a>0):return float('inf')
    upper=1.
    while derivative(upper)<0:
        upper*=2
        if not np.isfinite(upper):raise ArithmeticError('Cannot bracket boosting coefficient')
    return float(brentq(derivative,0,upper,xtol=1e-12))


@dataclass
class ConstantLearner:
    label: float
    def predict(self,X):return np.full(len(X),self.label)


@dataclass
class MIMLBoost:
    classifiers: list
    coefficients: np.ndarray
    n_labels: int
    feature_dimension: int
    history: list
    stop_reason: str
    final_weights: np.ndarray

    def decision_function(self,bags,*,batch_size=4096):
        arrays,_,_=conditional_layout(bags,self.n_labels)
        if arrays[0].shape[1]!=self.feature_dimension:raise ValueError('Feature dimension differs from training')
        result=np.zeros(len(arrays)*self.n_labels)
        if not self.classifiers:return result.reshape(len(arrays),self.n_labels)
        for X,group in conditional_blocks(arrays,self.n_labels,batch_size=batch_size):
            scores=np.zeros(len(X))
            for model,c in zip(self.classifiers,self.coefficients):scores+=c*model.predict(X)
            np.add.at(result,group,scores)
        return result.reshape(len(arrays),self.n_labels)

    def predict_labels(self,bags):return self.decision_function(bags)>=0


def fit_mimlboost(bags,word_labels,*,rounds=100,C=1.,gamma=1.,seed=0):
    labels=np.asarray(word_labels,float)
    if labels.ndim!=2 or len(labels)!=len(bags) or not np.isin(labels,[0,1]).all():raise ValueError('Binary bag-by-label matrix required')
    if rounds<1 or C<=0 or gamma<=0:raise ValueError('Positive rounds, cost and gamma required')
    X,counts=conditional_instances(bags,labels.shape[1]);group=np.repeat(np.arange(len(counts)),counts)
    y=np.repeat((2*labels-1).ravel(),counts)
    weights=np.full(len(counts),1/len(counts));rng=np.random.default_rng(seed)
    models=[];coefficients=[];history=[];reason='max_rounds'
    for iteration in range(rounds):
        probability=weights[group]/counts[group]
        sampled=rng.choice(len(X),len(X),replace=True,p=probability/probability.sum())
        unique=np.unique(y[sampled]);constant=len(unique)==1
        if constant:model=ConstantLearner(float(unique[0]))
        else:
            model=SVC(C=C,gamma=gamma,kernel='rbf',tol=1e-3,max_iter=100000)
            model.fit(X[sampled],y[sampled])
            if model.fit_status_!=0:raise ArithmeticError('MIMLBoost base SVM did not converge')
        errors=np.bincount(group,weights=(model.predict(X)!=y).astype(float),minlength=len(counts))/counts
        entry=dict(round=iteration+1,errors=errors.tolist(),weights_before=weights.tolist(),constant_bootstrap=constant,stored=False)
        history.append(entry)
        if np.all(errors<=.5):reason='all_errors_at_most_half_before_store';break
        coefficient=exponential_step(weights,errors);entry['coefficient']=coefficient
        if coefficient<=0:reason='nonpositive_coefficient_before_store';break
        if not np.isfinite(coefficient):raise ArithmeticError('Unexpected infinite coefficient after author stop check')
        logweights=np.log(weights)+(2*errors-1)*coefficient
        weights=np.exp(logweights-logsumexp(logweights))
        models.append(model);coefficients.append(coefficient);entry['stored']=True
    return MIMLBoost(models,np.asarray(coefficients),labels.shape[1],X.shape[1]-1,history,reason,weights)
