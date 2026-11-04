"""Word-level linear mi-SVM representation for target-paper Classeme.

Li/Wu/Tu CVPR2013 section3.3 uses a linear single-concept classifier.
Implement classic at-least-one-positive mi-SVM with variable bag sizes;
no target labels/captions or event labels enter representation fitting.
"""
from dataclasses import dataclass
import numpy as np
from mil_pi.paper_models import dual_problem
from mil_pi.qp import qp_session
from .concept_data import classeme_word_bags


def variable_bag_update(scores,positive_sizes):
    scores=np.asarray(scores,float);sizes=np.asarray(positive_sizes,int)
    if sizes.ndim!=1 or not len(sizes) or np.any(sizes<1) or sizes.sum()>=len(scores):raise ValueError('Positive bag sizes and negative instances required')
    npos=int(sizes.sum());labels=np.r_[np.where(scores[:npos]>0,1.,-1.),-np.ones(len(scores)-npos)]
    start=0
    for size in sizes:
        if not np.any(labels[start:start+size]>0):labels[start+int(np.argmax(scores[start:start+size]))]=1.
        start+=size
    return labels


def train_variable_mi(K,positive_sizes,*,C=1.,max_iter=100):
    K=np.asarray(K,float);sizes=np.asarray(positive_sizes,int);n=len(K);npos=int(sizes.sum())
    if K.shape!=(n,n) or not np.isfinite(K).all() or C<=0 or max_iter<1:raise ValueError('Finite square kernel and positive cost/iterations required')
    # Validate sizes even if the first iterate converges immediately.
    variable_bag_update(np.zeros(n),sizes)
    labels=np.r_[np.ones(npos),-np.ones(n-npos)];seen=set();history=[]
    for iteration in range(max_iter):
        model=dual_problem(K,np.zeros_like(K),labels,np.ones(n),np.full(n,C),np.zeros(n,bool))
        model.labels=labels.copy();new=variable_bag_update(model.predict(K),sizes)
        history.append(dict(iteration=iteration+1,objective=model.objective,label_changes=int(np.count_nonzero(new!=labels))))
        if np.array_equal(new,labels):reason='labels_stable';break
        if new.tobytes() in seen:reason='label_cycle';break
        if iteration+1==max_iter:reason='max_iter';break
        seen.add(labels.tobytes());labels=new
    model.history=history;model.stop_reason=reason
    return model


@dataclass
class Classeme:
    weights: np.ndarray
    biases: np.ndarray
    diagnostics: list

    def decision_function(self,visual):
        X=np.asarray(visual,float)
        if X.ndim!=2 or X.shape[1]!=self.weights.shape[0] or not np.isfinite(X).all():raise ValueError('Finite visual feature matrix required')
        return X@self.weights+self.biases

    def augmented_features(self,visual):
        return np.column_stack([visual,self.decision_function(visual)])


def fit_classeme(visual,text,splits,*,C=1.,n_bags=5,bag_size=5,seed=0,max_iter=100):
    X=np.asarray(visual,float);T=np.asarray(text)
    if X.ndim!=2 or len(X)!=len(splits) or T.ndim!=2 or len(T)!=len(X):raise ValueError('Shared visual/text/split rows required')
    weights=[];biases=[];diagnostics=[]
    for word in range(T.shape[1]):
        pos,neg=classeme_word_bags(T,splits,word,n_bags=n_bags,bag_size=bag_size,seed=seed)
        rows=np.concatenate(pos+neg);source=X[rows]
        if not np.isfinite(source).all():raise ValueError('Finite source visual features required')
        with qp_session('osqp') as trace:
            model=train_variable_mi(source@source.T,[len(b) for b in pos],C=C,max_iter=max_iter)
        weights.append(source.T@model.coef);biases.append(model.bias)
        diagnostics.append(dict(word=word,positive_rows=[b.tolist() for b in pos],negative_rows=[b.tolist() for b in neg],stop_reason=model.stop_reason,history=model.history,max_constraint_violation=max((r['constraint_violation'] for r in trace),default=0.)))
    return Classeme(np.column_stack(weights),np.array(biases),diagnostics)
