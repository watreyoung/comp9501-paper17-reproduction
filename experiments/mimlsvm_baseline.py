"""Zhou & Zhang NIPS 2006 MIMLSVM, Table 2 (one MIML variant).

Bags use Euclidean Hausdorff distance, training-only k-medoids embedding,
and one RBF SVM per semantic word. Expose scores for the target IJCV
paper's downstream feature construction; no target captions are needed.
The target paper does not identify which MIML variant it used: this module
must be reported explicitly as MIMLSVM, not as an unspecified exact match.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from .hausdorff_blocks import blocked_hausdorff


def hausdorff(A,B):
    A=np.asarray(A,float);B=np.asarray(B,float)
    if A.ndim!=2 or B.ndim!=2 or A.shape[1]!=B.shape[1] or not len(A) or not len(B):raise ValueError('Nonempty bags with shared feature dimension required')
    if not np.isfinite(A).all() or not np.isfinite(B).all():raise ValueError('Finite bag features required')
    D=cdist(A,B,'euclidean')
    return float(max(D.min(axis=1).max(),D.min(axis=0).max()))


def bag_distances(bags,reference):
    return np.array([[hausdorff(A,B) for B in reference] for A in bags])


def kmedoids_paper(D,k,seed=0,max_iter=100):
    """Author 2008 MIML_cluster: random partition, first minimum on ties.

    RNG is deterministic instead of MATLAB clock-seeded randomness."""
    D=np.asarray(D,float);n=len(D)
    if D.shape!=(n,n) or not np.isfinite(D).all() or not np.allclose(D,D.T) or np.any(D<0):raise ValueError('Symmetric nonnegative bag distances required')
    if not 1<=k<=n or max_iter<1:raise ValueError('Invalid medoid count or iteration limit')
    rng=np.random.default_rng(seed)
    initial=rng.choice(n,k,replace=False)
    partition=rng.integers(k,size=n);partition[initial]=np.arange(k)
    medoids=np.array([members[np.argmin(D[np.ix_(members,members)].sum(axis=1))]
                      for members in [np.flatnonzero(partition==j) for j in range(k)]])
    history=[];reason='max_iter'
    for iteration in range(max_iter):
        assignments=np.argmin(D[:,medoids],axis=1)
        assignments[medoids]=np.arange(k)  # Step 2a: medoid remains in its own cluster.
        new=medoids.copy()
        for cluster in range(k):
            members=np.flatnonzero(assignments==cluster);cost=D[np.ix_(members,members)].sum(axis=1)
            new[cluster]=members[int(np.argmin(cost))]
        history.append(dict(iteration=iteration+1,assignment_objective=float(D[np.arange(n),medoids[assignments]].sum()),medoid_changes=int(np.count_nonzero(new!=medoids))))
        if set(new)==set(medoids):reason='medoids_stable';break
        medoids=new
    return medoids,history,reason


@dataclass
class MIMLSVM:
    medoid_bags: list
    medoid_indices: np.ndarray
    classifiers: list
    constants: dict
    history: list
    stop_reason: str

    def decision_function(self,bags):
        Z=blocked_hausdorff(bags,self.medoid_bags);columns=[]
        for j,model in enumerate(self.classifiers):
            columns.append(np.full(len(Z),self.constants[j]) if model is None else model.decision_function(Z))
        return np.column_stack(columns)

    def predict_labels(self,bags):
        scores=self.decision_function(bags);labels=scores>0
        labels[np.arange(len(scores)),np.argmax(scores,axis=1)]=True
        return labels


def fit_mimlsvm(bags,word_labels,*,medoid_fraction=.2,C=1.,gamma=.2,seed=0,max_iter=100,distance_backend="blocked"):
    labels=np.asarray(word_labels,dtype=float)
    if labels.ndim!=2 or len(labels)!=len(bags) or not labels.shape[1] or not np.isin(labels,[0,1]).all():raise ValueError('Binary bag-by-word labels required')
    if not 0<medoid_fraction<=1 or C<=0:raise ValueError('Valid medoid fraction and positive cost required')
    if distance_backend not in ['blocked','reference']:raise ValueError('Unknown distance backend')
    D=(blocked_hausdorff if distance_backend=='blocked' else bag_distances)(bags,bags);k=max(1,int(np.floor(medoid_fraction*len(bags))))
    medoids,history,reason=kmedoids_paper(D,k,seed,max_iter)
    Z=D[:,medoids];classifiers=[];constants={}
    for j in range(labels.shape[1]):
        y=2*labels[:,j]-1
        if len(np.unique(y))==1:
            constants[j]=float(y[0]);classifiers.append(None)
        else:
            model=SVC(C=C,kernel='rbf',gamma=gamma,tol=1e-7,max_iter=100000)
            model.fit(Z,y)
            if model.fit_status_!=0:raise ArithmeticError('MIMLSVM word classifier did not converge')
            classifiers.append(model)
    return MIMLSVM([np.asarray(bags[i],float).copy() for i in medoids],medoids,classifiers,constants,history,reason)
