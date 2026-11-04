"""Exact indexed product RBF for author MIMLBoost conditional instances.

Avoid storing N*Q copies of D-dimensional visual vectors. This is a kernel
primitive, not yet a scalable full MIMLBoost solver.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .miml_conditionals import conditional_layout


class ConditionalRBF:
    def __init__(self,bags,n_labels,*,gamma=1.):
        arrays,self.lengths,self.ends=conditional_layout(bags,n_labels)
        if not np.isfinite(gamma) or gamma<=0:raise ValueError('Positive finite RBF gamma required')
        self.n_labels=n_labels;self.starts=np.r_[0,self.ends[:-1]]
        self.visual_starts=np.r_[0,np.cumsum(self.lengths)[:-1]]
        visual=np.vstack(arrays);d=visual.shape[1];er=1. if d+1<=50 else np.log(d+1)/np.log(50)
        self.visual_kernel=np.exp(-gamma*cdist(visual,visual,'sqeuclidean'))
        label=er*np.arange(n_labels)/(n_labels-1)
        self.label_kernel=np.exp(-gamma*(label[:,None]-label[None,:])**2)
        self.total_instances=int(self.ends[-1])

    def decode(self,indices):
        indices=np.asarray(indices)
        if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or np.any((indices<0)|(indices>=self.total_instances)):raise ValueError('Valid integer conditional instance indices required')
        bag=np.searchsorted(self.ends,indices,side='right');offset=indices-self.starts[bag]
        return self.visual_starts[bag]+offset%self.lengths[bag],offset//self.lengths[bag],bag*self.n_labels+offset//self.lengths[bag]

    def block(self,left,right):
        a,j,_=self.decode(left);b,k,_=self.decode(right)
        return self.visual_kernel[np.ix_(a,b)]*self.label_kernel[np.ix_(j,k)]


def compressed_bootstrap(sampled_indices,n_instances):
    """Duplicate-instance aggregation: repeated hinge losses become count*C.

    Preserve actual draws exactly. No deduplication of distinct visual rows or
    resampling on labels. Caller maps each active index to its fixed label.
    """
    sampled=np.asarray(sampled_indices)
    if sampled.ndim!=1 or not np.issubdtype(sampled.dtype,np.integer) or n_instances<1 or np.any((sampled<0)|(sampled>=n_instances)):raise ValueError('Valid bootstrap indices required')
    counts=np.bincount(sampled,minlength=n_instances);active=np.flatnonzero(counts)
    return active,counts[active]
