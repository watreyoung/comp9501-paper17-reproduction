"""Exact bounded-memory enumeration of author MIMLBoost conditional inputs.

Order: bag, label, instance. Streaming changes only storage, not the scalar
label coordinate, ensemble outputs or number of conditional instances.
"""
import numpy as np


def conditional_layout(bags,n_labels):
    arrays=[np.asarray(b,float) for b in bags]
    if not arrays or n_labels<2:raise ValueError('Nonempty bags and at least two labels required')
    d=arrays[0].shape[1] if arrays[0].ndim==2 else 0
    if d<1 or any(b.ndim!=2 or b.shape[1]!=d or not len(b) or not np.isfinite(b).all() for b in arrays):raise ValueError('Finite nonempty same-dimensional bags required')
    lengths=np.array([len(b) for b in arrays],dtype=np.int64)
    ends=np.cumsum(lengths*n_labels)
    return arrays,lengths,ends


def conditional_blocks(bags,n_labels,*,batch_size=4096):
    if not isinstance(batch_size,int) or batch_size<1:raise ValueError('Positive integer batch size required')
    arrays,lengths,ends=conditional_layout(bags,n_labels);starts=np.r_[0,ends[:-1]]
    d=arrays[0].shape[1];er=1. if d+1<=50 else np.log(d+1)/np.log(50)
    for start in range(0,int(ends[-1]),batch_size):
        index=np.arange(start,min(start+batch_size,int(ends[-1])))
        bag=np.searchsorted(ends,index,side='right');offset=index-starts[bag]
        label=offset//lengths[bag];instance=offset%lengths[bag]
        X=np.empty((len(index),d+1))
        for b in np.unique(bag):
            mask=bag==b;X[mask,:d]=arrays[b][instance[mask]]
        X[:,-1]=er*label/(n_labels-1)
        yield X,bag*n_labels+label


def conditional_resource_estimate(n_instances,n_bags,n_labels,dimension,*,batch_size=4096):
    if min(n_instances,n_bags,n_labels,dimension,batch_size)<1:raise ValueError('Positive dimensions required')
    n=int(n_instances)*int(n_labels)
    # Dense legacy fit constructs full X and bootstrap X[sampled], both float64.
    return dict(conditional_instances=n,dense_features_bytes=n*(dimension+1)*8,
                dense_features_plus_bootstrap_bytes=2*n*(dimension+1)*8,
                streamed_prediction_feature_bytes=min(n,batch_size)*(dimension+1)*8,
                bag_label_weights_bytes=n_bags*n_labels*8,
                note='Lower bounds, exclude labels/indices, SVM support vectors, caches, solver workspace and Python histories')
