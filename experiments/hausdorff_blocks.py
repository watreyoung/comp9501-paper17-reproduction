"""Exact Euclidean Hausdorff matrix with bounded temporary storage.

Compute squared Euclidean distances in compiled SciPy, then monotone
min/max reductions and one final square root per bag pair. Ragged bags are
masked, not duplicated. No approximation or feature projection.
"""
import numpy as np
from scipy.spatial.distance import cdist


def blocked_hausdorff(bags,reference,*,block_bags=16):
    if not isinstance(block_bags,int) or block_bags<1:raise ValueError('Positive integer block size required')
    arrays=[np.asarray(b,float) for b in bags];refs=[np.asarray(b,float) for b in reference]
    if not arrays or not refs:raise ValueError('Nonempty bag collections required')
    d=arrays[0].shape[1] if arrays[0].ndim==2 else 0
    if d<1 or any(b.ndim!=2 or b.shape[1]!=d or not len(b) or not np.isfinite(b).all() for b in arrays+refs):raise ValueError('Finite nonempty shared-dimensional bags required')
    lr=max(map(len,refs));R=np.zeros((len(refs),lr,d));rmask=np.zeros((len(refs),lr),bool)
    for i,b in enumerate(refs):R[i,:len(b)]=b;rmask[i,:len(b)]=True
    output=np.empty((len(arrays),len(refs)))
    for start in range(0,len(arrays),block_bags):
        block=arrays[start:start+block_bags];lb=max(map(len,block))
        A=np.zeros((len(block),lb,d));amask=np.zeros((len(block),lb),bool)
        for i,b in enumerate(block):A[i,:len(b)]=b;amask[i,:len(b)]=True
        D=cdist(A.reshape(-1,d),R.reshape(-1,d),'sqeuclidean').reshape(len(block),lb,len(refs),lr)
        # Directed A->R: ignore padded R candidates, then padded A maxima.
        forward=np.where(rmask[None,None,:,:],D,np.inf).min(axis=3)
        forward=np.where(amask[:,:,None],forward,0).max(axis=1)
        backward=np.where(amask[:,:,None,None],D,np.inf).min(axis=1)
        backward=np.where(rmask[None,:,:],backward,0).max(axis=2)
        output[start:start+len(block)]=np.sqrt(np.maximum(forward,backward))
    return output
