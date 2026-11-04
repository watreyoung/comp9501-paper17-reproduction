"""Optional compiled scans for identical max-gap SMO (no fast-math)."""
import numpy as np
from numba import njit


@njit(cache=True)
def select_pair(y,alpha,cost,f):
    upper=-np.inf;lower=np.inf;i=-1;j=-1
    for k in range(len(y)):
        r=y[k]-f[k]
        up=(y[k]>0 and alpha[k]<cost[k]) or (y[k]<0 and alpha[k]>0)
        low=(y[k]>0 and alpha[k]>0) or (y[k]<0 and alpha[k]<cost[k])
        # Strict comparisons preserve NumPy first-index tie behavior.
        if up and r>upper:upper=r;i=k
        if low and r<lower:lower=r;j=k
    return i,j,upper-lower


@njit(cache=True)
def update_scores(f,ki,kj,step):
    for k in range(len(f)):f[k]+=step*(ki[k]-kj[k])
