"""Principal-angle geodesics for GFK and sampled-flow baseline review.

Independent implementation from the subspace/projector definitions, not a
copy of the archived external reference. Public protocol not yet frozen.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PrincipalGeodesic:
    source_axes: np.ndarray
    normal_axes: np.ndarray
    angles: np.ndarray

    def basis(self,t):
        if not 0<=t<=1:raise ValueError('Geodesic parameter must lie in [0,1]')
        return self.source_axes*np.cos(t*self.angles)+self.normal_axes*np.sin(t*self.angles)

    def integrated_factor(self):
        """F such that FF.T = integral_0^1 Phi(t)Phi(t).T dt (GFK)."""
        theta=self.angles;d=len(theta)
        sinc=np.sinc(2*theta/np.pi)
        aa=(1+sinc)/2;cc=(1-sinc)/2
        # sin(theta)^2/(2 theta) is stable also at theta=0.
        bb=np.zeros(d);np.divide(np.sin(theta)**2,2*theta,out=bb,where=theta>0)
        block=np.block([[np.diag(aa),np.diag(bb)],[np.diag(bb),np.diag(cc)]])
        values,U=np.linalg.eigh(block)
        if values.min()<-1e-12:raise ArithmeticError('Integrated flow is not PSD')
        return np.c_[self.source_axes,self.normal_axes]@(U*np.sqrt(np.maximum(values,0)))

    def sampled_factor(self,samples):
        """Uniform concatenation of sampled intermediate subspace features.

        This is the sampled projector primitive. Full SGF source-specific
        preprocessing and classifier protocol still require original review.
        """
        if not isinstance(samples,int) or samples<2:raise ValueError('At least two samples required')
        return np.concatenate([self.basis(t) for t in np.linspace(0,1,samples)],axis=1)/np.sqrt(samples)


def principal_geodesic(source_basis,target_basis):
    Ps=np.asarray(source_basis,float);Pt=np.asarray(target_basis,float)
    if Ps.ndim!=2 or Ps.shape!=Pt.shape:raise ValueError('Bases must have identical shapes')
    d=Ps.shape[1]
    if not d or not np.isfinite(Ps).all() or not np.isfinite(Pt).all():raise ValueError('Finite nonempty bases required')
    for P in [Ps,Pt]:
        if not np.allclose(P.T@P,np.eye(d),atol=1e-9):raise ValueError('Orthonormal bases required')
    U,c,Vt=np.linalg.svd(Ps.T@Pt);c=np.clip(c,0,1);theta=np.arccos(c)
    A=Ps@U;Q=Pt@Vt.T;residual=Q-A*c
    sine=np.sin(theta);R=np.zeros_like(A);active=sine>1e-7
    R[:,active]=residual[:,active]/sine[active]
    # Shared axes contribute a constant projector and no normal direction.
    theta[~active]=0.
    return PrincipalGeodesic(A,R,theta)


@dataclass
class GeodesicFlow:
    factor: np.ndarray
    source_location: np.ndarray
    source_scale: np.ndarray
    target_location: np.ndarray
    target_scale: np.ndarray
    source_basis: np.ndarray
    target_basis: np.ndarray
    angles: np.ndarray
    preprocessing: str

    def transform_source(self,X):
        return self._transform(X,self.source_location,self.source_scale)

    def transform_target(self,X):
        return self._transform(X,self.target_location,self.target_scale)

    def _transform(self,X,location,scale):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=len(location) or not np.isfinite(X).all():raise ValueError('Finite shared-dimensional samples required')
        if self.preprocessing=='domain_zscore':
            X=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
            X=(X-location)/scale
        return X@self.factor


def fit_gfk(source,target,dimension,*,preprocessing='domain_zscore'):
    """PCA/PCA GFK Eq1-6, with explicit fixed subspace dimension.

    domain_zscore: unit-normalize samples, fit column zscores separately on
    source and unlabeled target (CVPR2012 experimental feature convention).
    none: caller-preprocessed features. PCA uses centered covariance; the
    resulting kernel projects the actual standardized samples, Eq4.
    Fixed dimension is a public-protocol choice, not Eq8 automatic SDM.
    No target labels are accepted. Separate target scaling is transductive.
    """
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1] or min(len(source),len(target))<2 or not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Finite source and target matrices with shared dimension required')
    if preprocessing not in ['none','domain_zscore']:raise ValueError('Unknown preprocessing')
    locations=[];scales=[];bases=[]
    for X in [source,target]:
        if preprocessing=='domain_zscore':
            X=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
            location=X.mean(axis=0);scale=X.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
            X=(X-location)/scale
        else:location=np.zeros(X.shape[1]);scale=np.ones(X.shape[1])
        _,s,Vt=np.linalg.svd(X-X.mean(axis=0),full_matrices=False)
        rank=int(np.count_nonzero(s>max(X.shape)*np.finfo(float).eps*s[0])) if len(s) and s[0]>0 else 0
        if not isinstance(dimension,int) or not 1<=dimension<=rank:raise ValueError('Dimension exceeds centered domain rank')
        bases.append(Vt[:dimension].T);locations.append(location);scales.append(scale)
    path=principal_geodesic(*bases)
    return GeodesicFlow(path.integrated_factor(),locations[0],scales[0],locations[1],scales[1],bases[0],bases[1],path.angles,preprocessing)


def subspace_disagreement(source_basis,target_basis,joint_basis):
    """GFK Eq7: mean sine of largest principal angles to joint PCA space.

    Every column prefix defines one candidate dimension. Returns D(1)..D(k)
    without enforcing monotonicity, which does not hold in general.
    """
    bases=[np.asarray(P,float) for P in [source_basis,target_basis,joint_basis]]
    if len({P.shape for P in bases})!=1 or bases[0].ndim!=2 or not bases[0].shape[1]:raise ValueError('Three matching nonempty basis matrices required')
    for P in bases:
        if not np.isfinite(P).all() or not np.allclose(P.T@P,np.eye(P.shape[1]),atol=1e-9):raise ValueError('Finite orthonormal PCA bases required')
    values=[]
    for d in range(1,bases[0].shape[1]+1):
        cosine=[np.clip(np.linalg.svd(P[:,:d].T@bases[2][:,:d],compute_uv=False)[-1],0.,1.) for P in bases[:2]]
        values.append(.5*sum(np.sqrt(max(0.,1-c*c)) for c in cosine))
    return np.array(values)


def select_sdm_dimension(disagreement,*,orthogonality_tolerance=1e-8):
    """Eq8 first D(d)=1, within explicit numerical tolerance.

    Return None when no candidate reaches 1: the paper does not specify a
    fallback. Callers must record/choose a policy, never silently use argmax
    or the last dimension as if it were Eq8's answer.
    """
    values=np.asarray(disagreement,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any((values<0)|(values>1)) or not 0<=orthogonality_tolerance<1:raise ValueError('Finite SDM values in [0,1] and valid tolerance required')
    candidates=np.flatnonzero(values>=1-orthogonality_tolerance)
    return int(candidates[0]+1) if len(candidates) else None
