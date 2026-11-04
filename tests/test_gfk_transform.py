import unittest
import numpy as np
from scipy.integrate import trapezoid
from experiments.geodesic_baselines import fit_gfk,principal_geodesic


class GFKTransformTest(unittest.TestCase):
    def test_full_pipeline_matches_numerical_flow_kernel(self):
        rng=np.random.default_rng(3);S=rng.normal(size=(13,6));T=rng.normal(size=(17,6))
        m=fit_gfk(S,T,2)
        def standard(X):
            X=X/np.linalg.norm(X,axis=1,keepdims=True)
            return (X-X.mean(axis=0))/X.std(axis=0)
        A=standard(S);B=standard(T);path=principal_geodesic(m.source_basis,m.target_basis)
        t=np.linspace(0,1,2001);kernels=np.array([(A@path.basis(x))@(B@path.basis(x)).T for x in t])
        reference=trapezoid(kernels,t,axis=0)
        np.testing.assert_allclose(m.transform_source(S)@m.transform_target(T).T,reference,atol=1e-6)

    def test_identical_domain_is_pca_projector_and_out_of_sample_fixed(self):
        rng=np.random.default_rng(5);X=rng.normal(size=(20,5));m=fit_gfk(X,X,2,preprocessing='none')
        np.testing.assert_allclose(m.factor@m.factor.T,m.source_basis@m.source_basis.T,atol=1e-12)
        Y=rng.normal(size=(3,5))
        np.testing.assert_allclose(m.transform_target(Y),Y@m.factor)
        np.testing.assert_allclose(m.transform_source(Y),m.transform_target(Y))

    def test_rank_and_finite_validation(self):
        with self.assertRaises(ValueError):fit_gfk(np.ones((4,3)),np.ones((4,3)),1)
        with self.assertRaises(ValueError):fit_gfk(np.eye(3),np.eye(3),3)


if __name__=='__main__':unittest.main()
