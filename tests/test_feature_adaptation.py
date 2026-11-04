import unittest
import numpy as np
from scipy.linalg import eigh
from experiments.feature_adaptation import fit_sa,fit_tca,row_zscore


class FeatureAdaptationTest(unittest.TestCase):
    def test_sa_solves_orthogonal_basis_alignment(self):
        rng=np.random.default_rng(21);s=rng.normal(size=(20,5));t=rng.normal(size=(18,5))
        m=fit_sa(s,t,3)
        # Normal equations of min_M ||Ps M-Pt||_F^2.
        np.testing.assert_allclose(m.source_basis.T@(m.source_basis@m.alignment-m.target_basis),0,atol=1e-14)
        self.assertEqual(m.transform_source(s).shape,(20,3))
        self.assertEqual(m.transform_target(t).shape,(18,3))
        np.testing.assert_allclose(m.transform_source(s),row_zscore(s)@m.source_basis@m.source_basis.T@m.target_basis,atol=1e-14)

    def test_tca_matches_regular_pencil_when_constraint_full_rank(self):
        # KHK is necessarily singular. Independent regularization limit
        # approaches the exact finite eigenpairs without discarding nullspace.
        rng=np.random.default_rng(8);x=rng.normal(size=(10,5));K=x@x.T+np.eye(10)
        m=fit_tca(K,4,3,mu=.4)
        e=np.r_[np.ones(4)/4,-np.ones(6)/6];v=K@e;A=.4*np.outer(v,v)+np.eye(10)
        H=np.eye(10)-np.ones((10,10))/10;B=K@H@K
        ref=eigh(A,B+1e-9*np.eye(10),eigvals_only=True)[:3]
        np.testing.assert_allclose(m.eigenvalues,ref,atol=1e-7)
        self.assertLess(m.constraint_error,1e-10)
        self.assertLess(m.stationarity_error,1e-10)

    def test_tca_low_rank_objective_and_constraint(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(12,4));K=x@x.T
        m=fit_tca(K,5,3,mu=.8);z=m.transform_kernel(K)
        H=np.eye(12)-np.ones((12,12))/12
        np.testing.assert_allclose(z.T@H@z,np.eye(3),atol=1e-10)
        empirical_mmd=np.sum((z[:5].mean(axis=0)-z[5:].mean(axis=0))**2)
        self.assertAlmostEqual(.8*empirical_mmd+np.sum(m.coefficients**2),m.eigenvalues.sum(),places=10)
        with self.assertRaises(ValueError):fit_tca(K,5,5)


if __name__=='__main__':unittest.main()
