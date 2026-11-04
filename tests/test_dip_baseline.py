import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from experiments.dip_baseline import gaussian_mmd_value_gradient,grassmann_step,fit_dip


class DIPBaselineTest(unittest.TestCase):
    def test_gaussian_gradient_matches_independent_finite_difference(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(11,5));W=rng.normal(size=(5,2))
        value,g=gaussian_mmd_value_gradient(X,W,4,2.)
        ref=approx_derivative(lambda x:gaussian_mmd_value_gradient(X,x.reshape(5,2),4,2.)[0],W.ravel(),method='3-point')
        np.testing.assert_allclose(g.ravel(),ref.ravel(),atol=1e-9,rtol=1e-6)
        self.assertGreaterEqual(value,-1e-12)

    def test_geodesic_and_transport_preserve_manifold_metric(self):
        rng=np.random.default_rng(9);W=np.linalg.qr(rng.normal(size=(9,3)))[0]
        H=rng.normal(size=W.shape);H-=W@(W.T@H)
        Z=rng.normal(size=W.shape);Z-=W@(W.T@Z)
        new,transport=grassmann_step(W,H,.3);T=transport(Z)
        np.testing.assert_allclose(new.T@new,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(new.T@T,0,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(T),np.linalg.norm(Z),places=11)
        tiny,_=grassmann_step(W,H,1e-7)
        np.testing.assert_allclose((tiny-W)/1e-7,H,atol=1e-6)

    def test_full_optimizer_decreases_original_mmd_without_labels(self):
        rng=np.random.default_rng(19);s=rng.normal(size=(24,6));t=rng.normal(size=(20,6))+np.array([2,0,0,0,0,0])
        model=fit_dip(s,t,2,max_iter=50)
        self.assertTrue(model.history)
        self.assertLess(model.history[-1]['objective'],model.history[0]['previous_objective'])
        self.assertTrue(all(h['objective']<=h['previous_objective']+1e-12 for h in model.history))
        self.assertLess(model.orthogonality_error,1e-10)
        self.assertEqual(model.transform(t).shape,(20,2))
        self.assertNotEqual(model.stop_reason,'line_search_failed')


if __name__=='__main__':unittest.main()
