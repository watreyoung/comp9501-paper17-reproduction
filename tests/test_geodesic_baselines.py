import unittest
import numpy as np
from scipy.integrate import quad_vec
from experiments.geodesic_baselines import principal_geodesic


class GeodesicBaselineTest(unittest.TestCase):
    def test_orthogonal_lines_have_known_integral(self):
        g=principal_geodesic(np.array([[1.],[0.]]),np.array([[0.],[1.]]))
        F=g.integrated_factor()
        np.testing.assert_allclose(F@F.T,[[.5,1/np.pi],[1/np.pi,.5]],atol=1e-14)

    def test_closed_form_matches_independent_quadrature(self):
        rng=np.random.default_rng(83);Ps=np.linalg.qr(rng.normal(size=(12,3)))[0];Pt=np.linalg.qr(rng.normal(size=(12,3)))[0]
        g=principal_geodesic(Ps,Pt);F=g.integrated_factor()
        ref,_=quad_vec(lambda t:g.basis(t)@g.basis(t).T,0,1,epsabs=1e-12)
        np.testing.assert_allclose(F@F.T,ref,atol=1e-12)
        np.testing.assert_allclose(g.basis(0)@g.basis(0).T,Ps@Ps.T,atol=1e-12)
        np.testing.assert_allclose(g.basis(1)@g.basis(1).T,Pt@Pt.T,atol=1e-12)
        for t in [0,.2,.7,1]:np.testing.assert_allclose(g.basis(t).T@g.basis(t),np.eye(3),atol=1e-12)

    def test_same_subspace_is_constant_under_rotated_basis(self):
        rng=np.random.default_rng(7);P=np.linalg.qr(rng.normal(size=(10,3)))[0];rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
        g=principal_geodesic(P,P@rotation);F=g.integrated_factor()
        np.testing.assert_allclose(F@F.T,P@P.T,atol=1e-12)
        S=g.sampled_factor(5);np.testing.assert_allclose(S@S.T,P@P.T,atol=1e-12)

    def test_sampling_converges_toward_integrated_projector(self):
        g=principal_geodesic(np.array([[1.],[0.]]),np.array([[0.],[1.]]));F=g.integrated_factor()
        errors=[]
        for n in [5,50,500]:
            S=g.sampled_factor(n);errors.append(np.linalg.norm(S@S.T-F@F.T))
        self.assertLess(errors[2],errors[1]);self.assertLess(errors[1],errors[0])


if __name__=='__main__':unittest.main()
