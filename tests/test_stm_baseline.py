import unittest
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import LinearSVC
from mil_pi.qp import qp_session
from experiments.stm_baseline import quadratic_kernel_svm,selective_weights,fit_stm


class STMTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(24);self.X=rng.normal(size=(14,4));self.Z=rng.normal(size=(11,4))+.3
        self.K=self.X@self.X.T;self.Kst=self.X@self.Z.T
        self.y=np.r_[np.ones(7),-np.ones(7)]

    def test_quadratic_classifier_matches_independent_liblinear(self):
        cost=np.linspace(.1,.8,14)
        coef,loss,obj,violation=quadratic_kernel_svm(self.K,self.y,cost)
        ref=LinearSVC(C=1.,loss='squared_hinge',fit_intercept=False,dual=True,tol=1e-11,max_iter=100000).fit(self.X,self.y,sample_weight=cost)
        np.testing.assert_allclose(self.K@coef,ref.decision_function(self.X),atol=2e-6)
        self.assertAlmostEqual(obj,.5*np.sum(ref.coef_**2)+cost@np.maximum(1-self.y*ref.decision_function(self.X),0)**2,places=7)

    def test_zero_initial_loss_is_kmm_reference(self):
        n=14;kappa=n*self.Kst.mean(axis=1);H=self.K+1e-8*np.eye(n)
        with qp_session('osqp'):s,_=selective_weights(self.K,kappa,np.zeros(n),np.ones(n),1.,3.,0.)
        ref=minimize(lambda x:.5*x@H@x-kappa@x,np.ones(n),jac=lambda x:H@x-kappa,
            bounds=[(0,3)]*n,constraints=[dict(type='eq',fun=lambda x:x.sum()-n,jac=lambda x:np.ones(n))],
            method='SLSQP',options={'ftol':1e-11,'maxiter':2000})
        self.assertTrue(ref.success)
        self.assertAlmostEqual(.5*s@H@s-kappa@s,ref.fun,places=7)

    def test_alternating_objective_and_weight_constraints(self):
        m=fit_stm(self.K,self.Kst,self.y,C=.3,lambda_=2.,bound=3,max_iter=30)
        self.assertLess(m.max_constraint_violation,1e-6)
        self.assertAlmostEqual(m.weights.sum(),14,places=6)
        self.assertGreaterEqual(m.weights.min(),0);self.assertLessEqual(m.weights.max(),3)
        for a,b in zip(m.history,m.history[1:]):self.assertLessEqual(b['objective'],a['objective']+1e-6*max(1.,abs(a['objective'])))
        self.assertTrue(np.isfinite(m.predict(self.Kst.T)).all())


if __name__=='__main__':unittest.main()
