import unittest
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
from experiments.kmm_svm_baseline import fit_kmm_svm


class KMMSVMTest(unittest.TestCase):
    def test_weights_against_independent_optimizer_and_weighted_libsvm(self):
        rng=np.random.default_rng(91);X=rng.normal(size=(12,5));T=rng.normal(size=(8,5))+.3
        def rbf(A,B):return np.exp(-.4*np.sum((A[:,None]-B[None,:])**2,axis=2))
        K=rbf(X,X);Kst=rbf(X,T);y=np.r_[np.ones(6),-np.ones(6)];cost=np.r_[np.full(6,2.),np.ones(6)]
        model,w,diag=fit_kmm_svm(K,Kst,y,cost,bound=3.)
        H=K+1e-6*np.eye(12);f=-12*Kst.mean(axis=1)
        ref=minimize(lambda b:.5*b@H@b+f@b,np.ones(12),jac=lambda b:H@b+f,bounds=[(0,3)]*12,
                     constraints=[dict(type='eq',fun=lambda b:b.sum()-12,jac=lambda b:np.ones(12))],method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
        self.assertTrue(ref.success);np.testing.assert_allclose(w,ref.x,atol=2e-5)
        # Tiny dual stabilization differs from Libsvm by <=1e-8; compare margins.
        svm=SVC(C=1,kernel='precomputed',tol=1e-10).fit(K,y,sample_weight=cost*w)
        np.testing.assert_allclose(model.predict(K),svm.decision_function(K),atol=2e-6)
        self.assertAlmostEqual(w.sum(),12,places=6)
        self.assertLessEqual(w.max(),3);self.assertGreaterEqual(w.min(),0)
        self.assertLess(diag['max_constraint_violation'],1e-6)

    def test_invalid_target_shape_rejected(self):
        with self.assertRaises(ValueError):fit_kmm_svm(np.eye(2),np.empty((2,0)),[1,-1],[1,1])


if __name__=='__main__':unittest.main()
