import unittest
import numpy as np
from sklearn.svm import SVC
from experiments.multiview_baselines import fit_svm2k,fit_kcca


class MultiviewBaselineTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(39);X=rng.normal(size=(12,4));T=rng.normal(size=(12,3))
        self.K=X@X.T;self.T=T@T.T;self.y=np.r_[np.ones(6),-np.ones(6)]
        self.c=np.r_[np.full(6,.4),np.ones(6)]

    def test_uncoupled_visual_classifier_matches_svc(self):
        m=fit_svm2k(self.K,self.T,self.y,self.c,self.c,coupling=0)
        ref=SVC(C=1,kernel='precomputed',tol=1e-10).fit(self.K,self.y,sample_weight=self.c)
        np.testing.assert_allclose(m.predict_visual(self.K),ref.decision_function(self.K),atol=1e-5)

    def test_increasing_coupling_reduces_agreement_loss(self):
        low=fit_svm2k(self.K,self.T,self.y,self.c,self.c,coupling=.01)
        high=fit_svm2k(self.K,self.T,self.y,self.c,self.c,coupling=10)
        self.assertLessEqual(high.agreement_loss,low.agreement_loss+1e-6)
        self.assertLess(high.constraint_violation,1e-6)
        self.assertTrue(np.isfinite(high.predict_visual(self.K)).all())

    def test_identical_view_kcca_has_analytic_correlations(self):
        n=len(self.K);H=np.eye(n)-np.ones((n,n))/n;K=H@self.K@H
        ridge=.3;m=fit_kcca(self.K,self.K,3,ridge)
        ev=np.linalg.eigvalsh(K)[::-1][:3]
        np.testing.assert_allclose(m.canonical_correlations,ev/(ev+ridge),atol=1e-12)
        np.testing.assert_allclose(m.visual_coef.T@(K@K+ridge*K)@m.visual_coef,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(m.transform_visual(self.K).mean(axis=0),0,atol=1e-12)


if __name__=='__main__':unittest.main()
