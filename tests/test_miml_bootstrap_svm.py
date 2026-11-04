import unittest
import numpy as np
from sklearn.svm import SVC
from experiments.miml_conditional_kernel import ConditionalRBF
from experiments.mimlboost_baseline import conditional_instances
from experiments.miml_bootstrap_svm import fit_conditional_bootstrap


class BootstrapSVMTest(unittest.TestCase):
    def test_actual_draws_match_original_unweighted_repeated_augmented_svm(self):
        rng=np.random.default_rng(51);bags=[rng.normal(size=(n,3)) for n in [2,3,1]]
        labels=np.array([[0,1,0],[1,0,1],[0,1,1]])
        X,counts=conditional_instances(bags,3);y=np.repeat((2*labels-1).ravel(),counts)
        draws=np.r_[np.arange(len(X)),rng.integers(len(X),size=30)]
        kernel=ConditionalRBF(bags,3,gamma=.4)
        model,ids=fit_conditional_bootstrap(kernel,labels,draws,C=.7,tolerance=1e-8)
        self.assertEqual(model.stop_reason,'kkt_tolerance')
        reference=SVC(C=.7,gamma=.4,tol=1e-10).fit(X[draws],y[draws])
        scores=model.decision_function(kernel.block(np.arange(len(X)),ids))
        np.testing.assert_allclose(scores,reference.decision_function(X),atol=2e-6)
        self.assertLess(model.equality_residual,1e-10)
        limited,_=fit_conditional_bootstrap(kernel,labels,draws,max_iter=1)
        self.assertEqual(limited.stop_reason,'max_iter')


if __name__=='__main__':unittest.main()
