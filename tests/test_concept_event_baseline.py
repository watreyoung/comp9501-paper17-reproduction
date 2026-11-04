import unittest
import numpy as np
from experiments.core_models import fit, parameters
from experiments.concept_event_baseline import fit_concept_event


class ConceptEventTest(unittest.TestCase):
    def test_matches_core_svm_and_serialized_coefficients(self):
        rng = np.random.default_rng(48)
        X = rng.normal(size=(12,5));target = rng.normal(size=(7,5))
        p = dict(K=X@X.T, full_tK=np.zeros((12,12)), test_kernel=target@X.T,
                 Kst=X@target.T, n_bags=2, bag_size=3, kernel_gamma=1.)
        for scale in [.1,1.,10.]:
            original,_ = fit('SVM',p,parameters('SVM',scale))
            scores,diag,arrays = fit_concept_event(p,scale=scale)
            np.testing.assert_allclose(scores,original,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(p['test_kernel']@arrays['coef']+arrays['bias'],scores,atol=1e-12)
            self.assertLessEqual(diag['max_constraint_violation'],1e-6)

    def test_invalid_cost_rejected(self):
        with self.assertRaises(ValueError):fit_concept_event({},scale=0.)
