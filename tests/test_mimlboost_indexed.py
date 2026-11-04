import unittest
import numpy as np
from experiments.mimlboost_baseline import fit_mimlboost
from experiments.mimlboost_indexed import fit_indexed_mimlboost


class IndexedBoostTest(unittest.TestCase):
    def test_complete_rounds_match_dense_author_rule_reference(self):
        rng=np.random.default_rng(32);bags=[rng.normal(size=(4,2)) for _ in range(8)];labels=rng.integers(2,size=(8,3))
        a=fit_mimlboost(bags,labels,rounds=4,seed=0)
        b=fit_indexed_mimlboost(bags,labels,rounds=4,seed=0,svm_tolerance=1e-7)
        self.assertEqual(a.stop_reason,b.stop_reason);self.assertEqual(len(a.classifiers),len(b.classifiers))
        for x,y in zip(a.history,b.history):
            np.testing.assert_allclose(x['errors'],y['errors'],atol=1e-14)
            np.testing.assert_allclose(x['weights_before'],y['weights_before'],atol=1e-14)
            self.assertLess(y['solver']['kkt_gap'],1e-7)
        np.testing.assert_allclose(a.coefficients,b.coefficients,atol=1e-12)
        np.testing.assert_allclose(a.decision_function(bags),b.decision_function(bags),atol=1e-12)

    def test_zero_ensemble_and_unconverged_weak_rejected(self):
        bags=[np.array([[-2.],[-2.1]]),np.array([[2.],[2.1]])];labels=np.array([[0,1],[1,0]])
        m=fit_indexed_mimlboost(bags,labels,C=100,seed=7)
        self.assertEqual(len(m.classifiers),0);self.assertTrue(m.predict_labels(bags).all())
        with self.assertRaises(ArithmeticError):fit_indexed_mimlboost(bags,labels,C=100,seed=7,svm_max_iter=1)


if __name__=='__main__':unittest.main()
