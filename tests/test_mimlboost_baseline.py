import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from experiments.mimlboost_baseline import conditional_instances,exponential_step,fit_mimlboost,MIMLBoost,ConstantLearner


class MIMLBoostTest(unittest.TestCase):
    def test_scalar_augmentation_and_group_order(self):
        X,n=conditional_instances([np.array([[1.,2.],[3.,4.]]),np.array([[5.,6.]])],3)
        np.testing.assert_array_equal(n,[2,2,2,1,1,1])
        np.testing.assert_array_equal(X[:6,-1],[0,0,.5,.5,1,1])
        np.testing.assert_array_equal(X[6:,:2],[[5,6]]*3)

    def test_coefficient_independent_minimizer(self):
        w=np.array([.7,.3]);e=np.array([.1,.8])
        ref=minimize_scalar(lambda c:np.dot(w,np.exp((2*e-1)*c)),bounds=(0,10),method='bounded')
        c=exponential_step(w,e)
        self.assertAlmostEqual(c,ref.x,places=5)
        updated=w*np.exp((2*e-1)*c);updated/=updated.sum()
        self.assertAlmostEqual(updated.sum(),1.)
        self.assertEqual(exponential_step([.2,.8],[0.,1.]),0.)

    def test_author_zero_ensemble_and_determinism(self):
        bags=[np.array([[-2.],[-2.1]]),np.array([[2.],[2.1]])]
        labels=np.array([[0,1],[1,0]])
        a=fit_mimlboost(bags,labels,C=100,gamma=1,seed=7)
        b=fit_mimlboost(bags,labels,C=100,gamma=1,seed=7)
        self.assertEqual(a.stop_reason,'all_errors_at_most_half_before_store')
        self.assertEqual(len(a.classifiers),0);self.assertEqual(a.history,b.history)
        np.testing.assert_array_equal(a.decision_function(bags),np.zeros((2,2)))
        self.assertTrue(a.predict_labels(bags).all())

    def test_stored_round_weight_recurrence(self):
        rng=np.random.default_rng(32)
        bags=[rng.normal(size=(4,2)) for _ in range(8)]
        labels=rng.integers(2,size=(8,3))
        model=fit_mimlboost(bags,labels,rounds=4,seed=0)
        self.assertEqual(len(model.classifiers),4)
        expected=np.full(24,1/24)
        for h in model.history:
            np.testing.assert_allclose(h['weights_before'],expected)
            self.assertTrue(h['stored'])
            expected*=np.exp((2*np.array(h['errors'])-1)*h['coefficient'])
            expected/=expected.sum()
        np.testing.assert_allclose(model.final_weights,expected)
        self.assertTrue(np.isfinite(model.decision_function(bags)).all())

    def test_outputs_sum_instances_rather_than_mean(self):
        model=MIMLBoost([ConstantLearner(-1)],np.array([2.]),2,1,[],'test',np.array([]))
        np.testing.assert_array_equal(model.decision_function([np.ones((3,1)),np.ones((1,1))]),[[-6,-6],[-2,-2]])


if __name__=='__main__':unittest.main()
