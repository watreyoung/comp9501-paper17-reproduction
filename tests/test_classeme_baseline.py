import unittest
import numpy as np
from experiments.classeme_baseline import variable_bag_update,train_variable_mi,fit_classeme
from mil_pi.paper_models import feasible_update,train_mi
from mil_pi.qp import qp_session


class ClassemeTest(unittest.TestCase):
    def test_variable_repair_at_least_one_and_negative_fixed(self):
        s=np.array([-.4,-.1,-.8,2.,1.,-3.,9.,9.])
        y=variable_bag_update(s,[1,2,3])
        np.testing.assert_array_equal(y,[1,1,-1,1,1,-1,-1,-1])
        equal=np.array([-2.,-1.,3.,4.,-1.,-2.,7.])
        np.testing.assert_array_equal(variable_bag_update(equal,[2,2]),feasible_update(equal,4,2,.5))

    def test_equal_bag_solution_matches_existing_equation_solver(self):
        rng=np.random.default_rng(5);X=rng.normal(size=(10,3));K=X@X.T
        with qp_session('osqp'):
            a=train_variable_mi(K,[2,2],C=1.)
            b=train_mi(K,np.zeros_like(K),4,2,c=np.ones(10),sigma=.5)
        np.testing.assert_allclose(a.predict(K),b.predict(K),atol=1e-8)
        np.testing.assert_array_equal(a.labels,b.labels)
        self.assertEqual(a.stop_reason,b.stop_reason)

    def test_linear_representation_ignores_target_captions_and_visuals(self):
        X=np.array([[-2.,0],[-1.,0],[1.,0],[2.,0],[np.nan,np.nan]])
        T=np.array([[1,0],[1,0],[0,1],[0,1],[np.nan,np.nan]])
        m=fit_classeme(X,T,['train']*4+['test'],bag_size=2)
        self.assertEqual(m.weights.shape,(2,2))
        score=m.decision_function(X[:4]);self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[:2,0].mean(),score[2:,0].mean())
        self.assertEqual(m.augmented_features(X[:4]).shape,(4,4))
        for d in m.diagnostics:
            self.assertLess(d['max_constraint_violation'],1e-6)
            used=sum(d['positive_rows']+d['negative_rows'],[])
            self.assertEqual(len(used),len(set(used)))
            self.assertNotIn(4,used)


if __name__=='__main__':unittest.main()
