import unittest
import numpy as np
from experiments.dasvm_baseline import select_margin_targets,select_source_deletions,update_target_ages,source_cost,target_costs


class DASVMTransitionsTest(unittest.TestCase):
    def test_margin_selection_excludes_confident_outside_and_orders_boundaries(self):
        scores=np.array([2.,.1,.9,-.2,-.8,-3.,1.,-1.,0.])
        up,low=select_margin_targets(scores,np.ones(9,bool),2)
        np.testing.assert_array_equal(up,[6,2]);np.testing.assert_array_equal(low,[7,4])
        np.testing.assert_array_equal(select_source_deletions(scores,np.ones(9,bool),2,1,2),[0,6,5])
        np.testing.assert_array_equal(select_source_deletions(scores,np.ones(9,bool),0,0,1),[0,5])

    def test_inconsistent_return_unlabeled_survivors_age_new_start_one(self):
        labels,ages,bad=update_target_ages([1,-1,1,0,0],[2,3,4,0,0],[-.1,-.4,.3,-.8,.9],[3],4)
        np.testing.assert_array_equal(labels,[0,-1,1,-1,0]);np.testing.assert_array_equal(ages,[0,4,4,1,0]);np.testing.assert_array_equal(bad,[0])
        with self.assertRaises(ValueError):update_target_ages([1,0],[2,0],[-1,1],[0],4)

    def test_quadratic_schedules_at_endpoints_and_interior(self):
        self.assertEqual(source_cost(0,10,.1,5),10)
        self.assertAlmostEqual(source_cost(2,10,.1,5),10-9.9*4/25)
        self.assertAlmostEqual(source_cost(5,10,.1,5),.1)
        self.assertEqual(source_cost(20,10,.1,5),.1)
        np.testing.assert_allclose(target_costs([1,3,5],10,.1,5),[.1,.1+4.9/4,5.])




class DASVMTrainingTest(unittest.TestCase):
    def problem(self):
        rng=np.random.default_rng(13)
        S=np.r_[rng.normal(-1,.4,(8,2)),rng.normal(1,.4,(8,2))]
        T=np.r_[rng.normal(-.8,.5,(20,2)),rng.normal(.8,.5,(20,2))]
        X=np.r_[S,T];K=np.exp(-.5*np.sum((X[:,None]-X[None,:])**2,axis=2))
        return K,np.r_[np.full(8,-1),np.ones(8)]

    def test_converges_target_only_and_final_svm_reconstructs(self):
        from experiments.dasvm_baseline import fit_dasvm
        from sklearn.svm import SVC
        K,y=self.problem();m=fit_dasvm(K,y,C=1.,initial_target_cost=.1,max_iter=100)
        self.assertEqual(m.stop_reason,'target_only_empirical_convergence')
        self.assertTrue(np.all(m.training_indices>=len(y)))
        final=m.history[-1];self.assertEqual(final['source_remaining'],[])
        active=np.array(final['active_target_indices']);indices=len(y)+active
        labels=np.array(final['active_target_labels']);cost=np.array(final['training_costs'])
        reference=SVC(C=1,kernel='precomputed',tol=1e-7,max_iter=200000).fit(K[np.ix_(indices,indices)],labels,sample_weight=cost)
        np.testing.assert_allclose(m.decision_function(K),reference.decision_function(K[:,indices]),atol=1e-12)
        remaining=set(range(len(y)))
        for h in m.history:
            self.assertTrue(set(h['source_deleted'])<=remaining)
            remaining-=set(h['source_deleted'])
            self.assertEqual(remaining,set(h['source_remaining']))
            self.assertTrue(np.all(np.array(h['training_costs'])>0))

    def test_iteration_cap_remains_explicit(self):
        from experiments.dasvm_baseline import fit_dasvm
        K,y=self.problem();m=fit_dasvm(K,y,C=1.,initial_target_cost=.1,max_iter=1)
        self.assertEqual(m.stop_reason,'max_iter')
        self.assertGreater(len(m.history[-1]['source_remaining']),0)
        self.assertTrue(np.isfinite(m.decision_function(K)).all())

class DASVMConstantTest(unittest.TestCase):
    def test_constant_single_class_has_zero_primal_loss(self):
        from experiments.dasvm_baseline import ConstantDASVMClassifier
        for label in [-1.,1.]:
            score=ConstantDASVMClassifier(label).decision_function(np.zeros((4,3)))
            np.testing.assert_array_equal(np.maximum(1-label*score,0),np.zeros(4))


if __name__=='__main__':unittest.main()
