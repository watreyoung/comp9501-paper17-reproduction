import unittest
import numpy as np
from experiments.data import make_task
from experiments.models import prepare_problem
from experiments.literature_dispatch import fit_literature
from experiments.kmm_svm_baseline import fit_kmm_svm


class LiteratureDispatchTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        cache=dict(visual=rng.normal(size=(16,4)),text=rng.normal(size=(16,3)),labels=np.array([0]*6+[1]*6+[0,1,0,1]),splits=np.array(['train']*12+['validate']*4),video_ids=np.array([str(i) for i in range(16)]))
        self.data,_=make_task(cache,0,seed=0,noise=0,n_bags=2,bag_size=2,evaluation_split='validate')
        self.problem=prepare_problem(self.data)

    def test_kmm_matches_standalone_and_target_truth_is_unused(self):
        scores,diag,arrays=fit_literature('KMM+SVM',self.data,self.problem,scale=.1)
        y=np.r_[np.ones(4),-np.ones(4)]
        ref,_,_=fit_kmm_svm(self.problem['K'],self.problem['Kst'],y,np.r_[np.full(4,10.),np.ones(4)])
        np.testing.assert_allclose(scores,ref.predict(self.problem['test_kernel']))
        self.data.test_labels[:]=999
        after,_,_=fit_literature('KMM+SVM',self.data,self.problem,scale=.1)
        np.testing.assert_array_equal(scores,after)
        self.assertFalse(diag['target_labels_used'])

    def test_uniform_cost_control_is_explicit(self):
        scores,diag,arrays=fit_literature('SVM-uniform',self.data,self.problem,scale=.3)
        self.assertEqual(diag['parameters']['C'],.3)
        self.assertTrue(np.isfinite(scores).all())
        with self.assertRaises(ValueError):fit_literature('MILBoost',self.data,self.problem)


if __name__=='__main__':unittest.main()
