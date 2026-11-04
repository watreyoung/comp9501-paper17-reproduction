import unittest
import numpy as np
from experiments.concept_classification import concept_event_problem,classify_concepts
from experiments.models import prepare_problem
from experiments.data import make_task
from experiments.core_models import fit,parameters
from mil_pi.qp import qp_session


class ConceptClassificationTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        self.cache=dict(visual=rng.normal(size=(14,4)),text=rng.normal(size=(14,3)),video_ids=np.array([f'v{i}' for i in range(14)]),splits=np.array(['train']*10+['validate']*4),labels=np.array([0]*5+[1]*5+[0,1,0,1]))
        self.data,self.bag=make_task(self.cache,0,seed=0,noise=0,n_bags=2,bag_size=2,evaluation_split='validate')

    def test_zero_concepts_recover_core_kernel_and_predictions(self):
        p=prepare_problem(self.data);p['full_tK']=np.zeros_like(p['K'])
        Z=np.zeros((14,3));q=concept_event_problem(self.cache['visual'],Z,self.cache['video_ids'],self.bag)
        np.testing.assert_allclose(p['K'],q['K'],atol=1e-14)
        np.testing.assert_allclose(p['test_kernel'],q['test_kernel'],atol=1e-14)
        with qp_session('osqp'):
            ref,_=fit('SVM',p,parameters('SVM',1))
            actual,diag=classify_concepts(self.cache['visual'],Z,self.cache['video_ids'],self.bag)
        np.testing.assert_allclose(ref,actual,atol=1e-7)

    def test_target_values_do_not_change_training_kernel(self):
        Z=np.zeros((14,2));a=concept_event_problem(self.cache['visual'],Z,self.cache['video_ids'],self.bag)
        Z[10:]=1e4;b=concept_event_problem(self.cache['visual'],Z,self.cache['video_ids'],self.bag)
        np.testing.assert_array_equal(a['K'],b['K'])
        self.assertEqual(a['kernel_gamma'],b['kernel_gamma'])

    def test_id_alignment_independent_of_row_order(self):
        Z=np.arange(28).reshape(14,2);order=np.random.default_rng(2).permutation(14)
        a=concept_event_problem(self.cache['visual'],Z,self.cache['video_ids'],self.bag)
        b=concept_event_problem(self.cache['visual'][order],Z[order],self.cache['video_ids'][order],self.bag)
        np.testing.assert_array_equal(a['K'],b['K']);np.testing.assert_array_equal(a['test_kernel'],b['test_kernel'])


if __name__=='__main__':unittest.main()
