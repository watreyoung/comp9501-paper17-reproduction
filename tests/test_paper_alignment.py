import unittest
import numpy as np
from mil_pi.algorithms import s_mil_pi
from mil_pi.data import BagDataset
from mil_pi.qp import qp_session
from experiments.models import prepare_problem
from experiments.alignment_models import aligned_predict,correcting_svm,kmm_weights
from experiments.paper_alignment import candidate,domain_cache
from threadpoolctl import threadpool_limits

class PaperAlignmentTest(unittest.TestCase):
    def problem(self):
        rng=np.random.default_rng(4)
        data=BagDataset([rng.normal(size=(4,5)) for _ in range(2)],
                        [rng.normal(size=(4,5)) for _ in range(2)],
                        [rng.normal(size=(3,5)) for _ in range(2)],
                        [rng.normal(size=(3,5)) for _ in range(2)],rng.normal(size=(4,6)),np.array([1,-1]*3))
        p=prepare_problem(data);text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1)
        p['full_tK']=text.T@text
        return p

    def test_partial_dual_matches_existing_smil_pi_ranking(self):
        p=self.problem();params=candidate(1.)
        with threadpool_limits(limits=1):
            scores,_=aligned_predict('sMIL-PI',p,params)
            with qp_session('osqp'):
                model=s_mil_pi(p['K'],p['tK'],[5]*2,[5]*2,dict(svm_C=100.,gamma=100.,rho=.6,solver_max_iter=100000))
            expected=model.decision_function(p['test_kernel'])
        np.testing.assert_allclose(scores-scores.mean(),expected-expected.mean(),atol=1e-7)

    def test_full_pi_corrects_negative_samples_and_no_pi_ignores_text(self):
        p=self.problem();params=candidate(1.)
        for method in ['SVM','sMIL']:
            a,_=aligned_predict(method,p,params)
            changed=dict(p,full_tK=np.zeros_like(p['full_tK']))
            b,_=aligned_predict(method,changed,params)
            np.testing.assert_array_equal(a,b)
        a,_=aligned_predict('sMIL-PI-full',p,params)
        T=p['full_tK'].copy();T[10:,:]=0;T[:,10:]=0
        b,_=aligned_predict('sMIL-PI-full',dict(p,full_tK=T),params)
        self.assertGreater(np.max(np.abs(a-b)),1e-6)

    def test_kmm_identity_has_unit_weights(self):
        with qp_session('osqp'):
            w=kmm_weights(np.eye(5),np.eye(5))
        np.testing.assert_allclose(w,np.ones(5),atol=1e-7)

    def test_domain_selection_does_not_read_test_labels_or_fit_on_test(self):
        rng=np.random.default_rng(3);n=240
        c=dict(visual=rng.normal(size=(n,3)),text=np.ones((n,2)),labels=np.tile([0,1],n//2),
               video_ids=np.array([f'v{i}' for i in range(n)]),splits=np.array(['train']*160+['validate']*40+['test']*40))
        with threadpool_limits(limits=1):
            captions={v:('testonly secretword' if i>=200 else 'alpha beta gamma') for i,v in enumerate(c['video_ids'])}
            _,a=domain_cache(c,'visual_cluster',captions=captions)
            changed={k:v.copy() for k,v in c.items()};changed['labels'][-40:]=19
            _,b=domain_cache(changed,'visual_cluster',captions=captions)
        self.assertEqual(a,b)
        self.assertNotIn('testonly',a['text_vocabulary'])
        self.assertNotIn('secretword',a['text_vocabulary'])
        groups=[set(a['split_ids'][s]) for s in ['train','validate','test']]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

if __name__=='__main__':unittest.main()
