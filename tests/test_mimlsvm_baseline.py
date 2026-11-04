import unittest
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from experiments.mimlsvm_baseline import hausdorff,bag_distances,kmedoids_paper,fit_mimlsvm,MIMLSVM


class MIMLSVMTest(unittest.TestCase):
    def test_distance_matches_independent_directed_hausdorff(self):
        rng=np.random.default_rng(9);A=rng.normal(size=(4,3));B=rng.normal(size=(7,3))
        ref=max(directed_hausdorff(A,B)[0],directed_hausdorff(B,A)[0])
        self.assertAlmostEqual(hausdorff(A,B),ref,places=12)
        self.assertEqual(hausdorff(A,A),0.)

    def test_single_medoid_matches_exhaustive_optimum(self):
        bags=[np.array([[x]]) for x in [0.,1.,8.]];D=bag_distances(bags,bags)
        m,h,reason=kmedoids_paper(D,1,seed=12)
        self.assertEqual(m[0],int(np.argmin(D.sum(axis=1))))
        self.assertEqual(m[0],1);self.assertEqual(reason,'medoids_stable')
        for a,b in zip(h,h[1:]):self.assertLessEqual(b['assignment_objective'],a['assignment_objective']+1e-12)

    def test_author_floor_fraction_and_default_gamma(self):
        bags=[np.array([[float(x)]]) for x in range(9)]
        labels=np.array([[x%2] for x in range(9)])
        model=fit_mimlsvm(bags,labels)
        self.assertEqual(len(model.medoid_indices),1)
        self.assertEqual(model.classifiers[0].gamma,.2)

    def test_zero_scores_select_only_first_maximum(self):
        model=MIMLSVM([],np.array([],int),[],{},[],'test')
        model.decision_function=lambda bags: np.array([[0.,0.,-1.],[1.,0.,-1.]])
        np.testing.assert_array_equal(model.predict_labels([None,None]),[[True,False,False],[True,False,False]])

    def test_word_scores_and_tcriterion_with_constant_labels(self):
        bags=[np.array([[x],[x+.1]]) for x in [-2.,-1.,1.,2.]]
        labels=np.array([[0,1,0],[0,1,0],[1,1,0],[1,1,0]])
        m=fit_mimlsvm(bags,labels,medoid_fraction=.5,C=10)
        scores=m.decision_function(bags)
        self.assertEqual(scores.shape,(4,3));np.testing.assert_array_equal(scores[:,1],1);np.testing.assert_array_equal(scores[:,2],-1)
        prediction=m.predict_labels(bags);self.assertTrue(np.all(prediction.sum(axis=1)>=1))
        np.testing.assert_array_equal(prediction[:,0],labels[:,0].astype(bool))


if __name__=='__main__':unittest.main()
