import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from experiments.mimlboost_baseline import conditional_instances
from experiments.miml_conditional_kernel import ConditionalRBF,compressed_bootstrap


class ConditionalKernelTest(unittest.TestCase):
    def test_indexed_product_matches_original_augmented_rbf(self):
        rng=np.random.default_rng(6)
        for d in [3,65]:
            bags=[rng.normal(size=(n,d)) for n in [2,1,4]]
            X,counts=conditional_instances(bags,3);kernel=ConditionalRBF(bags,3,gamma=.2)
            ids=np.arange(len(X));expected=np.exp(-.2*cdist(X,X,'sqeuclidean'))
            np.testing.assert_allclose(kernel.block(ids,ids),expected,atol=2e-15)
            _,_,group=kernel.decode(ids)
            np.testing.assert_array_equal(group,np.repeat(np.arange(len(counts)),counts))

    def test_compressed_draws_match_repeated_svm_solution(self):
        rng=np.random.default_rng(9);X=rng.normal(size=(9,4));y=np.r_[np.ones(4),-np.ones(5)]
        draws=np.r_[np.arange(9),rng.integers(9,size=30)]
        active,count=compressed_bootstrap(draws,9);self.assertEqual(count.sum(),len(draws))
        a=SVC(C=.7,gamma=.3,tol=1e-10).fit(X[draws],y[draws])
        b=SVC(C=.7,gamma=.3,tol=1e-10).fit(X[active],y[active],sample_weight=count)
        np.testing.assert_allclose(a.decision_function(X),b.decision_function(X),atol=1e-6)


if __name__=='__main__':unittest.main()
