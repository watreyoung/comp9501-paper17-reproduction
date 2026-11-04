import unittest
import numpy as np
from experiments.hausdorff_blocks import blocked_hausdorff
from experiments.mimlsvm_baseline import bag_distances,fit_mimlsvm


class HausdorffBlocksTest(unittest.TestCase):
    def test_ragged_blocked_matches_original_definition(self):
        rng=np.random.default_rng(92)
        bags=[rng.normal(size=(n,7)) for n in [1,5,2,3]]
        reference=[rng.normal(size=(n,7)) for n in [4,1,6]]
        expected=bag_distances(bags,reference)
        for size in [1,2,20]:np.testing.assert_allclose(blocked_hausdorff(bags,reference,block_bags=size),expected,atol=1e-14)
        D=blocked_hausdorff(bags,bags)
        np.testing.assert_array_equal(np.diag(D),np.zeros(len(bags)))
        np.testing.assert_allclose(D,D.T,atol=1e-14)

    def test_complete_fit_matches_reference_distance_backend(self):
        rng=np.random.default_rng(8);bags=[rng.normal(size=(n,3)) for n in [1,3,2,4,2,3]]
        labels=rng.integers(2,size=(6,4))
        a=fit_mimlsvm(bags,labels,medoid_fraction=.5,distance_backend='blocked')
        b=fit_mimlsvm(bags,labels,medoid_fraction=.5,distance_backend='reference')
        np.testing.assert_array_equal(a.medoid_indices,b.medoid_indices)
        np.testing.assert_allclose(a.decision_function(bags),b.decision_function(bags),atol=1e-10)

    def test_padding_zero_is_not_an_instance(self):
        bags=[np.array([[3.],[4.]]),np.array([[10.]])]
        np.testing.assert_allclose(blocked_hausdorff(bags,[np.array([[0.]])]),[[4.],[10.]])


if __name__=='__main__':unittest.main()
