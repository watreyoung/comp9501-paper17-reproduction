import unittest
import numpy as np
from experiments.miml_conditionals import conditional_blocks,conditional_resource_estimate
from experiments.mimlboost_baseline import conditional_instances,MIMLBoost


class Threshold:
    def predict(self,X):return np.where(X[:,0]+X[:,-1]>.4,1.,-1.)


class ConditionalBlocksTest(unittest.TestCase):
    def test_stream_matches_dense_for_ragged_bags_and_large_feature_scale(self):
        rng=np.random.default_rng(3)
        for d in [2,60]:
            bags=[rng.normal(size=(n,d)) for n in [3,1,7]]
            dense,counts=conditional_instances(bags,4)
            for size in [1,5,1000]:
                blocks=list(conditional_blocks(bags,4,batch_size=size))
                np.testing.assert_array_equal(np.vstack([X for X,g in blocks]),dense)
                np.testing.assert_array_equal(np.concatenate([g for X,g in blocks]),np.repeat(np.arange(len(counts)),counts))

    def test_ensemble_scores_equal_dense_reference(self):
        bags=[np.array([[0.],[.2],[.8]]),np.array([[-1.]])]
        m=MIMLBoost([Threshold()],np.array([.7]),3,1,[],'test',np.array([]))
        X,counts=conditional_instances(bags,3);scores=.7*Threshold().predict(X)
        ref=np.add.reduceat(scores,np.r_[0,np.cumsum(counts)[:-1]]).reshape(2,3)
        for size in [1,2,9,100]:np.testing.assert_allclose(m.decision_function(bags,batch_size=size),ref,atol=1e-14)

    def test_resource_count_uses_full_word_vocabulary(self):
        r=conditional_resource_estimate(6513,1303,2000,1024)
        self.assertEqual(r['conditional_instances'],13026000)
        self.assertGreater(r['dense_features_plus_bootstrap_bytes'],200_000_000_000)
        self.assertLess(r['streamed_prediction_feature_bytes'],40_000_000)


if __name__=='__main__':unittest.main()
