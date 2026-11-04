import unittest
import numpy as np
from scipy.stats import zscore
from experiments.feature_adaptation import row_zscore,fit_sa


class SAPreprocessingTest(unittest.TestCase):
    def test_per_sample_zscore_and_positive_affine_invariance(self):
        rng=np.random.default_rng(67);S=rng.normal(size=(16,7));T=rng.normal(size=(19,7))
        np.testing.assert_allclose(row_zscore(S),zscore(S,axis=1,ddof=0),atol=1e-14)
        scale=rng.uniform(.2,3,size=(16,1));shift=rng.normal(size=(16,1))
        np.testing.assert_allclose(row_zscore(scale*S+shift),row_zscore(S),atol=1e-14)
        model=fit_sa(S,T,3)
        # ICCV2013 Algorithm1: Sa=S Xs Xs' Xt; Ta=T Xt.
        expected=zscore(S,axis=1)@model.source_basis@model.source_basis.T@model.target_basis
        np.testing.assert_allclose(model.transform_source(S),expected,atol=1e-14)
        np.testing.assert_allclose(model.transform_target(T),zscore(T,axis=1)@model.target_basis,atol=1e-14)
        np.testing.assert_array_equal(row_zscore(np.ones((2,7))),np.zeros((2,7)))


if __name__=='__main__':unittest.main()
