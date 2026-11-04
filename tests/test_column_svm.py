import unittest
import numpy as np
from sklearn.svm import SVC
from experiments.column_svm import fit_column_svm
from experiments.miml_conditional_kernel import ConditionalRBF
from experiments.mimlboost_baseline import conditional_instances


class ColumnSVMTest(unittest.TestCase):
    def test_weighted_kernel_dual_matches_libsvm(self):
        rng=np.random.default_rng(15);X=rng.normal(size=(30,5));y=np.r_[np.ones(13),-np.ones(17)]
        K=np.exp(-.3*np.sum((X[:,None]-X[None,:])**2,axis=2));cost=rng.uniform(.1,3,len(y))
        m=fit_column_svm(y,cost,lambda i:K[:,i],diagonal=np.diag(K),tolerance=1e-8)
        ref=SVC(C=1.,kernel='precomputed',tol=1e-10).fit(K,y,sample_weight=cost)
        self.assertEqual(m.stop_reason,'kkt_tolerance');self.assertLess(m.equality_residual,1e-12)
        np.testing.assert_allclose(m.decision_function(K),ref.decision_function(K),atol=2e-6)
        a=np.abs(ref.dual_coef_[0]);sv=ref.support_;coef=ref.dual_coef_[0]
        dual=a.sum()-.5*coef@K[np.ix_(sv,sv)]@coef
        self.assertAlmostEqual(m.dual_objective,dual,places=7)

    def test_indexed_conditional_oracle_and_cap(self):
        rng=np.random.default_rng(6);bags=[rng.normal(size=(3,2)) for _ in range(4)]
        kernel=ConditionalRBF(bags,2,gamma=.7);X,counts=conditional_instances(bags,2)
        ids=np.arange(len(X));y=np.repeat([1,-1,-1,1,1,-1,-1,1],counts)
        column=lambda i:kernel.block(ids,np.array([i])).ravel()
        m=fit_column_svm(y,np.ones(len(y)),column,diagonal=np.ones(len(y)),tolerance=1e-7)
        K=kernel.block(ids,ids);ref=SVC(C=1.,gamma=.7,tol=1e-10).fit(X,y)
        np.testing.assert_allclose(m.decision_function(K),ref.decision_function(X),atol=2e-6)
        limited=fit_column_svm(y,np.ones(len(y)),column,diagonal=np.ones(len(y)),max_iter=1)
        self.assertEqual(limited.stop_reason,'max_iter');self.assertGreater(limited.kkt_gap,1e-6)

    def test_compiled_backend_matches_every_capped_trajectory(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(24,3));K=np.exp(-.3*np.sum((X[:,None]-X[None,:])**2,axis=2))
        y=np.r_[np.ones(9),-np.ones(15)];cost=rng.uniform(.2,2,24)
        for cap in [1,2,5,20,10000]:
            options=dict(diagonal=np.diag(K),max_iter=cap,tolerance=1e-8)
            a=fit_column_svm(y,cost,lambda i:K[:,i],backend='numpy',**options)
            b=fit_column_svm(y,cost,lambda i:K[:,i],backend='numba',**options)
            np.testing.assert_array_equal(a.alpha,b.alpha)
            self.assertEqual(a.kkt_gap,b.kkt_gap);self.assertEqual(a.bias,b.bias)
            self.assertEqual(a.stop_reason,b.stop_reason)

    def test_tiny_asymmetric_costs_preserve_equality_and_interior_bias(self):
        K=np.eye(2);cost=np.array([1e-15,2e-15])
        m=fit_column_svm([1,-1],cost,lambda i:K[:,i],diagonal=np.ones(2))
        np.testing.assert_allclose(m.alpha,[1e-15,1e-15],rtol=1e-12,atol=0)
        self.assertLess(m.equality_residual,1e-28)
        self.assertAlmostEqual(m.bias,-1.,places=12)

    def test_identical_opposite_label_points_zero_curvature(self):
        K=np.ones((2,2));m=fit_column_svm([1,-1],[2,2],lambda i:K[:,i])
        np.testing.assert_allclose(m.alpha,[2,2]);self.assertLessEqual(m.kkt_gap,1e-6)


if __name__=='__main__':unittest.main()
