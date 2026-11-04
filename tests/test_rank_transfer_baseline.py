import unittest
import numpy as np
from sklearn.svm import LinearSVC
from experiments.rank_transfer_baseline import rank_svm_primal,fit_rank_transfer


def mirrored_liblinear(D,margins,C):
    # Eq.10/11: scale each pair by rho and weight its unit-margin loss by rho.
    scaled=D/margins[:,None];n=len(D)
    clf=LinearSVC(C=C,loss='hinge',fit_intercept=False,dual=True,tol=1e-10,max_iter=1000000)
    clf.fit(np.r_[scaled,-scaled],np.r_[np.ones(n),-np.ones(n)],sample_weight=np.r_[margins,margins]/2)
    return clf.coef_.ravel()


class RankTransferTest(unittest.TestCase):
    def test_rank_primal_matches_independent_liblinear_reparameterization(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(10,5));i=np.arange(5);j=np.arange(5,10);rho=np.linspace(.2,1.3,5)
        w,obj,violation=rank_svm_primal(X,i,j,rho,.4)
        ref=mirrored_liblinear(X[i]-X[j],rho,.4)
        np.testing.assert_allclose(X@w,X@ref,atol=2e-6)
        self.assertLess(violation,1e-6)
        self.assertAlmostEqual(obj,.5*ref@ref+.4*np.maximum(rho-(X[i]-X[j])@ref,0).sum(),places=7)

    def test_two_stage_constraints_include_same_class_ordering(self):
        T=np.array([[3.,1.],[2.,0.],[1.,-1.],[-1.,2.],[-2.,0.],[-3.,-2.]])
        X=np.c_[T[:,0],np.arange(6)/3];y=np.r_[np.ones(3),-np.ones(3)]
        m=fit_rank_transfer(X,T,y,teacher_C=.2,student_C=.3)
        self.assertEqual(len(m.teacher_pairs),9)
        i,j=m.student_pairs.T
        self.assertTrue(np.all(y[i]>=y[j]));self.assertTrue(np.any(y[i]==y[j]))
        np.testing.assert_allclose(m.student_margins,(T@m.privileged_weights)[i]-(T@m.privileged_weights)[j],atol=1e-12)
        self.assertTrue(np.all(m.student_margins>.1))
        ref=mirrored_liblinear(X[i]-X[j],m.student_margins,.3)
        np.testing.assert_allclose(m.predict(X),X@ref,atol=2e-6)

    def test_no_eligible_student_pairs_yields_valid_zero_ranker(self):
        X=np.eye(4);T=np.zeros((4,2));y=np.array([1,1,-1,-1])
        m=fit_rank_transfer(X,T,y)
        self.assertEqual(len(m.student_pairs),0)
        np.testing.assert_array_equal(m.predict(X),0)


if __name__=='__main__':unittest.main()
