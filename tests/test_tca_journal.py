import unittest
import numpy as np
from experiments.tca_journal import fit_tca_journal


class TCAJournalTest(unittest.TestCase):
    def test_journal_objective_constraint_and_stationarity(self):
        rng=np.random.default_rng(14);X=rng.normal(size=(11,4));K=X@X.T;mu=7.
        model=fit_tca_journal(K,5,3,mu_paper=mu);W=model.coefficients
        e=np.r_[np.full(5,1/5),np.full(6,-1/6)];v=K@e
        A=np.outer(v,v)+mu*np.eye(11);H=np.eye(11)-np.ones((11,11))/11;B=K@H@K
        np.testing.assert_allclose(W.T@B@W,np.eye(3),atol=1e-9)
        np.testing.assert_allclose(A@W,B@W*model.eigenvalues,atol=1e-8)
        self.assertAlmostEqual(np.trace(W.T@A@W),model.eigenvalues.sum(),places=8)


if __name__=='__main__':unittest.main()
