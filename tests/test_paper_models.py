import unittest
import itertools
import numpy as np
from sklearn.svm import SVC
from mil_pi.qp import qp_session,solve_qp
from mil_pi.paper_models import dual_problem,train_mi,train_cpb,bag_oracle,bag_patterns

class PaperEquationTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12);self.X=rng.normal(size=(8,3));self.T=rng.normal(size=(8,2))
        self.K=self.X@self.X.T;self.TK=self.T@self.T.T;self.y=np.r_[np.ones(4),-np.ones(4)]
        self.c=np.r_[np.full(4,.2),np.ones(4)]

    def test_dual_bias_multiplier_matches_svc(self):
        with qp_session('osqp'):
            m=dual_problem(self.K,self.TK,self.y,np.ones(8),self.c,np.zeros(8,bool))
        ref=SVC(C=1.,kernel='precomputed',tol=1e-10).fit(self.K,self.y,sample_weight=self.c)
        np.testing.assert_allclose(m.predict(self.K),ref.decision_function(self.K),atol=2e-6)

    def test_scaled_equality_multiplier_is_unscaled(self):
        with qp_session('osqp'):
            r=solve_qp(np.eye(1),[0.],A_eq=[[100.]],b_eq=[200.])
        self.assertAlmostEqual(r.equality_multipliers[0],-.02,places=7)

    def test_pi_dual_matches_independently_formed_primal(self):
        # Variables [w(3),b,u(2),b*,eta(4)], independent feature-space primal.
        P=np.arange(8)<4;gamma=3.;nvar=11
        H=np.zeros((nvar,nvar));H[:3,:3]=np.eye(3);H[4:6,4:6]=gamma*np.eye(2)
        f=np.zeros(nvar);f[4:6]=self.c[P]@self.T[P];f[6]=self.c[P].sum();f[7:]=self.c[~P]
        A=np.zeros((12,nvar));rhs=np.r_[-np.ones(8),np.zeros(4)]
        for i in range(8):
            A[i,:3]=-self.y[i]*self.X[i];A[i,3]=-self.y[i]
            if i<4:A[i,4:6]=-self.T[i];A[i,6]=-1
            else:A[i,7+i-4]=-1
        A[8:,4:6]=-self.T[P];A[8:,6]=-1
        with qp_session('osqp'):
            primal=solve_qp(H+1e-10*np.eye(nvar),f,A_ub=A,b_ub=rhs,
                            lb=np.r_[np.full(7,-np.inf),np.zeros(4)],max_iter=200000)
            dual=dual_problem(self.K,self.TK,self.y,np.ones(8),self.c,P,gamma=gamma)
        self.assertAlmostEqual(primal.objective,dual.objective,places=5)
        np.testing.assert_allclose(self.X@primal.x[:3]+primal.x[3],dual.predict(self.K),atol=1e-5)

    def test_da_constraints_and_mu_mapping(self):
        C2=.1
        with qp_session('osqp'):
            model=dual_problem(self.K,self.TK,self.y,np.ones(8),self.c,np.arange(8)<4,
                               gamma=3.,C2=C2,lam=100.,Kst=self.K[:,:3])
        theta=np.array(model.diagnostics['source_weights'])
        self.assertAlmostEqual(theta.sum(),8.,places=6)
        self.assertTrue(np.all(model.alpha<=C2*theta+1e-7))

    def test_no_pi_is_text_independent_and_mi_returns_feasible_labels(self):
        with qp_session('osqp'):
            a=train_mi(self.K,self.TK,4,2,c=self.c,sigma=.5)
            b=train_mi(self.K,100*self.TK,4,2,c=self.c,sigma=.5)
        np.testing.assert_array_equal(a.coef,b.coef)
        self.assertTrue(np.all(a.labels[4:]==-1))
        self.assertTrue(all((a.labels[i:i+2]>0).sum()>=1 for i in [0,2]))

    def test_oracle_coordinate_optimum_and_feasibility(self):
        alpha=np.arange(1,9)/10
        y,diag=bag_oracle(self.K+1.,alpha,4,2,.5)
        M=(self.K+1.)*np.outer(alpha,alpha)
        for start in [0,2]:
            for p in bag_patterns(2,1).T:
                other=y.copy();other[start:start+2]=p
                self.assertLessEqual(other@M@other,y@M@y+1e-8)
        self.assertEqual(diag['stop_reason'],'coordinate_stable')
        self.assertFalse(diag['global_optimum_claimed'])

    def test_cpb_single_label_kernel_matches_augmented_svm(self):
        with qp_session('osqp'):
            model=train_cpb(self.K,self.TK,4,2,c=self.c,pi='none',sigma=1.,max_iter=3)
            ref=dual_problem(self.K,self.TK,self.y,np.ones(8),self.c,np.zeros(8,bool),
                             label_kernel=(self.K+1.)*np.outer(self.y,self.y),regularized_bias=True)
        np.testing.assert_allclose(model.predict(self.K), (self.K+1.)@(ref.alpha*self.y),atol=2e-4)
        self.assertAlmostEqual(model.objective,ref.objective,places=6)
        self.assertAlmostEqual(model.bias,model.coef.sum(),places=10)

    def test_cpb_mkl_and_cut_status_complete(self):
        with qp_session('osqp'):
            m=train_cpb(self.K,self.TK,4,2,c=self.c,pi='partial',gamma=3.,sigma=.5,max_iter=10)
        d=np.array(m.diagnostics['kernel_weights']);Y=np.array(m.diagnostics['label_vectors'])
        self.assertAlmostEqual(d.sum(),1.,places=8)
        self.assertTrue(np.all(d>=0))
        np.testing.assert_allclose(m.coef,m.alpha*(Y@d),atol=1e-10)
        self.assertTrue(m.history)
        self.assertNotEqual(m.stop_reason,'max_iter')

    def test_restricted_conic_mkl_matches_independent_nested_dual(self):
        from scipy.optimize import minimize_scalar
        from mil_pi.paper_models import restricted_cpb
        Y=np.c_[self.y,self.y.copy()];Y[0,1]=-1
        Q=self.K+1.;mask=np.arange(8)<4
        with qp_session('osqp'):
            def objective(w):
                mixed=Q*((Y*np.array([w,1-w]))@Y.T)
                return dual_problem(self.K,self.TK,np.ones(8),np.ones(8),self.c,mask,
                    gamma=3.,label_kernel=mixed,regularized_bias=True).objective
            ref=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-8})
            model,d,_,diag=restricted_cpb(Q,self.TK,Y,self.c,mask,gamma=3.)
        self.assertAlmostEqual(model.objective,ref.fun,places=5)
        self.assertAlmostEqual(d.sum(),1.,places=8)
        self.assertLessEqual(diag['constraint_violation'],1e-6)

if __name__=='__main__':unittest.main()
