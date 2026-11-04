import unittest
from unittest.mock import patch
import numpy as np
from experiments.data import make_bag_indices
from experiments.prepare_msrvtt import fit_source_text, assign_splits
from mil_pi.qp import qp_session, solve_qp
from mil_pi.algorithms import mi_svm_pi, mil_cpb_pi, _update_positive_labels, _solve_smil_pi_qp


class ExperimentProtocolTest(unittest.TestCase):
    def test_bag_projection_preserves_global_threshold_and_score_translation(self):
        initial = np.array([1., 1., -1., -1., -1.])
        scores = np.array([10., 9., 8., 7., 6.])
        expected = np.array([1., 1., 1., -1., -1.])
        for shift in [0., -100.]:
            actual = _update_positive_labels(initial, scores+shift, np.array([5]), np.array([3]))
            np.testing.assert_array_equal(actual, expected)

    def test_smil_pi_is_invariant_to_joint_text_kernel_and_gamma_rescaling(self):
        k = np.eye(4)
        t = np.zeros((4,4))
        t[:2,:2] = [[2.,1.],[1.,2.]]
        y = np.array([1.,1.,-1.,-1.])
        with qp_session("osqp"):
            a, ra = _solve_smil_pi_qp(k,t,y,np.array([.2,.2,1.,1.]),np.ones(4),2.,2,2,solver_max_iter=10000)
            b, rb = _solve_smil_pi_qp(k,t*4,y,np.array([.2,.2,1.,1.]),np.ones(4),8.,2,2,solver_max_iter=10000)
        np.testing.assert_allclose(a,b,atol=1e-8)
        self.assertAlmostEqual(ra.objective,rb.objective,places=8)

    def test_cpb_iteration_cap_keeps_returned_coefficients_and_label_set_consistent(self):
        p=dict(svm_C=.1,svm_C_star=1.,gamma=1.,pos_ratio=.5,max_iter=1,solver_max_iter=10000)
        with qp_session("osqp"):
            model=mil_cpb_pi(np.eye(4),np.diag([1.,1.,0.,0.]),
                [np.arange(2),np.arange(2,4)],np.array([1.,-1.]),p)
        self.assertEqual(model.y_set.shape[1],len(model.d))
        np.testing.assert_allclose(model.effective_labels,model.y_set@model.d)

    def test_instance_pi_subproblems_remain_convex_with_correlated_text(self):
        # A nonzero off-diagonal PI block and a missing bottom-right block
        # makes the QP indefinite, even though shape-only smoke tests pass.
        k = np.eye(4)
        t = np.zeros((4, 4))
        t[:2, :2] = [[10, 9], [9, 10]]
        inspected = []
        def convex_solve(h, f, **kwargs):
            self.assertGreaterEqual(np.linalg.eigvalsh(h).min(), -1e-9)
            inspected.append(True)
            return solve_qp(h, f, **kwargs)
        p = dict(svm_C=0.1, svm_C_star=1, gamma=1, pos_ratio=0.5,
                 max_iter=5, solver_max_iter=10000)
        with qp_session("osqp"), patch("mil_pi.algorithms.solve_qp", side_effect=convex_solve):
            mi_svm_pi(k, t, [2], [2], p)
            mil_cpb_pi(k, t, [np.arange(2), np.arange(2, 4)], np.array([1, -1]), p)
        self.assertGreaterEqual(len(inspected), 2)

    def test_explicit_video_splits_are_complete_and_disjoint(self):
        ids = ["a", "b", "c"]
        np.testing.assert_array_equal(assign_splits(ids, {"a"}, {"b"}, {"c"}),
                                      ["train", "validate", "test"])
        with self.assertRaises(ValueError):
            assign_splits(ids, {"a", "c"}, {"b"}, {"c"})
        with self.assertRaises(ValueError):
            assign_splits(ids, {"a"}, {"b"}, set())

    def test_bags_have_exact_noise_without_reused_or_heldout_videos(self):
        labels = np.r_[np.ones(100), np.zeros(200)]
        train = np.r_[np.arange(70), np.arange(100, 240)]
        p, n = make_bag_indices(labels, train, 1, seed=7, noise=0.4, n_bags=10)
        self.assertTrue(np.all(labels[p].sum(axis=1) == 3))
        self.assertTrue(np.all(labels[n] == 0))
        self.assertEqual(np.unique(np.r_[p.ravel(), n.ravel()]).size, 100)
        self.assertTrue(set(np.r_[p.ravel(), n.ravel()]) <= set(train))
        again = make_bag_indices(labels, train, 1, seed=7, noise=0.4, n_bags=10)
        np.testing.assert_array_equal(p, again[0])

    def test_target_words_never_enter_vocabulary_or_text_features(self):
        x, vocabulary = fit_source_text(["person cooking food", "dog running outside", "targetonlysecret"],
                                        np.array([True, True, False]))
        self.assertNotIn("targetonlysecret", vocabulary)
        self.assertEqual(np.count_nonzero(x[2]), 0)

    def test_osqp_matches_analytic_solution_and_original_solver(self):
        args = dict(A_eq=np.ones((1, 2)), b_eq=np.array([1.0]),
                    lb=np.zeros(2), ub=np.ones(2), max_iter=10000)
        # min x0^2 + 2*x1^2 subject to x0+x1=1 gives (2/3, 1/3).
        results = []
        for backend in ["osqp", "slsqp"]:
            with qp_session(backend) as trace:
                r = solve_qp(np.diag([2.0, 4.0]), np.zeros(2), **args)
                np.testing.assert_allclose(r.x, [2/3, 1/3], atol=1e-6)
                self.assertEqual(len(trace), 1)
                self.assertTrue(trace[0]["success"])
                results.append(r.objective)
        self.assertAlmostEqual(*results, places=6)


if __name__ == "__main__":
    unittest.main()
