from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from mil_pi.algorithms import (
    mil_cpb_pi,
    mil_cpb_pi_da,
    mi_svm_pi,
    mi_svm_pi_da,
    s_mil_pi,
    s_mil_pi_da,
)
from mil_pi.data import load_mat_dataset
from mil_pi.kernels import KernelSpec, get_kernel, shrink_kernel, shrink_row


class MilPiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.visual = rng.normal(size=(4, 8))
        cls.visual[:, 4:] += 1.5
        cls.text = rng.normal(size=(3, 8))
        cls.test_visual = rng.normal(size=(4, 5))
        cls.K, cls.spec = get_kernel(cls.visual, spec=KernelSpec("gaussian", 0.5))
        cls.Kst = get_kernel(cls.visual, cls.test_visual, cls.spec)
        cls.tK, _ = get_kernel(cls.text, spec=KernelSpec("linear"))
        mask = np.r_[np.ones(4), np.zeros(4)]
        cls.tK *= np.outer(mask, mask)
        cls.test_kernel = cls.Kst.T
        cls.positive_bags = [2, 2]
        cls.negative_bags = [2, 2]
        cls.bags = [
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4, 5]),
            np.array([6, 7]),
        ]
        cls.bag_labels = np.array([1, 1, -1, -1])

    def test_shrink_helpers(self):
        small = shrink_kernel(self.K, [2, 2])
        rows = shrink_row(self.Kst, [2, 2])
        self.assertEqual(small.shape, (6, 6))
        self.assertEqual(rows.shape, (6, 5))
        self.assertTrue(np.allclose(small[:2, :2], small[:2, :2].T))

    def test_smil_pi(self):
        model = s_mil_pi(
            self.K,
            self.tK,
            self.positive_bags,
            self.negative_bags,
            {"svm_C": 1.0, "gamma": 1.0, "rho": 0.6, "solver_max_iter": 500},
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_smil_pi_da(self):
        model = s_mil_pi_da(
            self.K,
            self.tK,
            self.K,
            self.Kst,
            self.positive_bags,
            self.negative_bags,
            {
                "svm_C1": 1.0,
                "svm_C2": 0.1,
                "gamma1": 1.0,
                "gamma2": 1.0,
                "rho": 0.6,
                "B": 100.0,
                "epsilon": 0.0,
                "solver_max_iter": 1000,
            },
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_mi_svm_pi(self):
        model = mi_svm_pi(
            self.K,
            self.tK,
            self.positive_bags,
            self.negative_bags,
            {
                "svm_C": 0.5,
                "gamma": 1.0,
                "pos_ratio": 0.5,
                "max_iter": 2,
                "solver_max_iter": 1000,
            },
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_mi_svm_pi_da(self):
        model = mi_svm_pi_da(
            self.K,
            self.tK,
            self.K,
            self.Kst,
            self.positive_bags,
            self.negative_bags,
            {
                "svm_C1": 0.5,
                "svm_C2": 0.1,
                "gamma1": 1.0,
                "gamma2": 1.0,
                "B": 100.0,
                "epsilon": 0.0,
                "pos_ratio": 0.5,
                "max_iter": 2,
                "solver_max_iter": 1000,
            },
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_mil_cpb_pi(self):
        model = mil_cpb_pi(
            self.K,
            self.tK,
            self.bags,
            self.bag_labels,
            {
                "svm_C": 0.5,
                "svm_C_star": 1.0,
                "gamma": 1.0,
                "pos_ratio": 0.5,
                "max_iter": 2,
                "mkl_max_iter": 2,
                "solver_max_iter": 1000,
            },
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_mil_cpb_pi_da(self):
        model = mil_cpb_pi_da(
            self.K,
            self.tK,
            self.K,
            self.Kst,
            self.bags,
            self.bag_labels,
            {
                "svm_C1": 0.5,
                "svm_C2": 0.1,
                "gamma1": 1.0,
                "gamma2": 1.0,
                "B": 100.0,
                "epsilon": 0.0,
                "pos_ratio": 0.5,
                "max_iter": 2,
                "mkl_max_iter": 2,
                "solver_max_iter": 1000,
            },
        )
        self.assertEqual(model.coef.shape, (8,))
        self.assertEqual(model.decision_function(self.test_kernel).shape, (5,))

    def test_mat_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pos_visual = np.empty((2,), dtype=[("features", "O")])
            neg_visual = np.empty((2,), dtype=[("features", "O")])
            pos_text = np.empty((2,), dtype=[("features", "O")])
            neg_text = np.empty((2,), dtype=[("features", "O")])
            for index in range(2):
                pos_visual[index]["features"] = self.visual[:, 2 * index : 2 * index + 2]
                neg_visual[index]["features"] = self.visual[:, 4 + 2 * index : 6 + 2 * index]
                pos_text[index]["features"] = self.text[:, 2 * index : 2 * index + 2]
                neg_text[index]["features"] = self.text[:, 4 + 2 * index : 6 + 2 * index]
            savemat(root / "visual_bags.mat", {"pos_bags": pos_visual, "neg_bags": neg_visual})
            savemat(root / "text_bags.mat", {"pos_bags": pos_text, "neg_bags": neg_text})
            savemat(root / "test_features.mat", {"test_features": self.test_visual})
            savemat(root / "test_labels.mat", {"test_labels": np.array([1, -1, 1, -1, 1])})
            data = load_mat_dataset(root, max_bag_num=2, text_dim=2)
            self.assertEqual(data.bag_size, 2)
            self.assertEqual(data.bag_count, 2)
            self.assertEqual(data.test_features.shape, (4, 5))
            self.assertEqual(data.test_labels.shape, (5,))


if __name__ == "__main__":
    unittest.main()
