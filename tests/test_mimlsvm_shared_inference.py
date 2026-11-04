import unittest
import numpy as np
from experiments.mimlsvm_baseline import fit_mimlsvm
from experiments.mimlsvm_shared_inference import compile_shared_mimlsvm


class SharedMIMLSVMTest(unittest.TestCase):
    def test_scores_match_independent_libsvm_with_constant_words(self):
        rng = np.random.default_rng(72)
        bags = [rng.normal(size=(1+i%4, 5)) for i in range(18)]
        labels = np.column_stack([rng.integers(0,2,size=(18,4)),np.zeros(18),np.ones(18)])
        model = fit_mimlsvm(bags, labels, medoid_fraction=.3)
        compiled = compile_shared_mimlsvm(model)
        target = [rng.normal(size=(1+i%3,5)) for i in range(7)]
        np.testing.assert_allclose(compiled.decision_function(target), model.decision_function(target), rtol=1e-10, atol=1e-10)
        chunks = np.vstack([compiled.decision_function(target[:3]), compiled.decision_function(target[3:])])
        np.testing.assert_allclose(chunks, model.decision_function(target), rtol=1e-10, atol=1e-10)

    def test_all_constant_words(self):
        rng = np.random.default_rng(9)
        bags = [rng.normal(size=(2,3)) for _ in range(6)]
        model = fit_mimlsvm(bags, np.tile([0,1],(6,1)))
        np.testing.assert_array_equal(compile_shared_mimlsvm(model).decision_function(bags[:2]),[[-1,1],[-1,1]])

    def test_inconsistent_kernel_rejected(self):
        rng = np.random.default_rng(20)
        model = fit_mimlsvm([rng.normal(size=(2,3)) for _ in range(10)],rng.integers(0,2,size=(10,3)))
        model.classifiers[1]._gamma *= 2
        with self.assertRaises(ValueError):compile_shared_mimlsvm(model)
