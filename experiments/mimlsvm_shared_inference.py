"""Exact shared-RBF inference for binary MIMLSVM word classifiers.

All classifiers were fitted on the same medoid-distance embedding. Their
support indices identify shared source-bag rows. No retraining, approximation
or changed vocabulary is involved; constants retain their original scores.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from .hausdorff_blocks import blocked_hausdorff


@dataclass
class SharedMIMLSVM:
    medoid_bags: list
    support_embedding: np.ndarray
    coefficients: np.ndarray
    biases: np.ndarray
    gamma: float

    def decision_function(self, bags):
        Z = blocked_hausdorff(bags, self.medoid_bags)
        if not len(self.support_embedding):
            return np.tile(self.biases, (len(Z), 1))
        K = np.exp(-self.gamma*cdist(Z, self.support_embedding, 'sqeuclidean'))
        return K@self.coefficients+self.biases


def compile_shared_mimlsvm(model):
    classifiers = model.classifiers
    active = [c for c in classifiers if c is not None]
    dimensions = len(model.medoid_bags)
    biases = np.zeros(len(classifiers))
    if not active:
        for j in range(len(classifiers)):
            biases[j] = model.constants[j]
        return SharedMIMLSVM(model.medoid_bags, np.empty((0, dimensions)), np.empty((0, len(classifiers))), biases, 1.)
    gamma = float(active[0]._gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Positive shared RBF gamma required')
    n = 1+max(int(c.support_.max()) for c in active)
    reference = np.zeros((n, dimensions))
    present = np.zeros(n, bool)
    coefficients = np.zeros((n, len(classifiers)))
    for j, clf in enumerate(classifiers):
        if clf is None:
            biases[j] = model.constants[j]
            continue
        if clf.kernel != 'rbf' or float(clf._gamma) != gamma:
            raise ValueError('All classifiers must share the same RBF kernel')
        if not np.array_equal(clf.classes_, [-1., 1.]) or clf.fit_status_ != 0:
            raise ValueError('Converged binary {-1,+1} classifiers required')
        indices = np.asarray(clf.support_, int)
        vectors = np.asarray(clf.support_vectors_)
        if vectors.shape != (len(indices), dimensions) or len(np.unique(indices)) != len(indices):
            raise ValueError('Support embedding/index mismatch')
        overlap = present[indices]
        if not np.array_equal(reference[indices[overlap]], vectors[overlap]):
            raise ValueError('Word classifiers do not share source embedding row identities')
        reference[indices] = vectors
        present[indices] = True
        coefficients[indices, j] = clf.dual_coef_.ravel()
        biases[j] = clf.intercept_.item()
    return SharedMIMLSVM(model.medoid_bags, reference[present], coefficients[present], biases, gamma)
