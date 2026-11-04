"""Verified four-adapter dispatch with an explicit strong-Wolfe DIP solver."""
import numpy as np
from .full_adaptation import HEADS, classify_adaptation, select_validation_family
from .full_adaptation import feature_adaptation as original_adaptation
from .baseline_extension import adapted_problem
from .dip_wolfe import fit_dip_wolfe
from mil_pi.kernels import l2_normalize_columns


def feature_adaptation(name, data, problem, *, dimension=10, tca_mu=.1):
    if name != 'DIP':
        return original_adaptation(name, data, problem, dimension=dimension, tca_mu=tca_mu)
    X = l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags, axis=1)).T
    target = l2_normalize_columns(data.test_features).T
    model = fit_dip_wolfe(X, target, dimension, max_iter=1000)
    source_new, target_new = model.transform(X), model.transform(target)
    if not np.isfinite(source_new).all() or not np.isfinite(target_new).all():
        raise ArithmeticError('Nonfinite DIP transformation')
    adapted, gamma = adapted_problem(problem, source_new, target_new)
    diag = dict(method=name, dimension=dimension, uses_target_features=True,
                uses_target_labels=False, stop_reason=model.stop_reason,
                history=model.history, orthogonality_error=model.orthogonality_error,
                sigma=model.sigma, kernel_gamma=gamma, solver='Grassmann PR+ strong Wolfe',
                max_iter=1000, gradient_tolerance=1e-6, relative_tolerance=1e-9,
                wolfe_c1=1e-4, wolfe_c2=.1)
    arrays = dict(joint_mean=model.joint_mean, pca_basis=model.pca_basis, projection=model.projection)
    return adapted, diag, arrays
