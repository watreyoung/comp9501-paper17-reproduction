"""DIP Gaussian MMD with unit-direction strong-Wolfe Grassmann PR+ CG.

Separate performance version: searches the same geodesic along H/||H||.
Frozen Armijo and raw-direction Wolfe experiments remain unchanged. The
paper specifies geodesic line search, not its Wolfe constants or tolerance.
"""
import warnings
import numpy as np
from scipy.optimize import line_search
from scipy.spatial.distance import pdist
from .dip_baseline import gaussian_mmd_value_gradient, grassmann_step, DomainInvariantProjection


def fit_dip_wolfe_normalized(source, target, dimension, *, max_iter=1000,
                  gradient_tolerance=1e-6, relative_tolerance=1e-9):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError('Shared feature dimension required')
    if min(len(source), len(target)) < 2:
        raise ValueError('Both domains require at least two samples')
    sigma = float(np.median(pdist(source, 'sqeuclidean')))
    if not sigma > 0:
        raise ValueError('Nonzero source bandwidth required')
    X = np.r_[source, target]
    mean = X.mean(axis=0)
    X = X-mean
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = int(np.count_nonzero(s > max(X.shape)*np.finfo(float).eps*s[0]))
    if not 1 <= dimension <= rank:
        raise ValueError('Dimension exceeds joint PCA rank')
    basis = Vt[:rank].T
    X = X @ basis
    W = np.eye(rank, dimension)
    value, ambient = gaussian_mmd_value_gradient(X, W, len(source), sigma)
    gradient = ambient-W@(W.T@ambient)
    H = -gradient
    history, reason = [], 'max_iter'
    previous_angle = 1.
    for iteration in range(max_iter):
        norm = float(np.linalg.norm(gradient))
        if norm <= gradient_tolerance:
            reason = 'gradient_tolerance'
            break
        # Sufficient-descent restart avoids nearly orthogonal PR directions.
        restart = float(np.sum(gradient*H)) >= -.01*norm**2
        if restart:
            H = -gradient
        direction_norm = float(np.linalg.norm(H))
        search_direction = H/direction_norm
        evaluated = {}

        def evaluate(a):
            step = float(np.asarray(a).ravel()[0])
            if step not in evaluated:
                candidate, transport = grassmann_step(W, search_direction, step)
                val, amb = gaussian_mmd_value_gradient(X, candidate, len(source), sigma)
                grad = amb-candidate@(candidate.T@amb)
                derivative = float(np.sum(grad*transport(search_direction)))
                evaluated[step] = val, derivative, candidate, transport, grad
            return evaluated[step]

        derivative = float(np.sum(gradient*search_direction))
        initial_angle = min(1., 2*previous_angle)
        step = None
        for rescale in range(5):
            trial_angle = initial_angle*(.1**rescale)
            with warnings.catch_warnings(record=True):
                alpha, *_ = line_search(lambda a: evaluate(a)[0],
                                        lambda a: np.array([evaluate(a)[1]]),
                                        np.array([0.]), np.array([trial_angle]),
                                        gfk=np.array([derivative]), old_fval=value,
                                        c1=1e-4, c2=.1, maxiter=30,
                                        amax=np.pi/trial_angle)
            if alpha is not None:
                step = float(alpha*trial_angle)
                break
        if step is None:
            reason = 'line_search_failed'
            break
        newvalue, newderivative, candidate, transport, newgradient = evaluate(step)
        if newvalue > value+1e-4*step*derivative or abs(newderivative) > .1*abs(derivative)+1e-14:
            raise ArithmeticError('Strong Wolfe conditions not satisfied')
        transported_gradient = transport(gradient)
        beta = max(0., float(np.sum(newgradient*(newgradient-transported_gradient)))/norm**2)
        newH = -newgradient+beta*transport(H)
        change = value-newvalue
        history.append(dict(iteration=iteration+1, objective=newvalue,
                            previous_objective=value, gradient_norm=float(np.linalg.norm(newgradient)),
                            step=float(step), raw_step=float(step/direction_norm), direction_norm=direction_norm, beta=beta, restart=restart,
                            initial_derivative=derivative, final_derivative=newderivative,
                            evaluations=len(evaluated), line_search_rescales=rescale,
                            initial_angle=initial_angle))
        previous_angle = step
        W, value, gradient, H = candidate, newvalue, newgradient, newH
        if change <= relative_tolerance*max(1., abs(value)):
            reason = 'objective_tolerance'
            break
    error = float(np.max(np.abs(W.T@W-np.eye(dimension))))
    if error > 1e-8:
        raise ArithmeticError('DIP orthogonality lost')
    return DomainInvariantProjection(mean, basis, W, sigma, history, reason, error)
