"""Explicit TCA2011 Eq6 parameter convention, no change to frozen runs.

Paper: KLK + mu_paper I. Existing solver: I + mu_internal KLK.
Their constrained minimizers coincide for mu_internal=1/mu_paper;
objective/eigenvalues must additionally be multiplied by mu_paper.
"""
import numpy as np
from .feature_adaptation import fit_tca,TransferComponents


def fit_tca_journal(kernel,n_source,dimension,*,mu_paper=10.):
    if not np.isfinite(mu_paper) or mu_paper<=0:raise ValueError('Positive finite journal regularizer required')
    model=fit_tca(kernel,n_source,dimension,mu=1/mu_paper)
    return TransferComponents(model.coefficients,mu_paper*model.eigenvalues,model.constraint_error,model.stationarity_error)
