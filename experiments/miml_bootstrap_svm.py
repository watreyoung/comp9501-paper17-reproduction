"""Exact bootstrap aggregation + on-demand conditional SVM training bridge."""
import numpy as np
from .miml_conditional_kernel import compressed_bootstrap
from .column_svm import fit_column_svm


def fit_conditional_bootstrap(kernel,bag_labels,draws,*,C=1.,tolerance=1e-6,max_iter=100000,backend="numpy"):
    """Return solver result and global conditional IDs in its coefficient order.

    Labels are binary bag-by-word values. Original bootstrap duplicates become
    integer loss multipliers. A nonconverged result stays marked max_iter;
    callers must not silently store it as a valid MIMLBoost weak learner.
    """
    labels=np.asarray(bag_labels,float)
    if labels.shape!=(len(kernel.lengths),kernel.n_labels) or not np.isin(labels,[0,1]).all() or not np.isfinite(C) or C<=0:raise ValueError('Matching binary bag-word labels and positive cost required')
    ids,multiplicity=compressed_bootstrap(draws,kernel.total_instances)
    visual,word,group=kernel.decode(ids);y=2*labels.ravel()[group]-1
    if len(np.unique(y))!=2:raise ValueError('Single-class bootstrap: caller must handle explicitly')
    # Decode once, not once per solver iteration. No augmented visual matrix.
    def column(i):
        return kernel.visual_kernel[visual,visual[i]]*kernel.label_kernel[word,word[i]]
    model=fit_column_svm(y,C*multiplicity,column,diagonal=np.ones(len(ids)),tolerance=tolerance,max_iter=max_iter,backend=backend)
    return model,ids
