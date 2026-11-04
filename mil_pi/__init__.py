"""Pure-Python implementation of the six MIL-PI demo programs in the archive."""

from .data import BagDataset, load_mat_dataset
from .kernels import (
    KernelSpec,
    average_precision,
    get_kernel,
    l2_normalize_columns,
    shrink_kernel,
    shrink_row,
)
from .algorithms import (
    mil_cpb_pi,
    mil_cpb_pi_da,
    mi_svm_pi,
    mi_svm_pi_da,
    s_mil_pi,
    s_mil_pi_da,
)

__all__ = [
    "BagDataset",
    "KernelSpec",
    "average_precision",
    "get_kernel",
    "l2_normalize_columns",
    "shrink_kernel",
    "shrink_row",
    "load_mat_dataset",
    "s_mil_pi",
    "s_mil_pi_da",
    "mi_svm_pi",
    "mi_svm_pi_da",
    "mil_cpb_pi",
    "mil_cpb_pi_da",
]
