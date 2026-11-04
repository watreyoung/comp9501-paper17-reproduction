# Implementation and numerical checks

The Python conversion was checked against the retained MATLAB entry points and independent numerical references. The checks cover text-kernel scaling, positive-label updates, convexity of PI subproblems, QP residuals, AP tie handling, solver failures, non-finite inputs, MAT shape checks, and cache identity.

The tests distinguish a successful convex subproblem from convergence of the outer non-convex MIL/MKL loop. Iteration caps, failed candidates, and solver status remain visible in records and audits.
