import unittest
import numpy as np
from experiments.dip_wolfe import fit_dip_wolfe


class WolfeDIPTest(unittest.TestCase):
    def test_geodesic_line_search_meets_independent_wolfe_bounds(self):
        rng = np.random.default_rng(19)
        s = rng.normal(size=(24, 6))
        t = rng.normal(size=(20, 6))+[2, 0, 0, 0, 0, 0]
        m = fit_dip_wolfe(s, t, 2)
        self.assertIn(m.stop_reason, ['gradient_tolerance', 'objective_tolerance'])
        self.assertTrue(m.history)
        for h in m.history:
            self.assertLessEqual(h['objective'], h['previous_objective']+1e-4*h['step']*h['initial_derivative'])
            self.assertLessEqual(abs(h['final_derivative']), .1*abs(h['initial_derivative'])+1e-14)
        self.assertLess(m.orthogonality_error, 1e-10)
        self.assertTrue(np.isfinite(m.transform(t)).all())
