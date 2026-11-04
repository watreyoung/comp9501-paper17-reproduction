import unittest
import numpy as np
from experiments.geodesic_baselines import subspace_disagreement,select_sdm_dimension


class SDMTest(unittest.TestCase):
    def test_known_angles_and_first_orthogonal_dimension(self):
        I=np.eye(5);source=I[:,[0,3]];target=I[:,[0,4]];joint=I[:,[0,1]]
        np.testing.assert_allclose(subspace_disagreement(source,target,joint),[0,1],atol=1e-14)
        self.assertEqual(select_sdm_dimension([0,1]),2)
        theta=.3;S=np.array([[np.cos(theta)],[np.sin(theta)],[0.]])
        self.assertAlmostEqual(subspace_disagreement(S,S,np.eye(3)[:,:1])[0],np.sin(theta))

    def test_no_threshold_crossing_not_silently_maximized(self):
        self.assertIsNone(select_sdm_dimension([.1,.8,.2]))
        self.assertEqual(select_sdm_dimension([.1,1-1e-10,.3]),2)
        self.assertIsNone(select_sdm_dimension([.1,1-1e-10,.3],orthogonality_tolerance=0))


if __name__=='__main__':unittest.main()
