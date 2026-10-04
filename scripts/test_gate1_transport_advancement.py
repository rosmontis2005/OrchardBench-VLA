"""Focused guard checks for the TRANSPORT geometric advancement predicate."""
import unittest
import numpy as np
from check_gate1_oracle import transport_pass_progress


class TransportPassTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[i * .015, 0., 0.] for i in range(7)])
        self.phases = ['TRANSPORT'] * 7

    def passed(self, tcp, rotation=0., index=3):
        return transport_pass_progress(self.positions, self.phases, index, tcp, rotation)['passed']

    def test_ahead_near_path_outside_exact_reach(self):
        self.assertTrue(self.passed([.057, .005, 0.]))

    def test_behind_off_path_beyond_next_and_rotation_rejected(self):
        for tcp, rotation in [([.044, 0., 0.], 0.), ([.057, .011, 0.], 0.),
                              ([.061, 0., 0.], 0.), ([.057, 0., 0.], .081)]:
            with self.subTest(tcp=tcp, rotation=rotation):
                self.assertFalse(self.passed(tcp, rotation))

    def test_phase_boundaries_remain_strict(self):
        for i in [0, 1, 5, 6]:
            self.assertFalse(self.passed(self.positions[i] + [.001, 0., 0.], index=i))
        for phase in ['REACH', 'GRASP', 'PULL', 'DROP']:
            self.phases[4] = phase
            self.assertFalse(self.passed([.057, 0., 0.]))

    def test_stationary_corner_and_large_gap_rejected(self):
        self.positions[4] = self.positions[3]
        self.assertFalse(self.passed([.057, 0., 0.]))
        self.positions[4] = [.045, .015, 0.]
        self.assertFalse(self.passed([.045, .012, 0.]))
        self.positions[4] = [.075, 0., 0.]
        self.assertFalse(self.passed([.057, 0., 0.]))


if __name__ == '__main__':
    unittest.main()
