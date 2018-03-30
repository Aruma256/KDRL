from unittest import TestCase
from kdrl.policy import *
import numpy as np

class TestPolicy(TestCase):
    def test_greedy(self):
        policy = Greedy()
        self.assertEqual(policy(np.array([5, 20, 10])), 1)
        self.assertEqual(policy(np.array([-5, -20, -10])), 0)
    def test_epsilongreedy(self):
        policy = EpsilonGreedy(0)
        policy.eps = 0
        self.assertEqual(policy(np.array([5, 20, 10])), 1)
        self.assertEqual(policy(np.array([-5, -20, -10])), 0)
        policy.eps = 1
        self.assertTrue(0 <= policy(np.array([5, 20, 10])) < 3)
    def test_boltzmann(self):
        policy = Boltzmann()
        self.assertTrue(0 <= policy(np.array([5, 20, 10])) < 3)

