from unittest import TestCase
from kdrl.policy import *
import numpy as np

class TestPolicy(TestCase):
    def test_greedy(self):
        policy = Greedy()
        assert policy(np.array([5, 20, 10])) == 1
        assert policy(np.array([-5, -20, -10])) == 0
    def test_epsilongreedy(self):
        policy = EpsilonGreedy(0)
        policy.eps = 0
        assert policy(np.array([5, 20, 10])) == 1
        assert policy(np.array([-5, -20, -10])) == 0
        policy.eps = 1
        assert 0 <= policy(np.array([5, 20, 10])) < 3
    def test_boltzmann(self):
        policy = Boltzmann()
        assert 0 <= policy(np.array([5, 20, 10])) < 3

