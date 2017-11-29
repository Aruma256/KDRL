from unittest import TestCase
from kdrl.policy import GreedyPolicy, EpsilonGreedyPolicy
import numpy as np

class TestMemory(TestCase):
    def test_greedy(self):
        policy = GreedyPolicy()
        assert policy(np.array([5, 20, 10])) == 1
        assert policy(np.array([-5, -20, -10])) == 0

    def test_epsilongreedy(self):
        policy = EpsilonGreedyPolicy(0)
        policy.eps = 0
        assert policy(np.array([5, 20, 10])) == 1
        assert policy(np.array([-5, -20, -10])) == 0
        

