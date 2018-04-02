from unittest import TestCase
from kdrl.policy import *
import numpy as np

_POLICY_TEST = 100

def _count_action(policy):
    result = {0: 0, 1: 0}
    for i in range(_POLICY_TEST):
        action = policy([0, 1])
        result[action] += 1
    return result[0], result[1]

class TestPolicy(TestCase):
    def test_random(self):
        policy = Random()
        action_zero, action_one = _count_action(policy)
        self.assertGreater(action_one, 0)
        self.assertGreater(action_zero, 0)
    def test_greedy(self):
        policy = Greedy()
        action_zero, action_one = _count_action(policy)
        self.assertEqual(action_zero, 0)
        self.assertEqual(action_one, _POLICY_TEST)
    def test_epsilongreedy(self):
        policy = EpsilonGreedy(0)
        action_zero, action_one = _count_action(policy)
        self.assertEqual(action_zero, 0)
        self.assertEqual(action_one, _POLICY_TEST)
        policy = EpsilonGreedy(0.5)
        action_zero, action_one = _count_action(policy)
        self.assertGreater(action_one, action_zero)
        self.assertGreater(action_one, 0)
        self.assertGreater(action_zero, 0)
    def test_boltzmann(self):
        policy = Boltzmann()
        action_zero, action_one = _count_action(policy)
        self.assertGreater(action_one, action_zero)
        self.assertGreater(action_one, 0)
        self.assertGreater(action_zero, 0)

