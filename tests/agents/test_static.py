from unittest import TestCase
from kdrl.agents.static import *
from kdrl.trainer import *
import numpy as np


class TestStaticAgent(TestCase):
    def test_random_agent(self):
        agent = RandomAgent(action_space=2)

        def random_test(f):
            first_action = f()
            for i in range(100):
                action = f()
                if first_action != action:
                    return True
            return False
        self.assertTrue(random_test(agent.select_best_action))
        self.assertTrue(random_test(agent.start_episode))
        self.assertTrue(random_test(agent.step))

    def test_constant_agent(self):
        agent = ConstantAgent(action_space=100, constant_action=0)

        def const_test(value):
            agent.constant_action = value
            self.assertEqual(value, agent.start_episode())
            self.assertEqual(value, agent.step())
            self.assertEqual(value, agent.select_best_action())
        const_test(0)
        const_test(2)
        const_test([1, 2])
