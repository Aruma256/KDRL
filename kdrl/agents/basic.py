import numpy as np
from .core import AbstractAgent


class _StaticAgentD(AbstractAgent):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def start_episode(self, state):
        return self.select_best_action(state)

    def step(self, state, reward):
        return self.select_best_action(state)

    def end_episode(self, state, reward):
        pass


class RandomAgentD(_StaticAgentD):
    def __init__(self, num_actions):
        super().__init__(num_actions)

    def select_best_action(self, state):
        return np.random.choice(self.num_actions)


class ConstantAgentD(_StaticAgentD):
    def __init__(self, num_actions, action):
        super().__init__(num_actions)
        self.action = action

    def select_best_action(self, state):
        return self.action
