import numpy as np
from .core import AbstractAgent, AbstractDiscreteAgent


class _StaticAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_episode(self, state):
        return self.select_best_action(state)

    def step(self, state, reward):
        return self.select_best_action(state)

    def end_episode(self, state, reward):
        pass


class _StaticAgentD(AbstractDiscreteAgent):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(num_actions, *args, **kwargs)

    def start_episode(self, state):
        return self.select_best_action(state)

    def step(self, state, reward):
        return self.select_best_action(state)

    def end_episode(self, state, reward):
        pass


class RandomAgentD(_StaticAgentD):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(num_actions, *args, **kwargs)

    def select_best_action(self, state):
        return np.random.choice(self.num_actions)


class ConstantAgent(_StaticAgent):
    def __init__(self, action, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = action

    def select_best_action(self, state):
        return self.action
