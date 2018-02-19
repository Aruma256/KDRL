class AbstractAgent:
    def __init__(self):
        pass

    def start_episode(self, state):
        raise NotImplementedError()

    def step(self, state, reward):
        raise NotImplementedError()

    def end_episode(self, state, reward):
        raise NotImplementedError()

    def select_best_action(self, state):
        raise NotImplementedError()
