class AbstractAgent:
    def __init__(self, *args, suppress_warnings=False, **kwargs):
        if not suppress_warnings and args:
            print('ignored args:', args)
        if not suppress_warnings and kwargs:
            print('ignored kwargs:', kwargs)

    def start_episode(self, state):
        raise NotImplementedError()

    def step(self, state, reward):
        raise NotImplementedError()

    def end_episode(self, state, reward):
        raise NotImplementedError()

    def select_best_action(self, state):
        raise NotImplementedError()
