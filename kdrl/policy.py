import numpy as np

class GreedyPolicy:
    def __call__(self, scores):
        return np.argmax(scores)

class EpsilonGreedyPolicy(GreedyPolicy):
    def __init__(self, eps):
        self.eps = eps
    def __call__(self, scores):
        if self.eps < np.random.uniform():
            return np.random.randint(len(scores))
        else:
            return GreedyPolicy.__call__(self, scores)

class BoltzmannPolicy:
    def __init__(self, t=1):
        self.t = t
    def __call__(self, scores):
        q = np.asarray(scores, dtype=np.float32)
        q -= np.max(q)
        eq = np.exp(q / self.t)
        probs = eq / np.sum(eq)
        action = np.random.choice(len(scores), p=probs)
        return action

