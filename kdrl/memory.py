import numpy as np

class SingleActionMemory:
    def __init__(self, capacity, state_shape, continuous_action=False):
        self.capacity = capacity
        self._next = 0
        self.filled = (capacity == self._next)
        #
        self.s = np.zeros(((capacity,) + state_shape), dtype=np.float32)
        if continuous_action:
            self.a = np.zeros(capacity, dtype=np.float32)
        else:
            self.a = np.zeros(capacity, dtype=np.int8)
        self.ns = np.zeros_like(self.s)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.c = np.zeros(capacity, dtype=np.bool)

    def push(self, state, action, next_state, reward, cont):
        p = self._next
        self.s[p] = state
        self.a[p] = action
        if cont:
            self.ns[p] = next_state
        self.r[p] = reward
        self.c[p] = cont
        #
        self._next += 1
        self._next %= self.capacity
        if self._next == 0:
            self.filled = True

    def _get_current_size(self):
        return self.capacity if self.filled else self._next

    def sample(self, num):
        assert self._get_current_size() >= num
        idx = np.random.choice(self._get_current_size(), size=num, replace=False)
        return self.s[idx], self.a[idx], self.ns[idx], self.r[idx], self.c[idx]
