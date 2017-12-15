from unittest import TestCase
from kdrl.memory import *
import numpy as np

class TestSingleActionMemory(TestCase):
    def test_discrete(self):
        capacity = 3
        state_shape = (3, 2)
        num_actions = 4
        memory = SingleActionMemory(capacity, state_shape)
        s = np.arange(state_shape[0] * state_shape[1]).reshape(state_shape) + 10
        memory.start_episode(s, 0)
        memory.step(s + 1, 1, 101)      # 0: 0->1
        memory.step(s + 2, 2, 102)      # 1: 1->2
        memory.step(s + 3, 3, 103)      # 2: 2->3
        memory.end_episode(s + 4, 104)  # 0: 3->4
        assert np.array_equal(memory.s[0], s + 3)
        assert np.array_equal(memory.s[1], s + 1)
        assert np.array_equal(memory.a, np.array([3, 1, 2]))
        assert np.array_equal(memory.s[0] + 1, memory.ns[0])
        assert np.array_equal(memory.ns[1], s + 2)
        assert np.array_equal(memory.s[2] + 1, memory.ns[2])
        assert np.array_equal(memory.r, np.array([104, 102, 103]))
        assert not memory.c[0]
        assert memory.c[1]
        assert memory.c[2]
        for i in range(capacity):
            for elem in memory.sample(i):
                assert len(elem) == i

