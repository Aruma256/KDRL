from unittest import TestCase
from kdrl.memory import Memory
import numpy as np

class TestMemory(TestCase):
    def test_discrete(self):
        capacity = 3
        state_shape = (3, 2)
        nb_actions = 4
        memory = Memory(capacity, state_shape, nb_actions)
        s = np.arange(state_shape[0] * state_shape[1]).reshape(state_shape)
        for i in range(5):
            memory.push(s + i, i % nb_actions, s + i + 1, i, i != 4)
        assert np.array_equal(memory.s[0], s + 3)
        assert np.array_equal(memory.s[1], s + 4)
        assert np.array_equal(memory.a, np.array([3, 0, 2]))
        assert np.array_equal(memory.s[0] + 1, memory.ns[0])
        assert np.array_equal(memory.ns[1], s + 2)
        assert np.array_equal(memory.s[2] + 1, memory.ns[2])
        assert np.array_equal(memory.r, np.array([3, 4, 2]))
        assert memory.c[0] and not memory.c[1] and memory.c[2]
        for i in range(capacity):
            for elem in memory.sample(i):
                assert len(elem) == i

