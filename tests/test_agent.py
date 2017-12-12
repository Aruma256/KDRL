from unittest import TestCase
from kdrl.agent import DQNAgent
from kdrl.policy import *
from kdrl.memory import *
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense

class TestAgent(TestCase):
    def test(self):
        dim_state = 3
        num_actions = 5
        input = Input((dim_state,))
        model = Model(input, Dense(num_actions)(input))
        agent = DQNAgent(core_model=model,
                         num_actions=num_actions,
                         optimizer='sgd',
                         policy=EpsilonGreedyPolicy(eps=0.01),
                         memory=SingleActionMemory(capacity=1000,
                                                   state_shape=(dim_state,)),
                         )



