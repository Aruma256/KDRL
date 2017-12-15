from unittest import TestCase
from kdrl.agent import DQNAgent
from kdrl.policy import *
from kdrl.memory import *
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense

import gym

class TestAgent(TestCase):
    def test_dqn_cartpole(self):
        env = gym.make('CartPole-v0')
        #
        state_shape = env.observation_space.shape
        num_actions = env.action_space.n
        model = Sequential([InputLayer(input_shape=state_shape),
                            Dense(16, activation='relu'),
                            Dense(16, activation='relu'),
                            Dense(16, activation='relu'),
                            Dense(num_actions)])
        agent = DQNAgent(core_model=model,
                         num_actions=num_actions,
                         optimizer='adam',
                         policy=BoltzmannPolicy(),
                         memory=SingleActionMemory(capacity=10000,
                                                   state_shape=state_shape),
                         )
        #
        for episode in range(1000):
            state = env.reset()
            action = agent.start_episode(state)
            while True:
                env.render()
                state, reward, done, info = env.step(action)
                if not done:
                    action = agent.step(state, reward)
                    continue
                else:
                    agent.end_episode(state, reward)
                    break

