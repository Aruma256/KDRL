from unittest import TestCase
from kdrl.agent import DQNAgent
from kdrl.policy import *
from kdrl.memory import *
from kdrl.trainer import *
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense

import gym


def get_model(state_shape, num_actions):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(64, activation='relu'),
                       Dense(num_actions)])


class TestAgent(TestCase):
    def test_dqn_init(self):
        env = gym.make('CartPole-v0')
        state_shape = env.observation_space.shape
        num_actions = env.action_space.n
        agent = DQNAgent(action_space=num_actions,
                         core_model=get_model(state_shape, num_actions),
                         optimizer='adam',
                         policy=EpsilonGreedy(eps=0.01),
                         memory=SingleActionMemory(capacity=10000,
                                                   state_shape=state_shape),
                         )
        agent = DQNAgent(action_space=num_actions,
                         core_model=get_model(state_shape, num_actions),
                         optimizer='adam',
                         policy=EpsilonGreedy(eps=0.01),
                         memory=10000,
                         )

    def test_dqn_cartpole(self):
        env = gym.make('CartPole-v0')
        state_shape = env.observation_space.shape
        num_actions = env.action_space.n
        agent = DQNAgent(action_space=num_actions,
                         core_model=get_model(state_shape, num_actions),
                         optimizer='adam',
                         policy=Boltzmann(),
                         memory=10000,
                         )
        #
        trainer = GymTrainer(env, agent)
        trainer.train(200, False)
        result = trainer.test(10, False)['steps']
        self.assertEqual(max(result), 200, result)
