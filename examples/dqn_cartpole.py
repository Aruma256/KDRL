from kdrl.agent import DQNAgent
from kdrl.policy import EpsilonGreedy
from kdrl.trainer import GymTrainer
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense

import gym

def get_model(state_shape, num_actions):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(64, activation='relu'),
                       Dense(64, activation='relu'),
                       Dense(num_actions)])

def main():
    env = gym.make('CartPole-v0')
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(core_model=get_model(state_shape, num_actions),
                     num_actions=num_actions,
                     optimizer='adam',
                     policy=EpsilonGreedy(eps=0.01),
                     memory=30000,
                     )
    trainer = GymTrainer(env, agent)
    # training
    trainer.train(500)
    # test
    trainer.test(5, render=True)

if __name__ == '__main__':
    main()

