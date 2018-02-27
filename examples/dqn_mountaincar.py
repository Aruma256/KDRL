from kdrl.agents.dqn import DQNAgent
from kdrl.policy import Greedy
from kdrl.trainer import GymTrainer
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Lambda

import gym

def get_model(state_shape, num_actions):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(64, activation='relu'),
                       Dense(num_actions)])

def main():
    env = gym.make('MountainCar-v0')
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(action_space=num_actions,
                     core_model=get_model(state_shape, num_actions),
                     optimizer='adam',
                     policy=Greedy(),
                     memory=30000,
                     )
    trainer = GymTrainer(env, agent)
    # training
    result = trainer.train(1000)
    # test
    result = trainer.test(5, render=True)
    for i, reward in enumerate(result['reward']):
        print('episode {}: {}'.format(i, reward))

if __name__ == '__main__':
    main()

