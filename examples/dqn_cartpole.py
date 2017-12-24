from kdrl.agent import DQNAgent
from kdrl.policy import *
from kdrl.memory import *
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
    # training
    for episode in range(500):
        state = env.reset()
        action = agent.start_episode(state)
        while True:
            state, reward, done, info = env.step(action)
            if not done:
                action = agent.step(state, reward)
                continue
            else:
                agent.end_episode(state, reward)
                break
    # test
    for episode in range(5):
        state = env.reset()
        reward_sum = 0
        while True:
            env.render()
            action = agent.select_best_action(state)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
        print('episode {} score: {}'.format(episode, reward_sum))

if __name__ == '__main__':
    main()

