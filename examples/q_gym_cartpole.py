from kdrl.agents import DescritizingQAgent
from kdrl.policy import Boltzmann
from kdrl.trainer import GymTrainer
import numpy as np

import gym

def main():
    env = gym.make('CartPole-v0')
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DescritizingQAgent(action_space=num_actions,
                               boundaries=[np.array([-0.3, 0, 0.3]),
                                           np.array([-0.2, 0, 0.2]),
                                           np.array([-0.2, 0, 0.2]),
                                           np.array([-0.2, 0, 0.2])],
                               policy=Boltzmann(),
                               )
    trainer = GymTrainer(env, agent)
    # training
    result = trainer.train(2000, render=False)
    # test
    result = trainer.test(5, render=True)
    for i, steps in enumerate(result['steps']):
        print('episode {}: {} steps'.format(i, steps))

if __name__ == '__main__':
    main()

