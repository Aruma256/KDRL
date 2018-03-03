from kdrl.agents.nd import NDAgent
from kdrl.trainer import GymTrainer
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, InputLayer, Dense, Lambda, Concatenate

import gym

def get_actor_model(state_shape):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(64, activation='relu'),
                       Dense(64, activation='relu'),
                       Dense(1, activation='tanh'),
                       Lambda(lambda x: 2 * x)])

def get_critic_model(state_shape):
    si = Input(state_shape)
    ai = Input((1,))
    x = Concatenate()([si, ai])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)
    return Model([si, ai], x)

def main():
    env = gym.make('Pendulum-v0')
    #
    state_shape = env.observation_space.shape
    action_space = [env.action_space.low[0], env.action_space.high[0]]
    agent = NDAgent(action_space=action_space,
                    core_actor_model=get_actor_model(state_shape),
                    core_critic_model=get_critic_model(state_shape),
                    optimizer='adam',
                    policy=lambda x: np.random.normal(x, 1/16),
                    memory=30000,
                    target_model_update=5,
                    )
    trainer = GymTrainer(env, agent)
    # training
    result = trainer.train(200)
    # test
    result = trainer.test(5, render=True)
    for i, reward in enumerate(result['reward']):
        print('episode {}: {}'.format(i, reward))

if __name__ == '__main__':
    main()

