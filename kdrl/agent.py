import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, dot

class DQNAgent:
    def __init__(self,
                 core_model,
                 num_actions,
                 optimizer,
                 policy,
                 memory,
                 loss='mean_squared_error',
                 gamma=0.99,
                 target_model_update=1,
                 warmup=100,
                 batch_size=32):
        self.core_model = core_model
        state_input = self.core_model.input
        action_switch = Input(shape=(1,), dtype='int32')
        if K.backend() == 'tensorflow':
            one_hot = Lambda(lambda x: K.flatten(K.one_hot(x, num_actions)), output_shape=(num_actions,))
        else:
            one_hot = Lambda(lambda x: K.one_hot(x, num_actions), output_shape=(num_actions,))
        self.model = Model([state_input, action_switch], dot([self.core_model(state_input), one_hot(action_switch)], axes=1))
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.loss = loss
        self.policy = policy
        self.memory = memory
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.warmup = warmup
        self.batch_size = batch_size
        #
        if target_model_update == 1:
            self.target_core_model = self.core_model
        else:
            self.target_core_model = model_from_json(self.core_model.to_json())
        self.last_state = None
        self.episode = 1
        self.learning = True
        # compile
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.sync_target_model()
    def sync_target_model(self):
        if self.target_model_update != 1:
            self.target_core_model.set_weights(self.core_model.get_weights())
    def select_best_action(self, state):
        scores = self.core_model.predict_on_batch(state)[0]
        return np.argmax(scores)
    def train(self):
        states, actions, next_states, rewards, cont_flags = self.memory.sample(self.batch_size)
        pred_Q = self.target_core_model.predict_on_batch(next_states)
        max_Q = np.max(pred_Q, axis=-1)
        self.model.train_on_batch([states, actions], rewards + cont_flags * self.gamma * max_Q)
    def _select_action(self, state):
        scores = self.core_model.predict_on_batch(np.asarray(state))
        action = self.policy(scores)
        return action
    def start_episode(self, state):
        action = self._select_action(state)
        self.memory.start(state, action)
        return action
    def step(self, state, reward):
        action = self._select_action(state)
        self.memory.step(state, action, reward)
        return action
    def end_episode(self, state, reward):
        self.memory.end_episode(state, reward)

