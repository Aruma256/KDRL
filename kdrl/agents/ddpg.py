import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, dot

from .core import AbstractAgent
from ..memory import SingleActionMemory

class DDPGAgent(AbstractAgent):
    def __init__(self,
                 action_space,
                 core_actor_model,
                 core_critic_model,
                 optimizer,
                 policy,
                 memory,
                 *args,
                 loss='mean_squared_error',
                 gamma=0.99,
                 target_model_update=1,
                 warmup=100,
                 batch_size=32,
                 **kwargs):
        super().__init__(action_space, *args, **kwargs)
        self.core_actor_model = core_actor_model
        self.core_critic_model = core_critic_model
        #
        self.policy = policy
        if isinstance(memory, int):
            self.memory = SingleActionMemory(int(memory), core_actor_model.inputs[0]._keras_shape[1:], continuous_action=True)
        else:
            self.memory = memory
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.warmup = warmup
        self.batch_size = batch_size
        #
        # compile
        i = Input(core_actor_model.inputs[0]._keras_shape[1:])
        self.combined_model = Model(i, core_critic_model([i, core_actor_model(i)]))
        if target_model_update == 1:
            self.target_core_critic_model = self.core_critic_model
            self.target_combined_model = self.combined_model
        else:
            self.target_core_critic_model = model_from_json(self.core_critic_model.to_json())
            self.target_combined_model = model_from_json(self.combined_model.to_json())
        self.episode_count = 0
        self.train_count = 0
        self.train_history = []
        self._fake_y_true = np.zeros((self.batch_size, 1))
        #
        core_critic_model.trainable = False
        self.combined_model.compile(optimizer, loss=lambda y_true, y_pred: -y_pred)
        core_critic_model.trainable = True
        self.core_critic_model.compile(optimizer, loss=loss)
        self._sync_target_model()
    # primary method
    def start_episode(self, state):
        self.memory.start_episode(state)
        action = self._select_action(state)
        self.memory.set_action(action)
        return action
    def step(self, state, reward):
        self.memory.step(state, reward)
        self._train()
        action = self._select_action(state)
        self.memory.set_action(action)
        return action
    def end_episode(self, state, reward):
        self.memory.end_episode(state, reward)
        self._train()
        self.episode_count += 1
    # select action
    def _select_action(self, state):
        action = self.core_actor_model.predict_on_batch(np.asarray([state]))[0]
        action = self.policy(action)
        action = np.clip(action, *self.action_space)
        return self.select_best_action(state)
    def select_best_action(self, state):
        action = self.core_actor_model.predict_on_batch(np.asarray([state]))[0]
        action = np.clip(action, *self.action_space)
        return action
    # training
    def _train(self):
        if self.warmup < self.memory._get_current_size():
            x, y = self._gen_training_data()
            history = self.core_critic_model.train_on_batch(x, y)
            self.train_history.append(history)
            #
            self.combined_model.train_on_batch(x[0], self._fake_y_true)
            #
            self.train_count += 1
            if self.target_model_update > 1 and self.train_count % self.target_model_update == 0:
                self._sync_target_model()
    def _gen_training_data(self):
        states, actions, next_states, rewards, cont_flags = self.memory.sample(self.batch_size)
        next_V = self.target_combined_model.predict_on_batch(next_states).reshape(-1)
        inputs = [states, actions]
        targets = rewards + cont_flags * self.gamma * next_V
        return inputs, targets
    def _sync_target_model(self):
        if self.target_model_update != 1:
            self.target_combined_model.set_weights(self.combined_model.get_weights())

