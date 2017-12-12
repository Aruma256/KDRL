import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, dot

def as_batch(state):
    return np.expand_dims(state, axis=-1)

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
        action_switch = Input(shape=(1,), dtype='uint8')
        one_hot = Lambda(lambda x: K.one_hot(x, num_classes=num_actions), output_shape=(num_actions,))
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
            self.target_model = self.model
        else:
            self.target_model = model_from_json(model.to_json())
        self.last_state = None
        self.episode = 1
        self.compiled = False
        self.learning = True
    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.sync_target_model()
        self.compiled = True
    def sync_target_model(self):
        if target_model_update != 1:
            self.target_model.set_weights(self.model.get_weights())
    def select_best_action(self, state):
        scores = self.core_model.predict_on_batch(as_batch(state))[0]
        return np.argmax(scores)
    def train(self):
        states, actions, next_states, rewards, step_flags = self.memory.getSample(self.batch_size)
        pred_Q = self.target_model.predict_on_batch()

