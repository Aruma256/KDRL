import numpy as np

from keras.models import Model, model_from_json
from keras.layers import Input, Multiply

from keras_util import build

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
        state_input = self.core_model.input()
        action_switch = Input(shape=(self.num_actions,))
        self.model = Model([state_input, action_switch], Multiply([self.core_model(state_input), action_switch]))
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
            self.target_model = model
        else:
            self.target_model = model_from_json(model.to_json())
        self.last_state = None
        self.episode = 1
        self.compiled = False
        self.learning = True
        #
        self.ones_single = np.ones((1, num_actions), dtype=np.int8)
        self.ones_batch = np.tile(self.ones_single, (batch_size, 1))
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


