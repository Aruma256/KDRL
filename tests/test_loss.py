from unittest import TestCase
from kdrl.loss import *
import numpy as np
from keras import backend as K


class TestPolicy(TestCase):

    def test_huber_loss(self):
        _TEST_SHAPE = (16, 100)
        y_true = np.random.uniform(-3, 3, _TEST_SHAPE)
        y_pred = np.random.uniform(-3, 3, _TEST_SHAPE)
        output = K.eval(huber_loss(K.variable(y_true), K.variable(y_pred)))
        y = y_true - y_pred
        assumed = np.mean(np.where(np.abs(y) < 1, 0.5 * np.square(y), np.abs(y) - 0.5), axis=-1)
        self.assertEqual(output.shape, assumed.shape)
        self.assertEqual(output.shape, _TEST_SHAPE[:-1])
        self.assertTrue(np.allclose(output, assumed))

