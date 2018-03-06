from keras import backend as K


def huber_loss(y_true, y_pred):
    diff = y_true - y_pred
    error = K.minimum(0.5 * K.square(diff), K.maximum(K.abs(diff) - 0.5, 0.5))
    return K.mean(error, axis=-1)
