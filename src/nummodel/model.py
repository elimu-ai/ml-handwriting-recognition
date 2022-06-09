
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import config

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
class CTClayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length*tf.ones(shape=(batch_len,1),dtype="int64")
        label_length = label_length*tf.ones(shape=(batch_len,1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
        
def decoder(y_pred):
    input_shape = tf.keras.backend.shape(y_pred)
    input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
        input_shape[1], 'float32')
    unpadded = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
    unpadded_shape = tf.keras.backend.shape(unpadded)
    padded = tf.pad(unpadded,
                    paddings=[[0, 0], [0, input_shape[1] - unpadded_shape[1]]],
                    constant_values=-1)
    return padded

def deco2(y_pred):
    input_shape = tf.keras.backend.shape(y_pred)
    input_len = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
        input_shape[1], 'float32')
    predictions_decoded = keras.backend.ctc_decode(y_pred, input_length = input_len, greedy=True)[0][0][:, 21]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )
    return sparse_predictions
def CTCDecoder():
    return tf.keras.layers.Lambda(decoder, name='decode')


def build_model(vocab_size):
    input_img = keras.Input(shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))
    x = keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2,2), name="pool1")(x)

    x = keras.layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2,2), name="pool2")(x)

    new_shape = ((config.IMAGE_WIDTH//4), (config.IMAGE_HEIGHT//4)*64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout = 0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    x = keras.layers.Dense(len(vocab_size)+2, activation="softmax", name="dense2")(x)

    output = CTClayer(name="ctc_loss")(labels,x)

    model = keras.models.Model(
        inputs=[input_img,labels], outputs=output, name="hand_recog"
    )

    optim  = keras.optimizers.Adam()
    model.compile(optimizer=optim)
    
    return model

