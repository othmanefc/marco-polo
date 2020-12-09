import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input,
from tensorflow.keras.optimizers import SGD


class Bert(object):
    def __init__(self,
                 hub_handle: str,
                 max_seq_length: int = 256,
                 lr: float = 0.01):
        self.bert_layer = hub.KerasLayer(hub_handle, trainable=True)
        self.max_seq_length = max_seq_length

    def build_model(self, num_labels):
        input_word_ids = Input(shape=(self.max_seq_length, ),
                               dtype=tf.int32,
                               name="input_word_ids")
        input_mask = Input(shape=(self.max_seq_length, ),
                           dtype=tf.int32,
                           name="input_mask")
        segment_ids = Input(shape=(self.max_seq_length, ),
                            dtype=tf.int32,
                            name="segment_ids")

        _, sequence_output = self.bert_layer(
            [input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(num_labels, activation="softmax")(clf_output)

        model = Model(inputs=[input_word_ids, input_mask, segment_ids],
                      outputs=out)
        optimizer = SGD(learning_rate=self.lr, momentum=0.8)
        model.compile(loss='SparseCategoricalCrossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
