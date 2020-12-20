from datetime import datetime
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

from src.train.optimizer import AdamWarmup
from datasets.utils import create_dir


class Bert:

    def __init__(
        self,
        hub_handle: str,
        max_seq_length: int = 256,
    ):
        self.bert_layer = hub.KerasLayer(hub_handle, trainable=True)
        self.max_seq_length = max_seq_length

    def build_model(self, num_labels: int, lr: float, decay_steps: int,
                    warmup_steps: int, weight_decay: float) -> tf.keras.Model:
        input_word_ids = Input(shape=(self.max_seq_length,),
                               dtype=tf.int32,
                               name="input_word_ids")
        input_mask = Input(shape=(self.max_seq_length,),
                           dtype=tf.int32,
                           name="input_mask")
        segment_ids = Input(shape=(self.max_seq_length,),
                            dtype=tf.int32,
                            name="segment_ids")

        _, sequence_output = self.bert_layer(
            [input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        if num_labels == 2:
            out = Dense(1, activation="sigmoid")(clf_output)
        elif num_labels > 2:
            out = Dense(num_labels, activation="softmax")(clf_output)
        else:
            raise ValueError("num_labels should be higher than 1")

        model = Model(inputs=[input_word_ids, input_mask, segment_ids],
                      outputs=out)
        optimizer = AdamWarmup(lr=lr,
                               decay_steps=decay_steps,
                               warmup_steps=warmup_steps,
                               weight_decay=weight_decay)
        if num_labels == 2:
            model.compile(loss="binary_crossentropy",
                          optimizer=optimizer,
                          metrics=["accuracy"])
        else:
            model.compile(loss='SparseCategoricalCrossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        return model

    def get_checkpoint(self,):
        date_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = os.path.join("models", "models", f"BERT_{date_now}")
        create_dir(checkpoint_dir)
        file = os.path.join(checkpoint_dir,
                            "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=file,
            monitor="loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        return model_checkpoint

    def get_tensorboard(self,):
        date_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("models", "logs", f"BERT_{date_now}")
        create_dir(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)
        return tensorboard_callback

    def callbacks(self,):
        return [self.get_checkpoint(), self.get_tensorboard()]
