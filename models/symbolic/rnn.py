import os
import json
import tensorflow as tf


class CharLSTM(tf.keras.Model):

    def __init__(self, model_path, data_dimensions):
        super(CharLSTM, self).__init__()
        self.data_dimensions = data_dimensions
        self.model = self.__create_model__(
            data_dimensions["max_timesteps"], 
            data_dimensions["tune_vocab_size"],
            data_dimensions["rhythm_vocab_size"], 
            data_dimensions["meter_vocab_size"],
            data_dimensions["key_vocab_size"]
        )
        self.model_path = model_path


    def __create_model__(
            self, 
            max_timesteps,
            tune_vocab_size,
            rhythm_vocab_size,
            meter_vocab_size,
            key_vocab_size
        ):
        musical_tokens = tf.keras.Input(shape=(max_timesteps, tune_vocab_size,))
        rhythm_signal = tf.keras.Input(shape=(rhythm_vocab_size,))
        meter_signal = tf.keras.Input(shape=(meter_vocab_size,))
        key_signal = tf.keras.Input(shape=(key_vocab_size,))

        rhythm_stack = tf.stack([rhythm_signal] * max_timesteps)
        meter_stack = tf.stack([meter_signal] * max_timesteps)
        key_stack = tf.stack([key_signal] * max_timesteps)

        context_signal = tf.concat(
            [rhythm_stack, meter_stack, key_stack],
            axis=-1
        )
        input_signal = tf.concat(
            [
                tf.keras.preprocessing.sequence.pad_sequences(
                    tf.transpose(
                        musical_tokens,
                        perm=[1,0,2]
                    ),
                    maxlen=self.data_dimensions['max_timesteps']
                ), context_signal
            ],
            axis = -1
        )
        print(input_signal)

        lstm_layer = tf.keras.layers.LSTM(max_timesteps)
        lstm_output = lstm_layer(input_signal)
        dense = tf.keras.layers.Dense(512)
        dense_output = dense(lstm_output)
        next_tokens = tf.keras.layers.Dense(tune_vocab_size)(dense_output)
        print(next_tokens)
        model = tf.keras.Model(
            inputs=[musical_tokens, rhythm_signal, meter_signal, key_signal],
            outputs=[next_tokens],
        )
        return model


    def call(self, context, sequence, training=False):
        return self.model([
            tf.one_hot(
                tf.sparse.to_dense(sequence['tune']),
                self.data_dimensions["tune_vocab_size"]
            ),
            tf.one_hot(context['R'], self.data_dimensions["rhythm_vocab_size"]),
            tf.one_hot(context['M'], self.data_dimensions["meter_vocab_size"]),
            tf.one_hot(context['K'], self.data_dimensions["key_vocab_size"])
        ])


    def train(self, dataset):
        pass


    def generate(self, seed, conditioning_signals, output_path):
        pass