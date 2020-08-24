import os
import json
import tensorflow as tf


class CharLSTM(tf.keras.Model):

    def __init__(self, model_path, model_config, data_dimensions):
        super(CharLSTM, self).__init__()
        self.model_config = MODEL_CONFIG
        print(self.model_config)
        self.model = self.__create_model__(
            data_dimensions["max_timesteps"]["word_index"], 
            data_dimensions["musical_vocab_size"],
            data_dimensions["rhythm_vocab_size"], 
            data_dimensions["meter_vocab_size"],
            data_dimensions["key_vocab_size"]
        )
        self.model_path = model_path


    def call(self, inputs, training=False):
        return self.model(inputs)


    def train(dataset):
        pass

    def generate(seed, conditioning_signals, output_path):
        pass

    def __create_model__(
            self, 
            max_timesteps,
            musical_vocab_size,
            rhythm_vocab_size,
            meter_vocab_size,
            key_vocab_size
        ):
        musical_tokens = tf.keras.Input(shape=(max_timesteps, musical_vocab_size,))
        rhythm_signal = tf.keras.Input(shape=(rhythm_vocab_size,))
        meter_signal = tf.keras.Input(shape=(meter_vocab_size,))
        key_signal = tf.keras.Input(shape=(key_vocab_size,))


        rhythm_stack = tf.stack([rhythm_signal] * max_timesteps)
        meter_stack = tf.stack([meter_signal] * max_timesteps)
        key_stack = tf.stack([key_signal] * max_timesteps)

        input_signal = tf.concat([rhythm_stack, meter_stack, key_stack], axis=-1)
        input_signal = tf.concat([tf.transpose(musical_tokens, [1, 0, 2]), input_signal], axis=-1)
        print(input_signal)

        lstm_layer = tf.keras.layers.LSTM(max_timesteps)
        lstm_output = lstm_layer(input_signal)
        dense = tf.keras.layers.Dense(512)
        dense_output = dense(lstm_output)
        next_token = tf.keras.layers.Dense(musical_vocab_size)(dense_output)
        print(next_token)
        model = tf.keras.Model(
            inputs=[musical_tokens, rhythm_signal, meter_signal, key_signal],
            outputs=[next_token],
        )
        model.summary()
        return model