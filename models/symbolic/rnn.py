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
        tune = tf.keras.Input(shape=(512,1))
        rhythm = tf.keras.Input(shape=(1,))
        meter = tf.keras.Input(shape=(1,))
        key = tf.keras.Input(shape=(1,))
        print('---------- ' + 'Input ' + '----------')
        print(tune)
        print(rhythm)
        print(key)
        print(meter)

        print('---------- ' + 'Context ' + '----------')
        rhythm_tensor = tf.keras.backend.repeat(rhythm, 512)
        rhythm_embedding = tf.keras.layers.Embedding(
            input_dim=rhythm_vocab_size,
            output_dim=16,
        )(rhythm_tensor)
        print(rhythm_embedding)

        key_tensor = tf.keras.backend.repeat(key, 512)
        key_embedding = tf.keras.layers.Embedding(
            input_dim=key_vocab_size,
            output_dim=16,
        )(key_tensor)
        print(key_embedding)

        meter_tensor = tf.keras.backend.repeat(meter, 512)
        meter_embedding = tf.keras.layers.Embedding(
            input_dim=meter_vocab_size,
            output_dim=16,
        )(meter_tensor)
        print(meter_embedding)
        context = tf.keras.layers.Concatenate(axis=-1)(
            [rhythm_embedding, meter_embedding, key_embedding]
        )

        tune_embedding = tf.keras.layers.Embedding(
            input_dim=tune_vocab_size,
            output_dim=32
        )(tune)

        print('---------- ' + 'Embeddings ' + '----------')
        print(context)
        print(tune_embedding)

        full_input = tf.keras.layers.Concatenate(axis=-1)(
            [tune_embedding, context]
        )
        print(full_input)
        squeezed = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(full_input)
        print(squeezed)
        print('-------------------------------')

        lstm_layer = tf.keras.layers.LSTM(128, return_sequences=True)
        lstm_output = lstm_layer(squeezed)
        print(lstm_output)
        dense = tf.keras.layers.Dense(tune_vocab_size)
        next_tokens = dense(lstm_output)
        print(next_tokens)
        model = tf.keras.Model(
            inputs=[tune, rhythm, meter, key],
            outputs=next_tokens,
        )
        return model


    def call(self, context, sequence, training=False):
        return self.model([
            tf.reshape(tf.sparse.to_dense(sequence['tune']),(-1, 512, 1)), 
            context['R'],
            context['M'],
            context['K']
        ])


    def train(self, dataset):
        pass


    def generate(self, seed, conditioning_signals, output_path):
        pass