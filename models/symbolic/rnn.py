import os
import json
from datetime import datetime
import tensorflow as tf


class CharLSTM(tf.keras.Model):

    def __init__(self, model_path, data_dimensions = None):
        super(CharLSTM, self).__init__()
        self.data_dimensions = data_dimensions
        self.model_path = model_path
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.tensorboard_logdir = os.path.join(
            model_path,
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir, 'metrics')
        )
        self.file_writer.set_as_default()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model = self.__create_model__(
            data_dimensions["max_timesteps"], 
            data_dimensions["tune_vocab_size"],
            data_dimensions["rhythm_vocab_size"], 
            data_dimensions["meter_vocab_size"],
            data_dimensions["key_vocab_size"]
        )
        self.ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.optimizer,
            net = self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(self.model_path, 'ckpt'),
            max_to_keep = 3
        )
        tokenizer_path = os.path.join(self.model_path, 'tunes_vocab.json')
        with open(tokenizer_path, 'r') as fp:
            self.tunes_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(fp))
        self.idx_to_music = self.tunes_tokenizer.index_word
        print(self.idx_to_music)


    def __create_model__(
            self, 
            max_timesteps,
            tune_vocab_size,
            rhythm_vocab_size,
            meter_vocab_size,
            key_vocab_size
        ):
        #----------------------------------------
        tune = tf.keras.Input(shape=(512 - 1, 1))
        rhythm = tf.keras.Input(shape=(1,))
        meter = tf.keras.Input(shape=(1,))
        key = tf.keras.Input(shape=(1,))
        tune_length = tf.keras.Input(shape=(1,))
        #----------------------------------------
        rhythm_embedding = tf.keras.layers.Embedding(
            input_dim=rhythm_vocab_size + 1,
            output_dim=16,
            name='rhythm_embedding'
        )(rhythm)
        r_temp = tf.keras.layers.Reshape((16,))(rhythm_embedding)

        key_embedding = tf.keras.layers.Embedding(
            input_dim=key_vocab_size + 1,
            output_dim=16,
            name='key_embedding'
        )(key)
        k_temp = tf.keras.layers.Reshape((16,))(key_embedding)

        meter_embedding = tf.keras.layers.Embedding(
            input_dim=meter_vocab_size + 1,
            output_dim=16,
            name='meter_embedding'
        )(meter)
        m_temp = tf.keras.layers.Reshape((16,))(meter_embedding)

        tune_embedding = tf.keras.layers.Embedding(
            input_dim=tune_vocab_size + 1,
            output_dim=32,
            name='tune_embedding',
            mask_zero=True
        )
        t_temp = tune_embedding(tune)
        #tune_mask = tune_embedding.compute_mask(tune)
        #----------------------------------------        
        tune_tensor = tf.keras.layers.Reshape((-1, 32))(t_temp)
        key_tensor = tf.keras.layers.RepeatVector(t_temp.shape[1])(k_temp)
        rhythm_tensor = tf.keras.layers.RepeatVector(t_temp.shape[1])(r_temp)
        meter_tensor = tf.keras.layers.RepeatVector(t_temp.shape[1])(m_temp)
        
        context = tf.keras.layers.Concatenate(axis=-1)(
            [rhythm_tensor, meter_tensor, key_tensor]
        )
        #----------------------------------------

        #----------------------------------------
        full_input = tf.keras.layers.Concatenate(axis=-1)([context, tune_tensor])
        #----------------------------------------
        lstm_output = tf.keras.layers.RNN(
            [tf.keras.layers.LSTMCell(256)],
            return_sequences=True,
            zero_output_for_mask=True,
        )(full_input, mask=tf.reshape(tf.sequence_mask(tune_length, 512 - 1),(-1, 512-1)))
        #----------------------------------------
        next_tokens = tf.keras.layers.Dense(tune_vocab_size)(lstm_output)
        #----------------------------------------
        model = tf.keras.Model(
            inputs=[tune, rhythm, meter, key, tune_length],
            outputs=next_tokens
        )
        #----------------------------------------
        tb_callback = tf.keras.callbacks.TensorBoard(
            os.path.join(self.model_path, 'tensorboard')
        )
        tb_callback.set_model(model=model)
        return model


    def __call_model__(self, context, input_sequence, training=False):
        return self.model([
            tf.reshape(tf.sparse.to_dense(input_sequence),(-1, 512- 1,)), 
            context['R'],
            context['M'],
            context['K'],
            context['tune_length']
        ])


    def call(self, context, sequence, training=False):
        return self.__call_model__(context, sequence['input'], training)

    def grad(self, context, inputs, targets):
        with tf.GradientTape() as tape:
            outputs = self.__call_model__(context, inputs)
            targets = tf.reshape(tf.sparse.to_dense(targets),(-1, 512 - 1))
            loss_value = self.loss_fn(
                y_pred = outputs,
                y_true = targets,
            )
            print(loss_value)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value, outputs


    def update_tensorboard(self, loss, step, grads=None):
        with self.file_writer.as_default():
            tf.summary.scalar("Categorical Cross-Entropy", loss, step=step)
            self.file_writer.flush()

    def map_tokens_to_text(self, output_tensor, sparse):
        return self.tunes_tokenizer.sequences_to_texts(tf.squeeze(output_tensor).numpy())


    def save_model_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))



    def train(self, dataset):
        train_loss_results = []
        train_accuracy_results = []
        print_outputs_frequency = 50
        save_frequency = 100
        num_epochs = 10000
        step = 0
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop - using batches of 32
            for i, (context, sequence) in enumerate(dataset):
                # Optimize the model
                loss_value, outputs = self.grad(
                    context,
                    sequence['input'],
                    sequence['output']
                )
                self.ckpt.step.assign_add(1)
                self.update_tensorboard(loss_value, step)
                if i % print_outputs_frequency is 0:
                    print('-------------------- Input Sequence --------------------')
                    print(self.map_tokens_to_text(tf.sparse.to_dense(sequence['input']), True))
                    print('--------------------------------------------------')
                    print('-------------------- Generated Sequence --------------------')
                    print(self.map_tokens_to_text(tf.argmax(outputs, axis = -1), False))
                    print('--------------------------------------------------')
                    print('-------------------- Target Sequence --------------------')
                    print(self.map_tokens_to_text(tf.sparse.to_dense(sequence['output']), True))
                    print('--------------------------------------------------')
                if i % save_frequency is 0:
                    self.save_model_checkpoint()
                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                #epoch_accuracy.update_state(sequence['output'], self.model(sequence['input'], training=True))
                step = step + 1

            self.save_model_checkpoint()
            tf.saved_model.save(self.model, os.path.join(self.model_dir, 'saved_model'))
            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 50 == 0:
                print(
                    "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    epoch_accuracy.result())
                )


    def generate(self, seed, conditioning_signals, output_path):
        output = self.model(conditioning_signals, seed)
        print(output)
        return output




