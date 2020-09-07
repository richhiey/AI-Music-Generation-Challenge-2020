import os
import json

import tensorflow as tf

from datetime import datetime
from .helpers import TransformerEncoder, TransformerDecoder

class FolkTransformer(tf.keras.Model):

    def __init__(self, model_path, data_dimensions):
        super(FolkTransformer, self).__init__()

        self.data_dimensions = data_dimensions
        self.model_path = model_path

        self.tensorboard_logdir = os.path.join(
            model_path,
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir, 'metrics')
        )
        self.file_writer.set_as_default()

        model_config_path = os.path.join(self.model_path, 'transformer.json')
        print(model_config_path)
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as fp:
                self.model_configs = json.loads(fp.read())
                print(self.model_configs)

        tokenizer_path = os.path.join(self.model_path, 'tunes_vocab.json')
        if os.path.exists(tokenizer_path):
            self.tunes_tokenizer = ABCTokenizer(tokenizer_path)

        self.model = Transformer(self.model_configs, self.data_dimensions)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
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

    def get_configs(self):
        return self.model_configs

    def __call_model__(self, sequence, target, mask, training=True):
        return self.model(sequence, target, mask)


    def call(self, context, sequence, training=True):
        return self.__call_model__(context, sequence, training)


    def grad(self, sequence, target, mask):
        with tf.GradientTape() as tape:
            outputs = self.__call_model__(context, inputs, mask)
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
        sequences = tf.squeeze(output_tensor).numpy()
        texts = self.tunes_tokenizer.sequences_to_texts(sequences)
        for text in texts:
            print('----------------------------------------------------------')
            print(text)


    def save_model_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def calculate_mask(self, sequence, target):
        

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
            for i, (sequence, target) in enumerate(dataset):
                # Optimize the model
                mask = self.calculate_mask(sequence, target)
                loss_value, outputs = self.grad(sequence, target, mask)
                self.ckpt.step.assign_add(1)
                self.update_tensorboard(loss_value, step)
                if i % print_outputs_frequency is 0:
                    print('-------------------- Input Sequence --------------------')
                    self.map_tokens_to_text(tf.sparse.to_dense(sequence['input']), True)
                    print('--------------------------------------------------')
                    print('-------------------- Generated Sequence --------------------')
                    self.map_tokens_to_text(tf.argmax(tf.nn.softmax(outputs), axis = 1), False)
                    print('--------------------------------------------------')
                    print('-------------------- Target Sequence --------------------')
                    self.map_tokens_to_text(tf.sparse.to_dense(sequence['output']), True)
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



class Transformer(tf.keras.Model):

    def __init__(self, configs, data_dimensions):
        super(Transformer, self).__init__()
        self.decoder = TransformerDecoder(
            int(configs["num_layers"]),
            int(configs["d_model"]),
            int(configs["num_heads"]),
            int(configs["dff"]),
            int(configs["target_vocab_size"]),
            int(configs["pe_target"]),
            float(configs["rate"])
        )
        self.final_layer = tf.keras.layers.Dense(
            int(configs["target_vocab_size"])
        )


    def call(self, sequence, target, look_ahead_mask, training=True):

        dec_output, attention_weights = self.decoder(
            sequence, tar, look_ahead_mask, training
        )
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights