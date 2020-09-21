import os
import json

import tensorflow as tf

from datetime import datetime
from .helpers import TransformerEncoder, TransformerDecoder

DEFAULT_TRAIN_CONFIGS = {
    'print_outputs_frequency': 100,
    'save_frequency': 1000,
    'num_epochs': 100
}


def load_musical_vocab(vocab_path):
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as fp:
            full_vocab = json.loads(fp.read())
        return full_vocab['idx_to_word']
    else:
        return {}

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class FolkTransformer(tf.keras.Model):

    def __init__(self, model_path, data_dimensions, vocab_path = None):
        super(FolkTransformer, self).__init__()

        self.data_dimensions = data_dimensions
        self.model_path = model_path
        self.vocab = load_musical_vocab(os.path.join(vocab_path, 'tunes_vocab.json'))

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

        self.model = Transformer(self.model_configs, self.data_dimensions)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        learning_rate = CustomSchedule(int(self.model_configs['d_model']))
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1 = 0.9,
            beta_2 = 0.98, 
            epsilon = 1e-9
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

    def get_configs(self):
        return self.model_configs

    def __call_model__(self, sequence, look_ahead_mask, padding_mask, training=True):
        return self.model(
            sequence,
            look_ahead_mask,
            padding_mask
        )

    def call(self, context, sequence, training=True):
        return self.__call_model__(context, sequence, training)


    def grad(self, sequence, targets, look_ahead_mask, padding_mask):
        with tf.GradientTape() as tape:
            outputs, _ = self.__call_model__(sequence, look_ahead_mask, padding_mask)
            loss_value = self.loss_fn(
                y_pred = outputs,
                y_true = targets,
            )
            print(loss_value)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        gradients = [(tf.clip_by_norm(grad, 1.0)) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value, outputs


    def update_tensorboard(self, loss, step, grads=None):
        with self.file_writer.as_default():
            tf.summary.scalar("Categorical Cross-Entropy", loss, step=step)
            self.file_writer.flush()


    def save_model_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def create_padding_mask(self, tune_lengths, max_seq_len):
        return 1 - tf.sequence_mask(
            tune_lengths, 
            maxlen=max_seq_len, 
            dtype=tf.dtypes.float32, 
            name='Padding Mask for Input Sequence'
        )

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def map_to_abc_notation(self, output):
        output = tf.squeeze(tf.argmax(tf.nn.softmax(output), axis = -1)).numpy()
        abc_tokens = []
        for token in output:
            if token:
                abc_tokens.append(self.vocab[str(token)])
        return ''.join(abc_tokens)


    def train(self, dataset, configs = DEFAULT_TRAIN_CONFIGS):
        train_loss_results = []
        train_accuracy_results = []

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(configs['num_epochs']):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            for i, (context, sequence) in enumerate(dataset):
                # Optimize the model
                input_sequence = tf.sparse.to_dense(sequence['input'])
                target_sequence = tf.sparse.to_dense(sequence['output'])
                loss_value, outputs = self.grad(
                    input_sequence,
                    target_sequence,
                    self.create_look_ahead_mask(self.data_dimensions['max_timesteps']),
                    self.create_padding_mask(context['tune_length'], self.data_dimensions['max_timesteps'])
                )
                abc_outputs = [self.map_to_abc_notation(output) for output in outputs]
                if i % configs['print_outputs_frequency'] == 0:
                    print('---------- Generated Output -----------')
                    print(abc_outputs[0])
                    print('.......................................')

                self.ckpt.step.assign_add(1)
                self.update_tensorboard(loss_value, int(self.ckpt.step))
                if i % configs['save_frequency'] is 0:
                    self.save_model_checkpoint()
                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                # epoch_accuracy.update_state(sequence['output'], self.model(sequence['input'], training=True))

            self.save_model_checkpoint()
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
            int(data_dimensions['tune_vocab_size']),
            int(configs["pe_target"]),
            float(configs["rate"])
        )
        self.final_layer = tf.keras.layers.Dense(
            int(data_dimensions['tune_vocab_size'])
        )


    def call(self, sequence, look_ahead_mask, padding_mask):

        dec_output, attention_weights = self.decoder(
            sequence,
            look_ahead_mask, 
            padding_mask
        )
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights