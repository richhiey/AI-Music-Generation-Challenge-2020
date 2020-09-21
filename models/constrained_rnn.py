import os
import json
from datetime import datetime
import tensorflow as tf

def load_musical_vocab(vocab_path):
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as fp:
            full_vocab = json.loads(fp.read())
        return full_vocab
    else:
        return {}

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

DEFAULT_TRAIN_CONFIG = {
    'print_outputs_frequency': 100,
    'save_frequency': 100,
    'num_epochs': 100,
}


class AnticipationLSTM(tf.keras.Model):

    def __init__(self, model_path, data_dimensions, vocab_path = None, training=True, initial_learning_rate = 0.01):
        super(AnticipationLSTM, self).__init__()
        
        self.data_dimensions = data_dimensions
        self.model_path = model_path
        self.tunes_vocab = load_musical_vocab(os.path.join(vocab_path, 'tunes_vocab.json'))
        self.constraint_vocab = load_musical_vocab(os.path.join(vocab_path, 'tunes_vocab.json'))

        self.tensorboard_logdir = os.path.join(
            model_path,
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir, 'metrics')
        )
        self.file_writer.set_as_default()

        model_config_path = os.path.join(model_path, 'lstm.json')
        print(model_config_path)
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as fp:
                self.model_configs = json.load(fp)
                print(self.model_configs)
        saved_model_dir = os.path.join(self.model_path, 'folk_lstm')

        self.music_model = self.__create_model__(self.model_configs, data_dimensions, training)
        self.contraint_model = self.create_model(self.model_configs, data_dimensions, training)

        if training:
            end_learning_rate = 0.00001
            decay_steps = 100000.0
            decay_rate = 0.
            learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
              initial_learning_rate, decay_steps, end_learning_rate, power=3
            )

            self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        
        
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
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def get_configs(self):
        return self.model_configs


    def __create_model__(self, model_configs, data_dimensions, training):
        #----------------------------------------
        if training:
            batch_size = data_dimensions['batch_size']
        else:
            batch_size = 1
        stateful = not training
        #----------------------------------------        
        constraints = tf.keras.Input(
            batch_input_shape = (batch_size, data_dimensions['max_timesteps'], 1),
        )
        tune = tf.keras.Input(
            batch_input_shape = (batch_size, data_dimensions['max_timesteps'], 1),
        )
        #----------------------------------------
        tune_embedding = tf.keras.layers.Embedding(
            input_dim = data_dimensions['tune_vocab_size'],
            output_dim = int(model_configs['tune_embedding_size']),
            name = 'tune_embedding',
            mask_zero = True
        )(tune)
        tune_tensor = tf.keras.layers.Reshape((-1, tune_embedding_size))(tune_embedding)
        
        constraint_embedding = tf.keras.layers.Embedding(
            input_dim = data_dimensions['constraint_vocab_size'],
            output_dim = int(model_configs['constraint_embedding_size']),
            name = 'constraint_embedding',
            mask_zero = True
        )(tune)
        constraint_tensor = tf.keras.layers.Reshape((-1, constraint_embedding_size))(constraint_embedding)
        #----------------------------------------                
        stacked_cells_1 = tf.keras.layers.StackedRNNCells(
            self.create_RNN_cells(model_configs['rnn'])
        )
        self.sequential_RNN = self.create_RNN_layer(stacked_cells, stateful)
        
        stacked_cells_2 = tf.keras.layers.StackedRNNCells(
            self.create_RNN_cells(model_configs['rnn'])
        )
        self.constraint_rnn = self.create_RNN_layer(stacked_cells, stateful)
        #----------------------------------------                
        rnn_inputs = tf.keras.layers.Concatenate([tune_tensor, self.constraint_rnn(constraint_tensor)])
        rnn_output = self.sequential_RNN(rnn_inputs)
        #----------------------------------------
        next_tokens = tf.keras.layers.Dense(data_dimensions['tune_vocab_size'])(rnn_output)
        #----------------------------------------
        model = tf.keras.Model(
            inputs=[tune, constraints],
            outputs=next_tokens
        )
        #----------------------------------------
        return model


    def create_RNN_cells(self, configs):
        if configs['unit_type'] == 'lstm':
            RNN_unit = tf.keras.layers.LSTMCell
        else:
            RNN_unit = tf.keras.layers.GRUCell
        return [RNN_unit(int(configs['num_units'])) for _ in range(int(configs['num_layers']))]


    def create_RNN_layer(self, cells, stateful = False, go_backwards = False):
        return tf.keras.layers.RNN(
            cells,
            stateful = stateful,
            go_backwards = go_backwards,
            return_sequences = True,
            zero_output_for_mask = True,
        )


    def __call_model__(self, tune, constraints, sparse=True):
        if sparse:
            input_sequence = tf.squeeze(tf.sparse.to_dense(tune))
        return self.model([input_sequence, constraints])


    def loss_function(self, outputs,  targets, weighted = False):
        mask = tf.math.logical_not(tf.math.equal(outputs, 0))
        loss_ = self.cross_entropy(
            y_pred = outputs, 
            y_true = targets
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    def call(self, tune, constraints, training=False):
        return self.__call_model__(sequence['input'])

    
    def grad(self, constraints, inputs, targets):
        with tf.GradientTape() as tape:
            outputs = self.__call_model__(inputs)
            targets = tf.reshape(tf.sparse.to_dense(targets), (-1, 255))
            loss_value = self.loss_function(
                outputs = outputs,
                targets = targets
            )
            print(loss_value)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        gradients = [(tf.clip_by_norm(grad, 3.0)) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value, outputs


    def update_tensorboard(self, loss, step, grads=None):
        with self.file_writer.as_default():
            tf.summary.scalar("Categorical Cross-Entropy", loss, step=step)
        self.file_writer.flush()

        
    def map_to_abc_notation(self, output):
        output = tf.squeeze(tf.argmax(tf.nn.softmax(output), axis = -1)).numpy()
        abc_tokens = []
        for token in output:
            if token:
                abc_tokens.append(self.vocab['idx_to_word'][str(token)])
        return ''.join(abc_tokens)


    def save_model_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
