import os
import json
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

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
    'validation_freq': 1000,
    'max_steps_for_model': 100000
}

DEFAULT_LR_CONFIG = {
    'initial_lr': 0.001,
    'final_lr': 0.00001,
    'decay_steps': 100000,
}

class FolkLSTM(tf.keras.Model):

    def __init__(self, model_path, data_dimensions, vocab_path = None, training=True, learning_rate = DEFAULT_LR_CONFIG):
        super(FolkLSTM, self).__init__()
        
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

        model_config_path = os.path.join(model_path, 'lstm.json')
        print(model_config_path)
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as fp:
                self.model_configs = json.load(fp)
                print(self.model_configs)
        saved_model_dir = os.path.join(self.model_path, 'folk_lstm')

        self.model = self.__create_model__(self.model_configs, data_dimensions, training)
        
        initial_learning_rate = learning_rate['initial_lr']
        end_learning_rate = learning_rate['final_lr']
        decay_steps = learning_rate['decay_steps']
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
        tune = tf.keras.Input(
            batch_input_shape = (batch_size, data_dimensions['max_timesteps'], 1),
        )
        #----------------------------------------
        tune_embedding_size = int(model_configs['tune_embedding_size'])
        tune_embedding = tf.keras.layers.Embedding(
            input_dim = data_dimensions['tune_vocab_size'],
            output_dim = tune_embedding_size,
            name = 'tune_embedding',
            mask_zero = True
        )(tune)
        tune_tensor = tf.keras.layers.Reshape((-1, tune_embedding_size))(tune_embedding)
        #----------------------------------------                
        stacked_cells = tf.keras.layers.StackedRNNCells(
            self.create_RNN_cells(model_configs['rnn'])
        )

        self.sequential_RNN = self.create_RNN_layer(stacked_cells, stateful)

        rnn_output = self.sequential_RNN(tune_tensor)
        #----------------------------------------
        next_tokens = tf.keras.layers.Dense(data_dimensions['tune_vocab_size'])(rnn_output)
        #----------------------------------------
        model = tf.keras.Model(
            inputs=tune,
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


    def __call_model__(self, input_sequence, sparse=True, training=False):
        if sparse:
            input_sequence = tf.squeeze(tf.sparse.to_dense(input_sequence))
        return self.model([
            input_sequence,
        ])


    def loss_function(self, outputs,  targets, weighted = False):
        mask = tf.math.logical_not(tf.math.equal(outputs, 0))
        loss_ = self.cross_entropy(
            y_pred = outputs, 
            y_true = targets
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    def call(self, sequence, training=False):
        return self.__call_model__(sequence['input'])

    
    def grad(self, context, inputs, targets):
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


    def train(self, dataset, configs = DEFAULT_TRAIN_CONFIG):
        train_loss_results = []
        train_accuracy_results = []

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        # Training loop
        for i, (context, sequence) in enumerate(dataset):

            # Optimize the model
            loss_value, outputs = self.grad(
                context,
                sequence['input'],
                sequence['output']
            )
            self.ckpt.step.assign_add(1)
            self.update_tensorboard(loss_value, tf.cast(self.ckpt.step, tf.int64))
            
            if i % configs['print_outputs_frequency'] == 0:
                abc_outputs = [self.map_to_abc_notation(output) for output in outputs]
                print('---------- Generated Output -----------')
                print(abc_outputs[0])
                print('.......................................')
                # print('-------------------- Input Sequence --------------------')
                # self.map_tokens_to_text(tf.sparse.to_dense(sequence['input']), True)
                # print('--------------------------------------------------')
                # print('-------------------- Generated Sequence --------------------')
                # self.map_tokens_to_text(tf.argmax(tf.nn.softmax(outputs), axis = 1), False)
                # print('--------------------------------------------------')
                # print('-------------------- Target Sequence --------------------')
                # self.map_tokens_to_text(tf.sparse.to_dense(sequence['output']), True)
                # print('--------------------------------------------------')
            
            if i % configs['save_frequency'] is 0:
                self.save_model_checkpoint()
        
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # epoch_accuracy.update_state(sequence['output'], self.model(sequence['input'], training=True))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
       
            if (self.ckpt.step % configs['validation_freq']) == 0:
                print('Step: ' + str(self.ckpt.step) + '\nLoss: ' + str(epoch_loss_avg.result().numpy()) + '\nAccuracy: ' + str(epoch_accuracy.result()))
                
            if (self.ckpt.step >= configs['max_steps_for_model']):
                print('Done with training!')
                break


            
    def complete_tune(self, start_tokens, temperature = 1.0):

        current_token = ''
        text_generated = start_tokens
        start_token_idx = [int(self.vocab['word_to_idx'][token]) for token in start_tokens]
        start_token_idx = tf.expand_dims(start_token_idx, 0)
        seed = tf.convert_to_tensor(start_token_idx, dtype=tf.int32)
        while (1):
           # Add batch dimension
            # Pad to max length
            if (current_token == '</s>'):
                break
            #seed = tf.squeeze(tf.sparse.to_dense(seed))
            predictions = self.model(seed)
            #print(seed)
            #print(predictions)
            # Remove batch dimension
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, 1)[-1,0].numpy()

            if predicted_id:
                seed = tf.expand_dims([predicted_id], 0)
                current_token = self.vocab['idx_to_word'][str(predicted_id)]
                text_generated.append(current_token)

        return text_generated
