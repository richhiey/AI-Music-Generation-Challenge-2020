import tensorflow as tf
from .helpers import TransformerEncoder, TransformerDecoder

class FolkTransformer(tf.keras.Model):

    def __init__(self, model_configs, data_dimensions)
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

        model_config_path = os.path.join(model_path, 'transformer.json')
        print(model_config_path)
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as fp:
                self.model_configs = json.load(fp)
                print(self.model_configs)  

        tokenizer_path = os.path.join(self.model_path, 'tunes_vocab.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as fp:
                self.tunes_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(fp))

        self.model = Transformer(model_configs, data_dimensions)
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


    def train(self, dataset):
        pass


class Transformer(tf.keras.Model):

    def __init__(self, configs, data_dimensions):
        super(Transformer, self).__init__()
        self.decoder = TransformerDecoder(
            configs["num_layers"],
            configs["d_model"],
            configs["num_heads"],
            configs["dff"],
            configs["target_vocab_size"],
            configs["pe_target"],
            configs["rate"]
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)


    def call(self, inp, tar, training, look_ahead_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights