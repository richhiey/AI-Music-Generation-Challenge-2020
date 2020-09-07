####################################################################
# -------------- Pre-processing utilities for ABC data -------------
####################################################################
import json
import glob
import os 

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from .constants import  METADATA_KEYS, CONDITIONAL_KEYS, \
                        MAX_TIMESTEPS_FOR_ABC_MODEL, \
                        BASE_ABC_VOCABULARY
####################################################################

## -------------------------------------------------------------------
## Parent Class for pre-processing data to extract meaningful features
## -------------------------------------------------------------------
class PreProcessor(object):
    def __init__(self, output_dir, output_name):
        self.num_files = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.json_path = os.path.join(output_dir,  output_name + '.json')
        self.tfrecord_path = os.path.join(output_dir, output_name + '.tfrecord')


## -------------------------------------------------------------------
## Converts a directory of ABC Text Notation files to dataset of tunes
## -------------------------------------------------------------------
class ABCPreProcessor(PreProcessor):
    # =============================================
    # Preprocess all ABC tunes in a directory
    # =============================================
    def process(self, data_dir):
        if not (os.path.exists(self.json_path)):
            print('Processing files and writing extracted information to JSON ..')
            write_json([], self.json_path)
            files = glob.glob(data_dir + '/**/*.abc', recursive = True)
            
            with open(self.json_path) as json_file: 
                data = json.load(json_file)
                for file_number, file in enumerate(files):
                    print('========== ' + str(file_number) + '. ' + file + ' ==========')            
                    abc_tunes = separate_all_abc_tunes(file)

                    for tune in abc_tunes:
                        processed_tune = self.__preprocess_abc_tune__(
                            tune.strip().split('\n')
                        )
                        if (valid_data(processed_tune)):
                            print(processed_tune)
                            self.num_files = self.num_files + 1
                            print('------------------- Extracted Tune ' + str(self.num_files) + ' --------------------')
                            print(processed_tune)
                            if data is not None:
                                data.append(processed_tune)
                            else:
                                data = [processed_tune]
                            write_json(data, self.json_path)

            print('Number of tunes - ' + str(self.num_files))
        else:
            print('The raw data has already been processed. Pre-processed information found at - ' + self.json_path)
        return self.json_path
    # =============================================


    # =============================================
    # Save a bunch of processed ABC tunes as a 
    # TFRecord dataset
    # =============================================
    def save_as_tfrecord_dataset(self):
        if not os.path.exists(self.tfrecord_path):
            print('Preparing to save extracted information into a TFRecord file at ' + self.tfrecord_path + ' ...')
            with open(self.json_path) as json_file:
                data = json.load(json_file)

            tokenizer = create_tunes_tokenizer(data, self.output_dir)
            key_vocab = create_vocabulary(
                get_key_data('K', data),
                os.path.join(self.output_dir, 'K_vocab.json')
            )
            meter_vocab = create_vocabulary(
                get_key_data('M', data),
                os.path.join(self.output_dir, 'M_vocab.json')
            )
            rhythm_vocab = create_vocabulary(
                get_key_data('R', data),
                os.path.join(self.output_dir, 'R_vocab.json')
            )

            print('Created required vocabularies for tokenizing the ABC tunes ...')
            writer = tf.io.TFRecordWriter(self.tfrecord_path)
            print('Creating TFRecord File ...')
            for i, abc_track in enumerate(data):
                print('------------ Serializing example ' + str(i) + ' -------------')
                print(abc_track)
                tokenized_tune = tokenizer.tokenize_tune(
                    '<s>M:' + abc_track['M'] + 'K:' + abc_track['K'] + abc_track['tune'] + '</s>'
                )
                len_tokenized_tune = len(tokenized_tune)
                if (len_tokenized_tune < MAX_TIMESTEPS_FOR_ABC_MODEL):
                    sequence_example = serialize_example(
                        tf.pad(
                            tf.convert_to_tensor(
                                tokenized_tune[:-1],
                                dtype=tf.int64
                            ),
                            [[0, MAX_TIMESTEPS_FOR_ABC_MODEL - len(tokenized_tune)]],
                            mode='CONSTANT'
                        ),
                        tf.pad(
                            tf.convert_to_tensor(
                                tokenized_tune[1:],
                                dtype=tf.int64
                            ),
                            [[0, MAX_TIMESTEPS_FOR_ABC_MODEL - len(tokenized_tune)]],
                            mode='CONSTANT'
                        ),
                        tf.convert_to_tensor(key_vocab[abc_track['K']], dtype=tf.int64),
                        tf.convert_to_tensor(meter_vocab[abc_track['M']], dtype=tf.int64),
                        tf.convert_to_tensor(rhythm_vocab[abc_track['R']], dtype=tf.int64),
                        tf.convert_to_tensor(len(tokenized_tune), dtype=tf.int64)
                    )
                    writer.write(sequence_example)
            writer.close()
            print('Done saving to TFRecord Dataset!')
        else:
            print('The TFRecord file already exists at ' + self.tfrecord_path + ' ...')
        return self.tfrecord_path
    # =============================================


    # =============================================
    # Load an existing TFRecord Dataset of ABC Tunes
    # =============================================
    def load_tfrecord_dataset(self, _path=None):
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        sequence_features = {
            'input': tf.io.VarLenFeature(tf.int64),
            'output': tf.io.VarLenFeature(tf.int64),
        }
        context_features = {
            'K': tf.io.FixedLenFeature([], dtype=tf.int64),
            'M': tf.io.FixedLenFeature([], dtype=tf.int64),
            'R': tf.io.FixedLenFeature([], dtype=tf.int64),
            'tune_length': tf.io.FixedLenFeature([], dtype=tf.int64),
        }

        def _parse_abc_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            context, sequence = tf.io.parse_single_sequence_example(
                example_proto,
                context_features=context_features,
                sequence_features=sequence_features
            )
            # sequence['tune'] = tf.io.decode_raw(sequence['tune'], tf.int64)
            # context['K'] = tf.cast(context['K'], tf.int64)
            # context['M'] = tf.cast(context['M'], tf.int64)
            # context['R'] = tf.cast(context['R'], tf.int64)
            return context, sequence

        parsed_dataset = raw_dataset.map(_parse_abc_function)
        print(parsed_dataset)
        return parsed_dataset
    # =============================================

    def filter_max_length(self, _, sequence, max_length = 512):
        return tf.logical_and(
            tf.size(sequence['input']) <= max_length,
            tf.size(sequence['output']) <= max_length
        )

    # =============================================
    # Run transformations on the dataset to prepare
    # for use with deep learning models
    # =============================================
    def prepare_dataset(self, parsed_dataset, configs = None):
        return (
            parsed_dataset
            .filter(self.filter_max_length)
            .map(self.__pad_to_max_length__)
            # cache the dataset to memory to get a speedup while reading from it.
            .cache()
            .shuffle(256)
            .batch(16)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    # =============================================


    # =============================================
    # Get data dimensions required to create inputs
    # for models
    # =============================================
    def get_data_dimensions(self):
        with open(os.path.join(self.output_dir, 'tunes_vocab.json'), 'r') as fp:
            len_tunes_vocab = len(json.loads(fp.read())['word_to_idx'])
        with open(os.path.join(self.output_dir, 'R_vocab.json'), 'r') as fp:
            len_rhythm_vocab = len(json.loads(fp.read()))
        with open(os.path.join(self.output_dir, 'M_vocab.json'), 'r') as fp:
            len_meter_vocab = len(json.loads(fp.read()))
        with open(os.path.join(self.output_dir, 'K_vocab.json'), 'r') as fp:
            len_key_vocab = len(json.loads(fp.read()))
        return {
            'max_timesteps': MAX_TIMESTEPS_FOR_ABC_MODEL,
            'tune_vocab_size': len_tunes_vocab,
            'rhythm_vocab_size': len_rhythm_vocab,
            'meter_vocab_size': len_meter_vocab,
            'key_vocab_size': len_key_vocab
        } 
    # =============================================


    # =============================================
    # Preprocess a single ABC tune
    # 1. Extract metadata information from the track
    # 2. Extract conditioning signal information 
    #    from the track
    # =============================================
    def __preprocess_abc_tune__(self, tune):
        _metadata, metadata_idx = extract_data_from_tune(tune, METADATA_KEYS)
        _conditional, conditional_idx = extract_data_from_tune(tune, CONDITIONAL_KEYS)

        keys_to_remove = conditional_idx + metadata_idx
        abc_tune_str = extract_notes_from_tune(tune, keys_to_remove)

        return {**{'tune': abc_tune_str}, **_conditional}
    # =============================================


    # =============================================
    # Filter elements from the raw dataset 
    # =============================================
    def __abc_filter_fn__(self, context, sequence):
        return tf.size(sequence['input']) <= 512
    # =============================================


    # =============================================
    # Run transformations on elements in the raw
    # dataset 
    # =============================================
    def __pad_to_max_length__(self, context, sequence):
        return context, sequence
    # =============================================


    # =============================================
    # Pad sequences 
    # =============================================
    def pad_sequences(self, dataset):
        sequences = list(dataset.as_numpy_iterator())
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
        return tf.data.Dataset.from_tensor_slices(padded_sequences)
    # =============================================


    # =============================================
    # Visualize different counts over the ABC dataset
    # =============================================
    def visualize_stats(self):
        with open(self.json_path) as json_file:
            data = json.load(json_file)
            __visualize_dataset_stats__(data)



#########################################################################
# HELPER FUNCTIONS
#########################################################################
class ABCTokenizer():
    def __init__(self, vocab_path, tunes = []):
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as fp:
                self.vocab = json.loads(fp.read())
        else:
            print(tunes)
            uniq_keys = list(set(get_key_data('K', tunes)))
            uniq_keys = ['K:' + key for key in uniq_keys]
            uniq_meters = list(set(get_key_data('M', tunes)))
            uniq_meters = ['M:' + meter for meter in uniq_meters]
            final_vocab = BASE_ABC_VOCABULARY + uniq_meters + uniq_keys
            idx = map(str, list(range(1, len(final_vocab) + 1)))
            word_to_idx = dict(zip(final_vocab, idx))
            idx_to_word = dict(zip(idx, final_vocab))
            self.vocab = {
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }
            write_json(self.vocab, vocab_path)

    def return_vocabulary(self):
        return self.vocab

    def tokenize_tune(self, abc_str):
        i = 0
        tokens = []
        abc_tokens = []
        abc_str_len = len(abc_str)
        chord = False
        while(i < abc_str_len):
            current_token = ''
            if (abc_str[i] == "\""):
                chord = not chord
            if not chord:
                for musical_token in list(self.vocab['word_to_idx'].keys()):
                    len_token = len(musical_token)
                    if (abc_str[i:i+len_token] == musical_token and len_token > len(current_token)):
                        current_token = musical_token
                if (current_token):
                    tokens.append(int(self.vocab['word_to_idx'][current_token]))
                    abc_tokens.append(current_token)
                    i = i + len(current_token)
                else:
                    i = i + 1
            else:
                i = i + 1
        print(abc_tokens)
        return tokens


def plot_with_matplotlib(freq, title, xaxis, yaxis):
    plt.plot(freq)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()

def categorical_hist_with_matplotlib(category_dict, title):
    names = list(category_dict.keys())
    values = list(category_dict.values())
    fig, axs = plt.subplots()
    axs.bar(names, values, 1)
    fig.suptitle(title)
    plt.xticks(rotation='vertical')
    plt.show()

def create_tunes_tokenizer(tunes, output_dir):
    vocab_fp = os.path.join(output_dir, 'tunes_vocab.json')
    tokenizer = ABCTokenizer(vocab_fp, tunes)
    return tokenizer

def create_vocabulary(labels, vocab_fp):
    vocab = dict(zip(set(labels), range(1, len(labels)+1))) 
    write_json(vocab, vocab_fp)
    return vocab

def convert_labels_to_indices(labels, vocab):
    return list(map(lambda x: vocab[x], labels))

def serialize_example(_input, _output, key, meter, rhythm, tune_length):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    example_proto = tf.train.SequenceExample(
        context = tf.train.Features(
            feature = {
                'K': _int64_feature(key),
                'M': _int64_feature(meter),
                'R': _int64_feature(rhythm),
                'tune_length': _int64_feature(tune_length),
            }
        ),
        feature_lists = tf.train.FeatureLists(
            feature_list = {
                'input': tf.train.FeatureList(
                    feature = [
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=_input)
                        )
                    ]  
                ),
                'output': tf.train.FeatureList(
                    feature = [
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=_output)
                        )
                    ]  
                ),
            }
        )
    )
    return example_proto.SerializeToString()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## Helper function to extract reference field information into a processable
## format
def extract_data_from_tune(tune, dictionary):
    data = {}
    data_idx = []
    for x in list(dictionary.keys()):
        for i, line in enumerate(tune):
            if (line.startswith(x + ':')):
                data_idx.append(i)
                data[x] = line.split(':')[1]
    return data, data_idx

def extract_notes_from_tune(tune, keys_to_remove):
    abc_tune_str = ''
    if keys_to_remove:
        for i, line in enumerate(tune):
            if not (i in keys_to_remove):
                abc_tune_str += line.strip()
    return abc_tune_str.replace(" ", "")

def separate_all_abc_tunes(abc_filepath):
    abc_tunes = open(abc_filepath, 'r').read()
    abc_tunes = list(filter(None, abc_tunes.split("<|endoftext|>")))
    return abc_tunes

# function to add to JSON 
def write_json(data, filename): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 

def read_json(filename):
    with open(filename, 'r') as f:
        json_data = json.load(f)
    return json_data

def valid_data(x):
    print(x)
    return (x.get('tune') and x.get('K') and x.get('M'))

def get_key_data(key, data):
    temp = []
    for x in data:
        if (valid_data(x)):
            temp.append(x[key])
    return temp

def __visualize_dataset_stats__(data):
    freq = [len(x['tune']) for x in data]
    freq.sort()
    plot_with_matplotlib(
        freq,
        'Tune lengths over dataset',
        'Tune index',
        'Tune length'
    )

    category_tune_lengths = {}
    tunes_in_each_category = {}
    tunes_in_each_key = {}
    tunes_in_each_meter = {}
    rhythms = set([x['R'] for x in data])
    keys = set([x['K'] for x in  data])
    meters = set([x['M'] for x in  data])
    
    for rhythm in rhythms:
        category_tune_lengths[rhythm] = 0
        tunes_in_each_category[rhythm] = 0
    for key in keys:
        tunes_in_each_key[key] = 0
    for meter in meters:
        tunes_in_each_meter[meter] = 0

    for x in data:
        category_tune_lengths[x['R']] = max(len(x['tune']), category_tune_lengths[x['R']])
        tunes_in_each_category[x['R']] += 1
        tunes_in_each_key[x['K']] += 1
        tunes_in_each_meter[x['M']] += 1

    categorical_hist_with_matplotlib(
        category_tune_lengths,
        'Max tune length across categories'
    )
    categorical_hist_with_matplotlib(
        tunes_in_each_category,
        'Number of tunes in each category'
    )
    categorical_hist_with_matplotlib(
        tunes_in_each_meter,
        'Number of tunes in each meter'
    )
    categorical_hist_with_matplotlib(
        tunes_in_each_key,
        'Number of tunes in each key'
    )