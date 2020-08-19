# Pre-processing utilities for ABC data
import json
import glob
import os 

import pandas as pd
import tensorflow as tf
####################################################################

## http://abcnotation.com/wiki/abc:standard:v2.1
## ---------------------------------------------
## Reference Fields that contain  information about a transcribed tune

metadata = {
    'A':'area',
    'B':'book',
    'C':'composer',
    'D':'discography',
    'F':'file url',
    'G':'group',
    'H':'history',
    'I':'instruction',
    'L':'unit note length',
    'm':'macro',
    'N':'notes',
    'O':'origin',
    'P':'parts',
    'Q':'tempo',
    'r':'remark',
    'S':'source',
    's':'symbol line',
    'T':'tune title',
    'U':'user defined',
    'V':'voice',
    'W':'words',
    'w':'words',
    'X':'reference number',
    'Z':'transcription',
}


## Reference fields to be used as conditioning for the symbolic models

conditional = {
    'K':'key',
    'M':'meter',
    'R':'rhythm',
}


## Class for pre-processing data to extract meaningful features
## Currently. there are two pre-processors
## ---------------------------------------------------------------
## 1. ABCPreProcessor
## 2. AudioPreProcessor

class PreProcessor(object):
    def __init__(self, data_dir, output_dir, output_name):
        self.data_dir = data_dir
        self.num_files = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.json_path = os.path.join(output_dir,  output_name + '.json')
        self.tfrecord_path = os.path.join(output_dir, output_name + '.tfrecord')


## Converts a directory of ABC Text Notation files to dataset of tunes
## -------------------------------------------------------------------
## Step 1 - Extract Metadata information, index by ID
## Step 2 - Extract conditioning infomation, index by ID
## Step 3 - Extract Tune, index by ID
## Step 4 - Write tune information to a CSV

class ABCPreProcessor(PreProcessor):

    def prepare_dataset(self, processed_dataset):
        return processed_dataset


    def process(self):
        if not (os.path.exists(self.json_path)):
            write_json([], self.json_path)
            files = glob.glob(self.data_dir + '/**/*.abc', recursive = True)
            
            with open(self.json_path) as json_file: 
                data = json.load(json_file)
                for file_number, file in enumerate(files):
                    print('------ ' + str(file_number) + '. ' + file + ' ------')            
                    abc_tunes = separate_all_abc_tunes(file)

                    for tune in abc_tunes:
                        processed_tune = self.__preprocess_abc_tune__(tune.strip().split('\n'))
                        print(processed_tune)
                        if (valid_data(processed_tune)):
                            print('------------------- Extracted Tune --------------------')
                            print(processed_tune)
                            if data is not None:
                                data.append(processed_tune)
                            else:
                                data = [processed_tune]
                            write_json(data, self.json_path)
                            self.num_files = self.num_files + 1

            print('Number of tunes - ' + str(self.num_files))
        else:
            print('The raw data has already been processed. Pre-processed information found at - ' + self.json_path)
        return self.json_path


    def __preprocess_abc_tune__(self, tune):
        _metadata, metadata_idx = extract_data_from_tune(tune, metadata)
        _conditional, conditional_idx = extract_data_from_tune(tune, conditional)

        keys_to_remove = conditional_idx + metadata_idx
        abc_tune_str = extract_notes_from_tune(tune, keys_to_remove)

        return {**{'tune': abc_tune_str}, **_conditional}


    def calculate_statistics(self):
        pass


    def save_as_tfrecord_dataset(self):
        if not os.path.exists(self.tfrecord_path):
            tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, char_level=True)
            with open(self.json_path) as json_file: 
                data = json.load(json_file)

            tunes = get_key_data('tune', data)
            tokenizer.fit_on_texts(tunes)
            write_json(tokenizer.to_json(), os.path.join(self.output_dir, 'tunes_vocab.json'))
            tunes_tensor = tokenizer.texts_to_sequences(tunes)
            tunes_tensor = tf.keras.preprocessing.sequence.pad_sequences(tunes_tensor, padding='post')
            

            keys = get_key_data('K', data) 
            keys_idx, keys_vocab = convert_labels_to_indices(keys)
            write_json(keys_vocab, os.path.join(self.output_dir, 'keys_vocab.json'))
 
            meters = get_key_data('M', data)
            meters_idx, meters_vocab = convert_labels_to_indices(meters)
            write_json(meters_vocab, os.path.join(self.output_dir, 'meters_vocab.json'))
 
            rhythms = get_key_data('R', data)
            rhythms_idx, rhythms_vocab = convert_labels_to_indices(rhythms)
            write_json(rhythms_vocab, os.path.join(self.output_dir, 'rhythms_vocab.json'))
 
            features_dataset = tf.data.Dataset.from_tensor_slices((
                tunes_tensor, 
                keys_idx, 
                meters_idx, 
                rhythms_idx
            ))
            print(features_dataset)
            
            def generator():
                for features in features_dataset:
                    yield serialize_example(*features)

            serialized_features_dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=tf.string,
                output_shapes=()
            )
            print(serialized_features_dataset)    

            print('Creating TFRecord File ...')
            writer = tf.data.experimental.TFRecordWriter(self.tfrecord_path)
            writer.write(serialized_features_dataset)
            print('Done!')
        else:
            print('The TFRecord file already exists at ' + self.tfrecord_path + ' ...')
        return self.tfrecord_path


    def load_tfrecord_dataset(self, tfrecord_path):
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

        # Create a dictionary describing the features.
        tune_feature_description = {
            'tune': tf.io.FixedLenFeature([], tf.string),
            'key': tf.io.FixedLenFeature([], tf.int64),
            'meter': tf.io.FixedLenFeature([], tf.int64),
            'rhythm': tf.io.FixedLenFeature([], tf.int64),
        }
        def _parse_abc_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, tune_feature_description)

        parsed_image_dataset = raw_dataset.map(_parse_abc_function)
        return parsed_image_dataset



def convert_labels_to_indices(labels):
    mapping = dict(zip(set(labels), range(len(labels))))
    outputs = list(map(lambda x: mapping[x], labels))
    return outputs, mapping

def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'tune': _bytes_feature(tf.io.serialize_tensor(feature0)),
      'key': _int64_feature(feature1),
      'meter': _int64_feature(feature2),
      'rhythm': _int64_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar

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
    abc_tunes = list(filter(None, abc_tunes.split('\n\n')))
    return abc_tunes

# function to add to JSON 
def write_json(data, filename): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 

def valid_data(x):
    return (x.get('tune') and x.get('K') and x.get('M') and x.get('R'))

def get_key_data(key, data):
    temp = []
    for x in data:
        if (valid_data(x)):
            temp.append(x[key])
    return temp