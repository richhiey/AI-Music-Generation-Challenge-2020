# Pre-processing utilities for ABC data
import glob
import os 
import json
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
                abc_tune_str += line
        abc_tune_str = '<s>' + abc_tune_str + '</s>'
    return abc_tune_str


def separate_all_abc_tunes(abc_filepath):
    abc_tunes = open(abc_filepath, 'r').read()
    abc_tunes = list(filter(None, abc_tunes.split('\n\n')))
    return abc_tunes

# function to add to JSON 
def write_json(data, filename): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 


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
        self.json_path = os.path.join(output_dir,  output_name + '.json')
        self.tfrecord_path = os.path.join(output_dir, output_name + '.tfrecord')


## Converts a directory of ABC Text Notation files to dataset of tunes
## -------------------------------------------------------------------
## Step 1 - Extract Metadata information, index by ID
## Step 2 - Extract conditioning infomation, index by ID
## Step 3 - Extract Tune, index by ID
## Step 4 - Write tune information to a CSV

class ABCPreProcessor(PreProcessor):

    def process(self):
        write_json([], self.json_path)
        files = glob.glob(self.data_dir + '/**/*.abc', recursive = True)
        
        with open(self.json_path) as json_file: 
            data = json.load(json_file)
            for file_number, file in enumerate(files):
                print('------ ' + str(file_number) + '. ' + file + ' ------')            
                abc_tunes = separate_all_abc_tunes(file)

                for tune in abc_tunes:
                    processed_tune = self.__preprocess_abc_tune__(tune.strip().split('\n'))
                    if (processed_tune['metadata'] and processed_tune['conditional'] and processed_tune['tune']):
                        print('------------------- Extracted Tune --------------------')
                        print(processed_tune)
                        if data is not None:
                            data.append(processed_tune)
                        else:
                            data = [processed_tune]
                        write_json(data, self.json_path)

        print('Number of tunes - ' + str(self.num_files))


    def __preprocess_abc_tune__(self, tune):
        _metadata, metadata_idx = extract_data_from_tune(tune, metadata)
        _conditional, conditional_idx = extract_data_from_tune(tune, conditional)

        keys_to_remove = conditional_idx + metadata_idx
        abc_tune_str = extract_notes_from_tune(tune, keys_to_remove)

        self.num_files = self.num_files + 1
        return {'conditional': _conditional, 'metadata': _metadata, 'tune': abc_tune_str}


    def calculate_statistics(self):
        with open(self.json_path) as json_file: 
            data = json.load(json_file)
        print('Length of data: ' + str(len(data)))

        tunes_str = ''
        keys = []
        meters = []
        rhythms = []
        for i, item in enumerate(data):
            tunes_str += item['tune']
            for k, v in item['conditional'].items():
                if k is 'K':
                    keys.append(v)
                if k is 'R':
                    rhythms.append(v)
                if k is 'M':
                    meters.append(v)
        
        vocab_set = set(list(tunes_str))
        vocab_set.add('<s>')
        vocab_set.add('<s>')
        print('Size of Vocabulary: ' + str(len(vocab_set)))
        print(vocab_set)
        
        keys_set = set(keys)
        keys_set.remove('')
        print('Number of modal keys: ' + str(len(keys_set)))
        print(keys_set)
        
        meters_set = set(meters)
        meters_set.remove('')
        meters_set.remove('C')
        meters_set.remove('C|')
        meters_set.add('4/4')
        meters_set.add('2/2')
        print('Number of musical meters: ' + str(len(meters_set)))
        print(meters_set)
        
        rhythms_set = set(rhythms)
        print('Number of rhythms: ' + str(len(rhythms_set)))
        print(rhythms_set)


    def save_as_tfrecord_dataset(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, char_level=True)
        tunes = []
        keys = []
        meters = []
        rhythms = []
        with open(self.json_path) as json_file: 
            data = json.load(json_file)
            ## Initializing indexes of the dictionary
            for x in data:
                if (x['tune'] and x['conditional'].get('K') and x['conditional'].get('M') and x['conditional'].get('R')):
                    tunes.append(x['tune'])
                    keys.append(x['conditional']['K'])
                    meters.append(x['conditional']['M'])
                    rhythms.append(x['conditional']['R'])
        tokenizer.fit_on_texts(tunes)
        tunes_tensor = tokenizer.texts_to_sequences(tunes)
        tunes_tensor = tf.keras.preprocessing.sequence.pad_sequences(tunes_tensor, padding='post')
        keys_idx, keys_vocab = convert_labels_to_indices(keys)
        #print(keys_idx)
        print(keys_vocab)
        meters_idx, meters_vocab = convert_labels_to_indices(meters)
        #print(meters_idx)
        print(meters_vocab)
        rhythms_idx, rhythms_vocab = convert_labels_to_indices(rhythms)
        #print(rhythms_idx)
        print(rhythms_vocab)
        features_dataset = tf.data.Dataset.from_tensor_slices((tunes_tensor, keys_idx, meters_idx, rhythms_idx))
        print(features_dataset)
        serialized_features_dataset = features_dataset.map(tf_serialize_example)
        
        def generator():
            for features in features_dataset:
                yield serialize_example(*features)

        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator, output_types=tf.string, output_shapes=())
    

        writer = tf.data.experimental.TFRecordWriter(self.tfrecord_path)
        writer.write(serialized_features_dataset)


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
      'feature0': _bytes_feature(tf.io.serialize_tensor(feature0)),
      'feature1': _int64_feature(feature1),
      'feature2': _int64_feature(feature2),
      'feature3': _int64_feature(feature3),
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



class AudioPreProcessor(PreProcessor):
    pass