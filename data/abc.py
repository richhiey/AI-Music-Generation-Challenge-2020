# Pre-processing utilities for ABC data
import glob
import os 

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
    keys_idx = []
    
    for x in list(dictionary.keys()):
        for i, line in enumerate(tune):
            if (line.startswith(x + ':')):
                keys_idx.append(i)
                
    return keys_idx


## Class for pre-processing data to extract meaningful features
## Currently. there are two pre-processors
## ---------------------------------------------------------------
## 1. ABCPreProcessor
## 2. AudioPreProcessor

class PreProcessor(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir    


## Converts a directory of ABC Text Notation files to dataset of tunes
## -------------------------------------------------------------------
## Step 1 - Extract Metadata information, index by ID
## Step 2 - Extract conditioning infomation, index by ID
## Step 3 - Extract Tune, index by ID
## Step 4 - Write tune information to a CSV


class ABCPreProcessor(PreProcessor):

    def process(self):
        files = glob.glob(self.data_dir + '/**/*.abc', recursive = True)

        num_tunes = 0

        for file_number, file in enumerate(files):
            print('------ ' + str(file_number) + '. ' + file + ' ------')
            num_tunes += self.__preprocess_abc_file__(file)
        print('Number of tunes - ' + str(num_tunes))

    def __preprocess_abc_file__(self, abc_txt_file):
        abc_tunes = open(os.path.join(self.data_dir, abc_txt_file), 'r').read()
        abc_tunes = list(filter(None, abc_tunes.split('\n\n')))
        print(len(abc_tunes))

        for tune_id, tune in enumerate(abc_tunes):
            tune = tune.strip().split('\n')

            metadata_keys = extract_data_from_tune(tune, metadata)
            conditional_keys = extract_data_from_tune(tune, conditional)
            keys_to_remove = conditional_keys + metadata_keys

            abc_tune_str = ''
            if keys_to_remove:
                for i, line in enumerate(tune):
                    if not (i in keys_to_remove):
                        abc_tune_str += line

            print('------------------- Extracted Tune --------------------')
            print(abc_tune_str)

        return len(abc_tunes)

    def calculate_statistics(self):
        pass

    def save_as_tfrecord_dataset(self):
        pass


class AudioPreProcessor(PreProcessor):
    pass