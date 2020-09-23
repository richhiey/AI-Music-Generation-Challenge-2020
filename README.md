# AI-Music-Generation-Challenge-2020

Hello. Welcome to this repository which contains work being done on modelling [Irish Folk Music](https://en.wikipedia.org/wiki/Irish_traditional_music) using Deep Learning. The task of interest here is to apply language modelling techniques to a corpus of traditional Irish Folk tunes from [TheSession](https://thesession.org/). The [ABC Notation](http://abcnotation.com/) is an important musical notation tool used by various folk musicians around the world to annotate their tunes. So the goal of this little project is to apply language modelling techniques from deep learning on a corpus of Irish Folk Tunes transcribed in the ABC Music Notation Format.

Since the above task is that of music generation through text, it also requires thoughts for efficient ways to evaluate the learnt models. Thus, there are small attempts here at trying to evaluate a corpus of computer generated tunes against a smaller corpus of [Irish Double Jigs](http://norbeck.nu/abc/book/). 

I would like to mention that I am not an expert at Irish Music, so this is just me looking at the world of Irish Music from the outside, and trying to understand its structures using Deep Learning. Hence, any feedback is welcome. Thanks!

This project is heavily influenced by the work and ideas of [Bob Sturm](https://www.kth.se/profile/bobs) and the [Folk-RNN](https://folkrnn.org/) project. Thanks for all your insights and research!


## Setup
`python3 -m venv venv`

`source venv/bin/activate`

`git clone https://github.com/richhiey1996/AI-Music-Generation-Challenge-2020`

`pip install --user --upgrade pip`

`pip install -r requirements.txt`

## Data Collection
After completing the setup, you will have to collect a processed set of tunes to create a Tensorflow Dataset. I have already processed some tunes that I extracted from theSession Data Dump as CSV, so you can find them [here (ADD TUNES!!)](). Since we are trying to model the sequences of notes which make up the music, extra metadata fields are irrelevant and are thus removed.

## Preprocessing
Once you have downloaded the above tune collection, you can create a Tensorflow Dataset to get started with the Deep Learning process. The root directory of this project contains 'Preprocessing.ipynb' which holds instructions for converting a set of tunes into a TFRecord Dataset. The details involved with PreProcessing of the data can be found [here (ADD TUNES!!)](). You also might have to configure the directory paths according to instructions mentioned in the Preprocessing notebook.

If everything works well, by the end of this, you end up with directory containing the `tune vocabulary` (for mapping musical symbols to numbers), `tfrecord file` (dataset of tunes for the deep learning models) and the `tunes csv file` (file downloaded during data collection)

## Experimenting with Language Models on Irish Folk Tunes
Lets run some deep learning experiments now. Since we have a dataset of Irish Folk Tunes, it is easy to feed data to various language models and see how well they are able to model the language of music. There are two phases here -

### Training
First you will have to train a Model. Currently, there are two types of models implemented here - [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) and [RNN Models](https://en.wikipedia.org/wiki/Recurrent_neural_network). There are default configs stored within the `configs` folder for both these models, but you can modify it to create your own models with little hassle. More information is provided in the notebooks. Training can be monitored by looking at your [tensorboard](https://www.tensorflow.org/tensorboard) dashboard.

### Generating
After the training process comes to an end and checkpoints are saved, you can restore the model to generate new tunes from what it has already learnt. In context of the [AI Music Generation Challenge](https://boblsturm.github.io/aimusic2020/MusicAI_Challenge_2020.pdf), I generate 10000 tunes from the trained Folk LSTM model. However, this is a general purpose learning system and the generation process can be tweaked by changing things around within the Model notebooks.

## Evaluation of Generated Material
Need to add information here ..

## Just some thoughts
Need to add information here ..

## Credits and references
Need to add information here ..
