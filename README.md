# AI-Music-Generation-Challenge-2020
Hello. This system is called LSTM-512. It also has 512 units inside its [RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) block. 

[GENERATED TUNES](https://drive.google.com/drive/folders/14oErgLoqIgCxI7BVO1WVO722fvgR1iBr?usp=sharing)

Welcome to this repository which contains work being done on modelling [Irish Folk Music](https://en.wikipedia.org/wiki/Irish_traditional_music) using [Deep Learning](http://deeplearning.net/). The task of interest here is to apply [language modelling](https://en.wikipedia.org/wiki/Language_model) techniques to a corpus of [Irish Folk tunes](https://github.com/adactio/TheSession-data) from [TheSession](https://thesession.org/). The [ABC Notation](http://abcnotation.com/) is an important musical notation tool used by various folk musicians around the world to annotate their tunes. So the goal of this little project is to apply language modelling techniques from deep learning on a corpus of Irish Folk Tunes transcribed in the ABC Music Notation Format.

Since the above task is that of [music generation](https://teropa.info/loop/) through text, it also requires thoughts for efficient ways to evaluate the learnt models. Thus, there are small attempts here at trying to evaluate a corpus of computer generated tunes against a smaller corpus of [Irish Double Jigs](http://norbeck.nu/abc/book/). 

I would like to mention that I am not an expert at Irish Music, so this is just me looking at the world of Irish Music from the outside, and trying to understand its structures using Deep Learning. Hence, any feedback is welcome. Thanks!

This project is heavily influenced by the work and ideas of [Bob Sturm](https://highnoongmt.wordpress.com/about/) and the [Folk-RNN](https://folkrnn.org/) project. Thanks for all your insights and research!


## Setup
`python3 -m venv venv`

`source venv/bin/activate`

`git clone https://github.com/richhiey1996/AI-Music-Generation-Challenge-2020`

`pip install --user --upgrade pip`

`pip install -r requirements.txt`

## Data Collection
After completing the setup, you will have to collect a processed set of tunes to create a [Tensorflow Dataset](https://www.tensorflow.org/tutorials/load_data/tfrecord). I have already processed some tunes that I extracted from theSession Data Dump as CSV, so you can find them [here](https://github.com/richhiey1996/AI-Music-Generation-Challenge-2020/wiki/Data-Collection). Since we are trying to model the sequences of notes which make up music, extra metadata fields are irrelevant and are thus removed.

## Preprocessing
Once you have downloaded the above tune collection, you can create a Tensorflow Dataset to get started with the Deep Learning process. The root directory of this project contains [Preprocessing.ipynb](https://github.com/richhiey1996/AI-Music-Generation-Challenge-2020/blob/master/Preprocessing.ipynb) which holds instructions for converting a set of tunes into a TFRecord Dataset. The details involved with PreProcessing of the data can be found [here](https://github.com/richhiey1996/AI-Music-Generation-Challenge-2020/wiki/Preprocessing). You will also have to configure the directory paths according to instructions mentioned in the Preprocessing notebook.

If everything works well, by the end of this, you end up with directory containing the `tune vocabulary` (for mapping musical symbols to numbers), `tfrecord file` (dataset of tunes for the deep learning models) and the `tunes csv file` (file downloaded during data collection

Processed TFRecord Dataset files can be found [here](https://drive.google.com/drive/folders/1HZfZ8LetsaCAZwCMw3Gfo0xoaC_Vg7JD?usp=sharing).

## Experimenting with Language Models on Irish Folk Tunes
Lets run some deep learning experiments now. Since we have a dataset of Irish Folk Tunes in a textual format, it is easy to feed data to various language models and see how well they are able to model the language of music. In order to do this, I provide the models with a basic vocabulary of musical words required to generate fluent ABC text. You can find the vocabulary used over [here](). It is automatically generated during the preprocessing stage.

### Training
First you will have to train a Model. Currently, there are two types of models implemented here - [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) and [RNN Models](https://en.wikipedia.org/wiki/Recurrent_neural_network). There are default configs stored within the `configs` folder for both these models, but you can modify it to create your own models with little hassle. More information is provided in the notebooks. Training can be monitored by looking at your [tensorboard](https://www.tensorflow.org/tensorboard) dashboard.

### Generating
After the training process comes to an end and [checkpoints](https://www.tensorflow.org/guide/checkpoint) are saved, you can restore the model to generate new tunes from what it has already learnt. I generate 10000 tunes from the trained Folk LSTM like model. However, this is a general purpose learning system and the generation process can be tweaked by changing things around within the Model notebooks.

## Evaluation of Generated Material
In context of the [AI Music Generation Challenge 2020](https://boblsturm.github.io/aimusic2020/MusicAI_Challenge_2020.pdf), the tunes generated by such models will be evaluated against a [small corpus of Irish double jigs](http://www.norbeck.nu/abc/book/book.asp?book=2). So with that in mind, I tried simulating something similar.

I created two sets of tunes. One set contained the 365 double jigs. The second corpus contained 100 tunes generated by the model. After this, I converted tunes in both these sets to MIDI using [ABC2MIDI](https://www.systutorials.com/docs/linux/man/1-abc2midi/). You can find a sample run of my evaluation [here](https://drive.google.com/drive/folders/10p9Dcr8BHSk7ujukZOH3LZv9Qud1wK_s?usp=sharing). 

These MIDI files were then moved into Ableton Live and connected to an instrument. I felt that evaluation by listening on a small set of generated content would be a good first step to understand how well the model has trained. After this, I converted these tunes into sheet music to understand how well the visual patterns match across both sets of tunes. [EasyABC](https://nilsliberg.se/ksp/easyabc/) makes it easy to do this. So I just feed it both the ABC files, and inspect the visual patterns in the dataset.

I did think of coming up with metrics and numbers for evaluating the models, but I couldn't imagine how such a metric would be useful in assessing the quality of a set of tunes. I do plan to implement edit distance matrix calculations between the set of generated tunes and the 365 double jigs. But again, I do not know how assessing the similarity of tunes could tell us much about the musical quality of a set of tunes.

## Just some thoughts
I really enjoyed working with the ABC Music Notation, and Irish Folk Music in general. I love how the music is constantly moving, but it can still make you feel at ease. I also learnt that though none of these methods are novel or state-of-the-art in any way, this task can be used as a tool for learning about the possibilites that lie between the worlds of Folk Music (or music in general) and Deep Learning.

## Credits and references
- https://folkrnn.org/
- https://highnoongmt.wordpress.com/
- https://abcnotation.com/
- https://thesession.org/
- https://www.gwern.net/GPT-2-music
- https://ai.ovgu.de
