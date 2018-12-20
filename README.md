
# Word level and character level text generation using RNN

## Overview

An LSTM model was trained on five parts of Harry Potter series to produce new text whose distribution resembles the original text.

## Dependencies

 - Python 3.6
 - Pytorch
 - Bcolz
 - PyPDF2
## Installation

All the dependency packages can be installed with `pip install` command.
If anaconda distribution is installed then `conda install` can be used.

## Data and pre processing
Five books from Harry Potter series were pre processed and made into a text file such that each line contains a sentence that can be given as an input to the LSTM network.
 

## Training
Both the word level and character level model can be trained by running `python train.py` from the terminal.  The training process also outputs the sampled text after the mentioned iteration interval.

