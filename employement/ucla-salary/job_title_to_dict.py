import os
import json
import pandas as pd

job_txt_file = 'job_list.txt'


if not os.path.exists(job_txt_file):
    rawfname = 'raw-uc-salary.parquet.gzip'
    df = pd.read_parquet(rawfname)
    job_list = df['Job Title'].unique().tolist()
    print('job_list',len(job_list))
    with open(job_txt_file,'w') as f:
        for x in job_list:
            f.write(f'{x}\n')

with open(job_txt_file,'r') as f:
    job_list = f.read().split('\n')

tmp_list = []
for x in job_list:
    tmp_list.extend(x.split(' '))
vocab_size = len(set(tmp_list))
print(f'vocab_size {vocab_size}')

sequence_length = max([len(x.split(' ')) for x in job_list])
print(f'max_vec_len {sequence_length}')
print(f'so many job titles, len: {len(job_list)}')


# https://www.tensorflow.org/text/tutorials/word2vec
# Now, create a custom standardization function to lowercase the text and
# remove punctuation.


# Define the vocabulary size and the number of words in a sequence.
print('vocab_size',vocab_size)
print('sequence_length',sequence_length)

import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

text_ds = tf.data.TextLineDataset(job_txt_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])



# 

# use knn to classify job to 32 categories ??
# word2vec



"""

docker run -it -u $(id -u):$(id -g) -v /mnt:/mnt -w $PWD \
    pangyuteng/ml:latest bash

"""