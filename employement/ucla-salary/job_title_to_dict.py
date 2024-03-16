import os
import sys
import json
import pandas as pd
import numpy as np

job_txt_file = 'job_list.txt'
word2vec_file = 'word2vec.json'

def adjust_weights(title):
    title = title.lower()
    manager_code = " aaaa AAAA KKKK ZZZZ aaaa aaaa"
    prof_code = " bbbb bbbb bbbb YYYY XXXX YYYY"
    if '-exec ' in title:
        title+= manager_code
    if 'mgr ' in title:
        title+= manager_code
    if 'supv ' in title:
        title+= manager_code
    if 'supvr ' in title:
        title+= manager_code
    if 'prof- ' in title:
        title+= prof_code
    if 'professor' in title:
        title+= prof_code
    if ' coach' in title:
        title+= prof_code
    if 'dean' in title:
        title+= prof_code
    return title


if not os.path.exists(job_txt_file):
    rawfname = 'raw-uc-salary.parquet.gzip'
    df = pd.read_parquet(rawfname)
    df['Job Title']= df['Job Title'].apply(lambda x: adjust_weights(x))
    job_list = df['Job Title'].unique().tolist()
    print('job_list',len(job_list))
    with open(job_txt_file,'w') as f:
        for x in job_list:
            f.write(f'{x}\n')

if not os.path.exists(word2vec_file):
    with open(job_txt_file,'r') as f:
        job_list = [x for x in f.read().split('\n') if len(x)>0]


    # https://www.tensorflow.org/text/tutorials/word2vec
    # Now, create a custom standardization function to lowercase the text and
    # remove punctuation.


    # Define the vocabulary size and the number of words in a sequence.

    tmp_list = []
    for x in job_list:
        tmp_list.extend(x.split(' '))
    vocab_size = len(set(tmp_list))
    print(f'vocab_size {vocab_size}')

    sequence_length = max([len(x.split(' ')) for x in job_list])
    print(f'max_vec_len {sequence_length}')

    import io
    import re
    import string
    import tensorflow as tf
    from tensorflow.keras import layers

    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE

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

    print(len(inverse_vocab))
    print(inverse_vocab[:20])


    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences))

    print(sequences[0])
    print(sequences[-1])

    print(f'so many job titles, len: {len(job_list)}')
    print('vocab_size',vocab_size)
    print('sequence_length',sequence_length)
    print(type(sequences[0]))
    mydict = {}
    for job_title,vec in zip(job_list,sequences):
        mydict[job_title]=vec.tolist()

    with open(word2vec_file,'w') as f:
        f.write(json.dumps(mydict,indent=True,default=str,sort_keys=True))

with open(word2vec_file,'r') as f:
    word2vec = json.loads(f.read())

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
X = np.array(list(word2vec.values()))
K = np.array(list(word2vec.keys()))
print(X.shape)
kmeans = KMeans(n_clusters=128).fit(X)
Y = kmeans.labels_

job_cat_dict = {}
for category,title in zip(Y,K):
    category = int(category)
    title = str(title)
    if category not in job_cat_dict.keys():
        job_cat_dict[category]=[]
    job_cat_dict[category].append(title)

with open('job-cat.json','w') as f:
    f.write(json.dumps(job_cat_dict,indent=True,default=str))

# use knn to classify job to 32 categories ??
# word2vec


"""

docker run -it -u $(id -u):$(id -g) -v /mnt:/mnt -w $PWD \
    pangyuteng/ml:latest bash

rm *.json *.txt
python job_title_to_dict.py > ok.txt

"""