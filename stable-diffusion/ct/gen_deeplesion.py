#!/usr/local/bin/python

import os
import sys
import pandas as pd
from pathlib import Path
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
from keras.utils import Sequence
import cv2
import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
"""
## Hyperparameters
"""

# data
dataset_repetitions = 5
num_epochs = 50  # train for at least 50 epochs for good results
image_size = 512
batch_size = 16
# image_size = 64
# batch_size = 64

min_val,max_val = -1000,1000
def png_read(file_path):
    file_path = file_path.decode('utf-8')
    image = cv2.imread(file_path, -1)  # -1 is needed for 16-bit image
    image = (image.astype(np.int32) - 32768).astype(np.int16) # HU
    image = image.astype(np.float32)
    image = ((image-min_val)/(max_val-min_val)).clip(0,1)-0.5
    image = np.expand_dims(image,axis=-1)
    dummpy = np.array([0.0]).astype(np.float32)
    return image, dummpy

def parse_fn(file_path):
    image, dummy = tf.numpy_function(
        func=png_read, 
        inp=[file_path],
        Tout=[tf.float32, tf.float32],
    )
    image = tf.cast(image, tf.float32)
    image = tf.tile(image, [1,1,1])
    image = tf.image.resize(image, [image_size,image_size],antialias=True)
    image = tf.reshape(image,[image_size,image_size,1]) # so tf won't complain about unknown image size
    return image

def cache_png_file_paths():
    directory = '/mnt/scratch/data/DeepLesion/Images_png'
    path_list = [{'png_path':str(x)} for x in Path(directory).rglob('*.png')]
    df = pd.DataFrame(path_list)
    df.to_csv('pngs.csv',index=False)

def prepare_dataset():
    if not os.path.exists('pngs.csv'):
        cache_png_file_paths()
    df = pd.read_csv('pngs.csv')
    path_list = df.png_path.tolist()

    norm_filenames = tf.constant(path_list[:1000])
    norm_ds = tf.data.Dataset.from_tensor_slices(norm_filenames).repeat(dataset_repetitions).shuffle(10 * batch_size).map(
        parse_fn, num_parallel_calls=tf.data.AUTOTUNE
    )

    train_filenames = tf.constant(path_list[:10000])
    train_filenames = tf.constant(path_list[:-1000])
    train_ds = tf.data.Dataset.from_tensor_slices(train_filenames).repeat(dataset_repetitions).shuffle(10 * batch_size).map(
        parse_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_filenames = tf.constant(path_list[-1000:])
    val_ds = tf.data.Dataset.from_tensor_slices(val_filenames).repeat(dataset_repetitions).shuffle(10 * batch_size).map(
        parse_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    return norm_ds, train_ds, val_ds

if __name__ == "__main__":
    norm_dataset, train_dataset , val_dataset = prepare_dataset()

    plt.figure(figsize=(10, 10))
    for images in train_dataset.take(1):
        print(images.shape)
        for i in range(batch_size):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i,:].numpy()+0.5,cmap='gray')
            plt.axis("off")
            if i > 7 :
                break
        os.makedirs('tmp',exist_ok=True)
        plt.savefig(f"tmp/test-deeplesion.png")
        plt.close()
