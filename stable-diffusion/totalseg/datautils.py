
import os
import sys
import math
import traceback
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

from tensorflow import keras
from keras import layers

THIS_DIR = "/mnt/hd1/aigonewrong/stable-diffusion/totalseg"
TMP_DIR = os.path.join(THIS_DIR,'tmp')
NIFTI_FILE = os.path.join(THIS_DIR,'niftis.csv')

# data
dataset_repetitions = 10
image_size = 512
batch_size = 16

label_count = 105
min_val,max_val = -1000,1000
axis = 2
WH = image_size
THICKNESS = 1
TARGET_SHAPE = (WH,WH,THICKNESS)
IMG_SIZE = (WH,WH,THICKNESS,1)

def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image_left"])[0]
    width = tf.shape(data["image_left"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image_left"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)


    label = tf.image.crop_to_bounding_box(
        data["segmentation_label"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )
    label = tf.cast(label, dtype=tf.float32)
    label = tf.image.resize(label, size=[image_size, image_size], antialias=False,method='nearest')

    return tf.clip_by_value(image / 255.0, 0.0, 1.0),tf.clip_by_value(label / label_count, 0.0, 1.0)

def nifti_read(folder_path):
    
    folder_path = os.path.dirname(folder_path.decode('utf-8'))

    image_path = os.path.join(folder_path,'ct.nii.gz')
    mask_path = os.path.join(folder_path,'segmentations.nii.gz')

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(image_path)
    file_reader.ReadImageInformation()
    image_size = file_reader.GetSize()

    # attempt to augment z spacing to be within 0.4 to 5.4mm
    extract_size = list(file_reader.GetSize())
    current_index = [0] * file_reader.GetDimension()

    mylist = np.arange(0,image_size[axis]-THICKNESS,1)
    idx = int(np.random.choice(mylist))
    current_index[axis] = idx
    extract_size[axis] = THICKNESS

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(image_path)
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    image_obj = file_reader.Execute()
    img = sitk.GetArrayFromImage(image_obj)

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(mask_path)
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    mask_obj = file_reader.Execute()
    mask = sitk.GetArrayFromImage(mask_obj)

    min_axis = int(np.argmin(img.shape))
    img = np.swapaxes(img,min_axis,-1)
    mask = np.swapaxes(mask,min_axis,-1)

    return img.astype(np.float32), mask.astype(np.float32)


def parse_fn(file_path):
    image, mask = tf.numpy_function(
        func=nifti_read, 
        inp=[file_path],
        Tout=[tf.float32, tf.float32],
    )
    image = tf.cast(image, tf.float32)
    image = tf.tile(image, [1,1,1]) # trick tf so image is of the proper type.
    image = tf.image.resize(image, [image_size,image_size],antialias=True)
    image = tf.reshape(image,[image_size,image_size,1]) # so tf won't complain about unknown image size

    mask = tf.cast(mask, tf.float32)
    mask = tf.tile(mask, [1,1,1])
    mask = tf.image.resize(mask, [image_size,image_size], antialias=False,method='nearest')
    mask = tf.reshape(mask,[image_size,image_size,1]) # so tf won't complain about unknown image size

    image = (image-min_val)/(max_val-min_val)
    mask = mask / label_count
    return tf.clip_by_value(image, 0.0, 1.0)-0.5, tf.clip_by_value(mask, 0.0, 1.0)

def parse_fn_one(file_path):
    image, mask = tf.numpy_function(
        func=nifti_read, 
        inp=[file_path],
        Tout=[tf.float32, tf.float32],
    )
    image = tf.cast(image, tf.float32)
    image = tf.tile(image, [1,1,1]) # trick tf so image is of the proper type.
    image = tf.image.resize(image, [image_size,image_size],antialias=True)
    image = tf.reshape(image,[image_size,image_size,1]) # so tf won't complain about unknown image size

    mask = tf.cast(mask, tf.float32)
    mask = tf.tile(mask, [1,1,1])
    mask = tf.image.resize(mask, [image_size,image_size], antialias=False,method='nearest')
    mask = tf.reshape(mask,[image_size,image_size,1]) # so tf won't complain about unknown image size

    image = (image-min_val)/(max_val-min_val)
    mask = mask / label_count
    return tf.clip_by_value(image, 0.0, 1.0)-0.5


def cache_file_paths():
    directory = '/mnt/scratch/data/Totalsegmentator_dataset'
    path_list = []
    for x in Path(directory).rglob('ct.nii.gz'):
        try:
            image_path = str(x)
            print(image_path)
            mask_path = os.path.join(os.path.dirname(image_path),'segmentations.nii.gz')
            if not os.path.exists(mask_path):
                continue
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(mask_path)
            file_reader.ReadImageInformation()
            extract_size = list(file_reader.GetSize())
            extract_size[2]=1
            current_index = [0] * file_reader.GetDimension()
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(mask_path)
            file_reader.SetExtractIndex(current_index)
            file_reader.SetExtractSize(extract_size)
            mask_obj = file_reader.Execute()
            print(mask_path)
            image_path = os.path.join(os.path.dirname(image_path),'ct.nii.gz')
            if not os.path.exists(image_path):
                continue
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(image_path)
            file_reader.ReadImageInformation()
            extract_size = list(file_reader.GetSize())
            extract_size[2]=1
            current_index = [0] * file_reader.GetDimension()
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(image_path)
            file_reader.SetExtractIndex(current_index)
            file_reader.SetExtractSize(extract_size)
            mask_obj = file_reader.Execute()
            path_list.append({'image_path':image_path})
            print(image_path)
        except:
            traceback.print_exc()
            print('err')
        

    df = pd.DataFrame(path_list)
    df.to_csv(NIFTI_FILE,index=False)

def prepare_dataset(func=parse_fn):
    if not os.path.exists(NIFTI_FILE):
        cache_file_paths()
    df = pd.read_csv(NIFTI_FILE)
    path_list = df.image_path.tolist()

    norm_filenames = tf.constant(path_list[:100])
    norm_ds = tf.data.Dataset.from_tensor_slices(norm_filenames).repeat(1).shuffle(10 * batch_size).map(
        func, num_parallel_calls=tf.data.AUTOTUNE
    )
    norm_ds = norm_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    train_filenames = tf.constant(path_list[:-900])
    train_ds = tf.data.Dataset.from_tensor_slices(train_filenames).repeat(dataset_repetitions).shuffle(10 * batch_size).map(
        func, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_filenames = tf.constant(path_list[-900:])
    val_ds = tf.data.Dataset.from_tensor_slices(val_filenames).repeat(dataset_repetitions).shuffle(10 * batch_size).map(
        func, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    return norm_ds, train_ds, val_ds

if __name__ == "__main__":


    norm_dataset, train_dataset , val_dataset = prepare_dataset()

    plt.figure(figsize=(10, 10))
    for images,labels in val_dataset.take(1):
        print(images.shape,labels.shape)
        for i in range(batch_size):
            ax = plt.subplot(3, 3, i + 1)
            tmp_image = images[i,:].numpy()
            tmp_label = labels[i,:].numpy()
            tmp = np.concatenate([tmp_image,tmp_label],axis=1)
            plt.imshow(tmp,cmap='gray')
            plt.axis("off")
            if i > 7 :
                break
        os.makedirs(TMP_DIR,exist_ok=True)
        plt.savefig(f"{TMP_DIR}/test.png")
        plt.close()
