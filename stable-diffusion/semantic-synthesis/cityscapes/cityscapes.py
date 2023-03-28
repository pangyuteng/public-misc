"""

https://www.tensorflow.org/datasets/catalog/cityscapes

https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L52-L99

"""


import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

builder = tfds.builder('cityscapes')
builder.download_and_prepare()

ds = builder.as_dataset(split='train', shuffle_files=True)

ds = ds.take(20)  # Only take a single example

for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    image_id = example["image_id"].numpy().decode("utf-8")
    print(image_id,type(image_id))
    image = example["image_left"].numpy()
    label = example["segmentation_label"].numpy()
    print(image.shape, label.shape)
    print(np.unique(label))
    imageio.imwrite(f'{image_id}_image_left.png',image)
    imageio.imwrite(f'{image_id}_segmentation_label.png',label)