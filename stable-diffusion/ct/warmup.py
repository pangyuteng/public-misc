
from tensorflow import keras

image_size = 128
ok = keras.applications.InceptionV3(
    include_top=False,
    input_shape=(image_size, image_size, 3),
    weights="imagenet",
)
