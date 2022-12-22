# https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion

import time
import keras_cv
from tensorflow import keras
import imageio
import sys
import os

text = sys.argv[1]
folder = sys.argv[2]

os.makedirs(folder,exist_ok=True)
print(text)
print(folder)

BATCH_SIZE = 5
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
for x in range(BATCH_SIZE):
    images = model.text_to_image(text,batch_size=1)
    file_path = os.path.join(folder,f'sample_{x:05d}.png')
    imageio.imwrite(file_path,images[0,:])

