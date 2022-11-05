# https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion

import time
import keras_cv
from tensorflow import keras
import imageio

BATCH_SIZE = 5
text = """
artificial intelligence gone wrong, watercolor
"""
text = text.replace("\n","")
print(text)

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
images = model.text_to_image(text,batch_size=BATCH_SIZE)
for x in range(BATCH_SIZE):
    imageio.imwrite(f'sample_{x:05d}.png',images[x,:])

