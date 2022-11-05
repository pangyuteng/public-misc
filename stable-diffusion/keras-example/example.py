# https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt


model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

# def main():
# if __name__ == "__main__":
#     main()
