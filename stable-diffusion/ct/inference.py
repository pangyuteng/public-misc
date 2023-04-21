import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# TODO: noise to mask

# noise+mask to latent
from diffusion_image import (
    get_diffusion_model, myfunc, prepare_dataset, batch_size,
    NUM_EMBEDDINGS
)
plot_diffusion_steps = 20

diffusion_model = get_diffusion_model()
norm_dataset, train_dataset , val_dataset = prepare_dataset()
val_dataset = val_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)

from gen_deeplesion import prepare_dataset as prepare_deeplesion_dataset

from vqvae import get_vqvae_model, TMP_DIR

vqvae_trainer = get_vqvae_model()
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
decoder = vqvae_trainer.vqvae.get_layer("decoder")


for images,labels in val_dataset.take(1):
    print(images.shape,labels.shape)

    encoded_outputs = diffusion_model.generate(
        num_images=batch_size,
        diffusion_steps=plot_diffusion_steps,
        labels=labels
    )

    generated_samples = decoder.predict(encoded_outputs)
    print('generated_samples',generated_samples.shape)

    plt.figure(figsize=(4, 4))
    for i in range(batch_size):
        ax = plt.subplot(4, 4, i + 1)
        tmp_image = generated_samples[i,:]
        plt.imshow(tmp_image,cmap='gray')
        plt.axis("off")
        print(tmp_image.shape,np.min(tmp_image),np.max(tmp_image))
    os.makedirs(TMP_DIR,exist_ok=True)
    plt.savefig(f"{TMP_DIR}/z.png")
    plt.close()

    print('done')

