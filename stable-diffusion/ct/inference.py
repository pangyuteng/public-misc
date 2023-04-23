import os
import sys
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# TODO: noise to mask

# noise+mask to latent
from diffusion_image import (
    get_diffusion_model, my_inference_func, prepare_dataset, batch_size
)
plot_diffusion_steps = 20

diffusion_model = get_diffusion_model()
norm_dataset, train_dataset , val_dataset = prepare_dataset()
val_dataset = val_dataset.map(my_inference_func, num_parallel_calls=tf.data.AUTOTUNE)

from vqvae import get_vqvae_model, TMP_DIR

vqvae_trainer = get_vqvae_model()
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
decoder = vqvae_trainer.vqvae.get_layer("decoder")


for images,labels,encoded_outputs,sm_labels in val_dataset.take(1):
    print('images',images.shape,'labels',labels.shape)

    encoded_outputs = diffusion_model.generate(
        num_images=batch_size,
        diffusion_steps=plot_diffusion_steps,
        labels=sm_labels
    )

    generated_samples = decoder.predict(encoded_outputs)
    print('generated_samples',generated_samples.shape)

    plt.figure(figsize=(4, 4))
    for i in range(batch_size):
        ax = plt.subplot(4, 4, i + 1)
        tmp_image = images[i,:].numpy()
        tmp_label = labels[i,:].numpy()
        tmp_gen = generated_samples[i,:]

        plt.imshow(tmp_gen,cmap='gray')
        plt.axis("off")
        
        tmp_image = (tmp_image-np.min(tmp_image))/(np.max(tmp_image)-np.min(tmp_image))
        tmp_image = (tmp_image*255).clip(0,255)

        tmp_label = (tmp_label*255).clip(0,255)
        
        tmp_gen = (tmp_gen-np.min(tmp_gen))/(np.max(tmp_gen)-np.min(tmp_gen))
        tmp_gen = (tmp_gen*255).clip(0,255)

        print(tmp_image.shape,np.min(tmp_image),np.max(tmp_image))
        print(tmp_label.shape,np.min(tmp_label),np.max(tmp_label))
        print(tmp_gen.shape,np.min(tmp_gen),np.max(tmp_gen))
        print('--')

        tmp_merged = np.concatenate([tmp_image,tmp_label,tmp_gen],axis=1)
        imageio.imwrite(f"{TMP_DIR}/z-{i}.png",tmp_merged)
    os.makedirs(TMP_DIR,exist_ok=True)
    plt.savefig(f"{TMP_DIR}/z.png")
    plt.close()

    print('done')

