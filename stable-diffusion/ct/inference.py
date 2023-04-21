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


_, _ , deeplesion_val_dataset = prepare_deeplesion_dataset()
for images in deeplesion_val_dataset.take(1):
    encoded_outputs = encoder.predict(images)
print('encoded_outputs',encoded_outputs.shape)
# 16, 128, 128, 3


for images,labels in val_dataset.take(1):
    print(images.shape,labels.shape)

    generated_latent = diffusion_model.generate(
        num_images=batch_size,
        diffusion_steps=plot_diffusion_steps,
        labels=labels
    )

    print(generated_latent.shape)

    # latent to image
    priors  = generated_latent.numpy() * NUM_EMBEDDINGS
    priors = priors.astype(np.int32)

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )

    quantized = tf.reshape(quantized, (-1, 128, 128, 3 ))
    # Generate novel images.
    decoder = vqvae_trainer.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)
    reconstructions_images = decoder.predict(generated_samples)

    print('reconstructions_images',reconstructions_images.shape)

    plt.figure(figsize=(4,4))
    for i in range(batch_size):
        ax = plt.subplot(4,4, i + 1)
        tmp_image = reconstructions_images[i,:]
        plt.imshow(tmp_image,cmap='gray')
        plt.axis("off")
        if i > 7 :
            break
    os.makedirs(TMP_DIR,exist_ok=True)
    plt.savefig(f"{TMP_DIR}/z.png")
    plt.close()

    print('done')

