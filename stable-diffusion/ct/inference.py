import tensorflow as tf

# TODO: noise to mask

# noise+mask to latent
from diffusion_image import get_diffusion_model, myfunc, prepare_dataset, batch_size
plot_diffusion_steps = 20

diffusion_model = get_diffusion_model()
norm_dataset, train_dataset , val_dataset = prepare_dataset()
val_dataset = val_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)

for images,labels in val_dataset.take(1):
    print(images.shape,labels.shape)
    generated_latent = diffusion_model.generate(
        num_images=batch_size,
        diffusion_steps=plot_diffusion_steps,
        labels=labels
    )
print(generated_latent.shape)

# latent to image

from vqvae import get_vqvae_model

vqvae_trainer = get_vqvae_model()

decoder = vqvae_trainer.vqvae.get_layer("decoder")
reconstructions_images = decoder.predict(quantized)

print('reconstructions_images',reconstructions_images.shape)
print('done')

