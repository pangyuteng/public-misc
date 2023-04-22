#!/usr/local/bin/python

"""
https://keras.io/examples/generative/ddim/
https://github.com/keras-team/keras-io/blob/master/examples/generative/ddim.py
20a32b18c0b95e914101ec226ef35af9fdac3970
"""
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

from gen_totalseg import (
    batch_size,
    prepare_dataset
)

TMP_DIR = 'tmp'
checkpoint_path = "checkpoints/diffusion_model_image"


# data
image_size = 64

num_epochs = 500  # train for at least 50 epochs for good results

num_cols = 4
num_rows = 4
widths = [32, 64, 96, 128]

# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
block_depth = 2

# optimization
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


"""
## Kernel inception distance

"""


class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        batch_size = tf.shape(real_images)[0]
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, l, skips = x
        for _ in range(block_depth):
            x = layers.Concatenate()([x,l])
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        l = layers.AveragePooling2D(pool_size=2)(l)
        return x,l

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, l, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    labels = keras.Input(shape=(image_size, image_size, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    l = layers.Conv2D(1, kernel_size=1)(labels)
    skips = []
    for width in widths[:-1]:
        x,l = DownBlock(width, block_depth)([x, l, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, l, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, labels], x, name="residual_unet")


class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, labels, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2, labels], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, labels):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, labels, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps, labels):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, labels)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, input_data):
        images,labels = input_data
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, labels, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, input_data):
        images,labels = input_data

        self._images = images.numpy()
        self._labels = labels.numpy()
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        
        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, labels, training=False
        )
        
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps,labels=labels
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=num_rows, num_cols=num_cols):
        # plot random generated images for visual evaluation of generation quality
        
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
            labels=self._labels
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                tmp_x = self._images[index,:]
                tmp_label = self._labels[index,:]
                tmp_label = np.tile(tmp_label, [1,1,3])
                tmp_xhat = generated_images[index].numpy()
                tmp = np.concatenate([tmp_x,tmp_label,tmp_xhat],axis=1)
                plt.imshow(tmp,cmap='gray')
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        os.makedirs(TMP_DIR,exist_ok=True)
        plt.savefig(f"{TMP_DIR}/image-{epoch:05d}.png")
        plt.close()
        if epoch % 20 == 0:
            self.network.save_weights(f'{TMP_DIR}/network_image_{epoch}.h5')
            self.ema_network.save_weights(f'{TMP_DIR}/ema_network_image_{epoch}.h5')


"""
## Training
"""

from vqvae import get_vqvae_model

vqvae_trainer = get_vqvae_model()
encoder = vqvae_trainer.vqvae.get_layer("encoder")
quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

def myfunc(x,y):
    encoded_outputs = encoder(x)
    y = tf.image.resize(y, [image_size,image_size],method='nearest',antialias=False)
    return encoded_outputs, y

def my_inference_func(x,y):
    encoded_outputs = encoder(x)
    sy = tf.image.resize(y, [image_size,image_size],method='nearest',antialias=False)
    return x,y, encoded_outputs, sy

def get_diffusion_model():
    norm_dataset, train_dataset , val_dataset = prepare_dataset()
    norm_dataset = norm_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)

    model = DiffusionModel(image_size, widths, block_depth)
    model.compile(
        optimizer=keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
        run_eagerly=True
    )
    model.normalizer.adapt(norm_dataset.map(lambda images, labels: images))
    model.load_weights(checkpoint_path)
    return model

if __name__ == "__main__":

    norm_dataset, train_dataset , val_dataset = prepare_dataset()

    train_dataset = train_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)
    norm_dataset = norm_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)

    plt.figure(figsize=(10, 10))
    for images,labels in val_dataset.take(1):
        print(images.shape,labels.shape)
        for i in range(batch_size):
            ax = plt.subplot(3, 3, i + 1)
            tmp_image = images[i,:].numpy()+0.5
            tmp_label = labels[i,:].numpy()
            tmp_label= np.tile(tmp_label,(1,1,3))
            print(tmp_image.shape,tmp_label.shape)
            print('tmp_image',np.min(tmp_image),np.max(tmp_image))
            print('tmp_label',np.min(tmp_label),np.max(tmp_label))
            tmp = np.concatenate([tmp_image,tmp_label],axis=1)
            plt.imshow(tmp,cmap='gray')
            plt.axis("off")
            if i > 7 :
                break
        os.makedirs(TMP_DIR,exist_ok=True)
        plt.savefig(f"{TMP_DIR}/codebook-seg.png")
        plt.close()

    model = DiffusionModel(image_size, widths, block_depth)
    model.network.summary()

    model.compile(
        optimizer=keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
        run_eagerly=True
    )
    # pixelwise mean absolute error is used as loss

    # save the best model based on the validation KID metric
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(norm_dataset.map(lambda images, labels: images))

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=50,
        validation_steps=10,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            checkpoint_callback,
        ],
    )

    """
    ## Inference
    """

    # load the best model and generate images
    model.load_weights(checkpoint_path)
    model.plot_images(epoch=999)
