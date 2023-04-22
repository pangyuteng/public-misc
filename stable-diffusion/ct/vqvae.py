"""

https://keras.io/examples/generative/vq_vae
https://github.com/keras-team/keras-io/blob/master/examples/generative/vq_vae.py

https://raw.githubusercontent.com/keras-team/keras-io/master/examples/generative/vq_vae.py

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

TMP_DIR = 'tmp'
os.makedirs(TMP_DIR,exist_ok=True)

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

from gen_deeplesion import prepare_dataset, batch_size, image_size

LATENT_DIM = 3
NUM_EMBEDDINGS = 2048
CODEBOOK_WH = image_size//4

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

block_depth = 2
def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(image_size, image_size, 1))
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=2,attention_axes=(1,2))(x,x)
    for _ in range(block_depth):
        x = ResidualBlock(128)(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = latent_inputs
    for _ in range(block_depth):
        x = ResidualBlock(128)(x)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=2,attention_axes=(1,2))(x,x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def get_vqvae(latent_dim,num_embeddings):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(image_size, image_size, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim, num_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_vq_loss_tracker = keras.metrics.Mean(name="val_vq_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def test_step(self, x):
        reconstructions = self.vqvae(x)

        # Calculate the losses.
        reconstruction_loss = (
            tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
        )
        total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Loss tracking.
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "vqvae_loss": self.val_vq_loss_tracker.result(),
        }

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


"""
## Train the VQ-VAE model
"""


class MyModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,):
        super().__init__()
    def on_epoch_begin(self, epoch, logs=None):
        vqvae_weights_file = f'{TMP_DIR}/vqvae-{epoch}.h5'
        self.model.vqvae.save_weights(vqvae_weights_file)

vqvae_weights_file = f'{TMP_DIR}/vqvae.h5'

def get_vqvae_model():

    norm_dataset, train_dataset , val_dataset = prepare_dataset()

    normalizer = layers.Normalization()
    normalizer.adapt(norm_dataset.map(lambda images: images))
    data_variance = normalizer.variance

    vqvae_trainer = VQVAETrainer(data_variance, LATENT_DIM, NUM_EMBEDDINGS)
    vqvae_trainer.vqvae.load_weights(vqvae_weights_file)
    return vqvae_trainer

if __name__ == "__main__":

    norm_dataset, train_dataset , val_dataset = prepare_dataset()

    normalizer = layers.Normalization()
    normalizer.adapt(norm_dataset.map(lambda images: images))
    data_variance = normalizer.variance

    # # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    vqvae_model_checkpoint_callback = MyModelCheckpoint()
    learning_rate = 1e-4
    vqvae_trainer = VQVAETrainer(data_variance, LATENT_DIM, NUM_EMBEDDINGS)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate))

    epochs = 200
    if not os.path.exists(vqvae_weights_file):
        vqvae_trainer.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[vqvae_model_checkpoint_callback],
            validation_data=val_dataset,
        )
        vqvae_trainer.vqvae.save_weights(vqvae_weights_file)
    else:
        vqvae_trainer.vqvae.load_weights(vqvae_weights_file)
    """
    ## Reconstruction results on the test set
    """

    def show_subplot(original, reconstructed,idx):
        plt.subplot(1, 2, 1)
        plt.imshow(original.squeeze() + 0.5 ,cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed.squeeze() + 0.5 ,cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")

        plt.show()
        plt.savefig(f"{TMP_DIR}/recon-{idx}.png")
        plt.close()

    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(val_dataset), 10)
    test_images = [x for x in val_dataset.take(1)][0]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for idx in range(batch_size):
        test_image = test_images.numpy()[idx,:]
        reconstructed_image = reconstructions_test[idx,:]
        show_subplot(test_image, reconstructed_image,idx)

    """
    These results look decent. You are encouraged to play with different hyperparameters
    (especially the number of embeddings and the dimensions of the embeddings) and observe how
    they affect the results.
    """

    """
    ## Visualizing the discrete codes
    """
    print('done with vae training')
    sys.exit(0)
    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images.numpy()[i].squeeze() + 0.5,cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i],cmap='gray')
        plt.title("Code")
        plt.axis("off")
        plt.show()
        plt.savefig(f"{TMP_DIR}/code-{i}.png")
        plt.close()
    """
    ## PixelCNN hyperparameters
    """

    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    # 16,16

    # The first layer is the PixelCNN layer. This layer simply
    # builds on the 2D convolutional layer, but includes masking.
    class PixelConvLayer(layers.Layer):
        def __init__(self, mask_type, **kwargs):
            super().__init__()
            self.mask_type = mask_type
            self.conv = layers.Conv2D(**kwargs)

        def build(self, input_shape):
            # Build the conv2d layer to initialize kernel variables
            self.conv.build(input_shape)
            # Use the initialized kernel to create the mask
            kernel_shape = self.conv.kernel.get_shape()
            self.mask = np.zeros(shape=kernel_shape)
            self.mask[: kernel_shape[0] // 2, ...] = 1.0
            self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
            if self.mask_type == "B":
                self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

        def call(self, inputs):
            self.conv.kernel.assign(self.conv.kernel * self.mask)
            return self.conv(inputs)


    # Next, we build our residual block layer.
    # This is just a normal residual block, but based on the PixelConvLayer.
    class ResidualBlock(keras.layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.conv1 = keras.layers.Conv2D(
                filters=filters, kernel_size=1, activation="relu"
            )
            self.pixel_conv = PixelConvLayer(
                mask_type="B",
                filters=filters // 2,
                kernel_size=3,
                activation="relu",
                padding="same",
            )
            self.conv2 = keras.layers.Conv2D(
                filters=filters, kernel_size=1, activation="relu"
            )

        def call(self, inputs):
            x = self.conv1(inputs)
            x = self.pixel_conv(x)
            x = self.conv2(x)
            return keras.layers.add([inputs, x])


    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=64, kernel_size=7, activation="relu", padding="same"
    )(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=64)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=64,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = keras.layers.Conv2D(
        filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    pixel_cnn.summary()

    # Generate the codebook indices.

    def myfunc(x):
        #tf.print(tf.shape(x), output_stream=sys.stderr) # None,64,64,1
        encoded_outputs = encoder(x)
        #tf.print(tf.shape(encoded_outputs), output_stream=sys.stderr) # None,16,16,8
        flat_enc_outputs = tf.reshape(encoded_outputs, [-1, LATENT_DIM])
        codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
        codebook_indices = tf.reshape(codebook_indices, [-1, CODEBOOK_WH,CODEBOOK_WH])
        #tf.print(tf.shape(codebook_indices), output_stream=sys.stderr) # None,16,16
        return codebook_indices,codebook_indices

    codebook_indices = train_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)
    for x,y in codebook_indices.take(1):
        print(x.shape,y.shape)

    val_codebook_indices = val_dataset.map(myfunc, num_parallel_calls=tf.data.AUTOTUNE)

    """
    ## PixelCNN training
    """
    pixelcnn_checkpoint_path = "./checkpoint_pixelcnn"
    pixelcnn_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        pixelcnn_checkpoint_path,
        monitor="val_loss"
        )

    epochs = 5
    pixel_cnn.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    pixel_cnn_weight_file = f'{TMP_DIR}/pixel_cnn.h5'
    if not os.path.exists(pixel_cnn_weight_file):
        pixel_cnn.fit(
            codebook_indices,
            epochs=epochs,
            callbacks=[pixelcnn_model_checkpoint_callback],
            validation_data=val_codebook_indices,
        )
        pixel_cnn.save_weights(pixel_cnn_weight_file)
    else:
        pixel_cnn.load_weights(pixel_cnn_weight_file)

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    """
    We now construct a prior to generate images. Here, we will generate 10 images.
    """

    # Create an empty array of priors.
    batch = 10
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]
            print(f'{row} of {rows},{col} of {cols}')
    print(f"Prior shape: {priors.shape}")

    """
    We can now use our decoder to generate the images.
    """

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))
    # Generate novel images.
    decoder = vqvae_trainer.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i],cmap='gray')
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5,cmap='gray')
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()
        plt.savefig(f"{TMP_DIR}/generated-{i}.png")