import keras
from ddim import DiffusionModel,checkpoint_path

model = DiffusionModel(image_size, widths, block_depth)
model.network.summary()

model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
    run_eagerly=True
)
# load the best model and generate images
model.load_weights(checkpoint_path)
model.plot_images(epoch=1000)
