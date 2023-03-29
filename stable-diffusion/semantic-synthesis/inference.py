import sys
from tensorflow import keras
from totalseg import (
    DiffusionModel, checkpoint_path,
    image_size, widths, block_depth,
    learning_rate, weight_decay,
    prepare_dataset, batch_size,
    TMP_DIR,
)
if __name__ == "__main__":

    epoch = sys.argv[1]
    
    norm_dataset, train_dataset , val_dataset = prepare_dataset()
    # load the best model and generate images
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
    model.network.load_weights(f'{TMP_DIR}/network_{epoch}.h5')
    model.ema_network.load_weights(f'{TMP_DIR}/ema_network_{epoch}.h5')

    for images,labels in val_dataset.take(1):
        model._labels = labels.numpy()
        model.plot_images(epoch=999,num_rows=4,num_cols=4)
