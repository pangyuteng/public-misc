import sys
import numpy as np
from skimage.transform import resize
import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from totalseg import (
    DiffusionModel, checkpoint_path,
    image_size, widths, block_depth,
    learning_rate, weight_decay,
    prepare_dataset, batch_size,
    label_count, TMP_DIR,
)
if __name__ == "__main__":

    epoch = sys.argv[1]
    label_file = sys.argv[2]
    output_file = sys.argv[3]

    plot_diffusion_steps = 20

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

    label_obj = sitk.ReadImage(label_file)
    labels = sitk.GetArrayFromImage(label_obj)
    labels = labels.astype(np.float32)
    original_shape = list(labels.shape)
    new_shape = [original_shape[0],image_size,image_size]
    labels = resize(labels,new_shape,order=0,preserve_range=True, anti_aliasing=False)
    labels = labels/label_count

    xhat = np.zeros_like(labels)
    num_images = labels.shape[0]
    initial_noise = tf.random.normal(shape=(1, image_size, image_size, 1))
    for x in range(num_images):
        tmp_label = np.expand_dims(np.expand_dims(labels[x,:],axis=-1),axis=0)
        generated_images = model.reverse_diffusion(initial_noise, plot_diffusion_steps, tmp_label)
        generated_images = model.denormalize(generated_images)
        xhat[x,:] = generated_images.numpy().squeeze()
    
    xhat = ((xhat*2000)-1000).astype(np.int16)
    xhat_obj =  sitk.GetImageFromArray(xhat)
    xhat_obj.CopyInformation(label_obj)
    sitk.WriteImage(xhat_obj,output_file)

"""

"""