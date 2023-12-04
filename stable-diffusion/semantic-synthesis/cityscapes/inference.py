from tensorflow import keras
from ddim import (
    DiffusionModel,checkpoint_path,
    image_size, widths, block_depth,
    weight_decay,learning_rate,
    network_weight_file,
    network_ema_weight_file,
    val_dataset,
    num_rows,num_cols,
    plot_diffusion_steps,
    tmp_folder
)

model = DiffusionModel(image_size, widths, block_depth)
model.normalizer.adapt(val_dataset.map(lambda images, labels: images))

#model.network.summary()
# model.compile(
#     optimizer=keras.optimizers.experimental.AdamW(
#         learning_rate=learning_rate, weight_decay=weight_decay
#     ),
#     loss=keras.losses.mean_absolute_error,
#     run_eagerly=True
# )
# load the best model and generate images
#model.load_weights(checkpoint_path)

model.network.load_weights(network_weight_file)
model.ema_network.load_weights(network_ema_weight_file)

for images,labels in val_dataset.take(1):

    generated_images = model.generate(
        num_images=num_rows * num_cols,
        diffusion_steps=plot_diffusion_steps,
        labels=labels
    )
    print(generated_images.shape)
    break
        
    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            tmp_image = generated_images[index].numpy()
            tmp_label = labels[index,:]
            tmp_label = np.repeat(tmp_label,3, axis=2)
            tmp = np.concatenate([tmp_image,tmp_label],axis=1)
            plt.imshow(tmp)
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    os.makedirs('tmp',exist_ok=True)
    plt.savefig(f"{tmp_folder}/{epoch:05d}.png")
    plt.close()
