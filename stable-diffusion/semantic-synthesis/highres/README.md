

docker run -it -u $(id -u):$(id -g) -w $PWD -v /mnt:/mnt -v /mnt/scratch/tmp/tensorflow_datasets:/tensorflow_datasets keras-stable-diffusion bash

https://keras.io/examples/generative/vq_vae/

# train the encode decode
vqvae.py

# generate the encoder, and train diffusion
diffusion.py





