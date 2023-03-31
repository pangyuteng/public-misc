

'''

docker run --runtime=nvidia -it -u $(id -u):$(id -g) -w $PWD -v /mnt:/mnt -v /mnt/scratch/tmp/tensorflow_datasets:/tensorflow_datasets keras-stable-diffusion bash

CUDA_VISIBLE_DEVICES=1 python vqvae.py


'''