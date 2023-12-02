

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /mnt:/mnt -v /mnt/scratch/tmp/tensorflow_datasets:/tensorflow_datasets keras-stable-diffusion bash


mkdir -p /radraid/pteng-public/tmp/tensorflow_datasets/downloads/manual

docker run -it -u $(id -u):$(id -g) \
    -e 
    -w $PWD -v /radraid:/radraid -v /cvibraid:/cvibraid \
    pangyuteng/keras-stable-diffusion:latest bash

https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch