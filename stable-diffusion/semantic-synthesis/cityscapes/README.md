

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /mnt:/mnt keras-stable-diffusion bash

https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch