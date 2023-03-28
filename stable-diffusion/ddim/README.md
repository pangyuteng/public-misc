
```
docker run --runtime=nvidia -it -w $PWD -v /mnt:/mnt keras-stable-diffusion bash

# NOTE0. no `-u` flag since flower dataset needs `/` write perm
# NOTE1. set hard memory - no balloning device if you are using VM!


docker run --runtime=nvidia -it -u $(id -u):$(id -g) -w $PWD -v /mnt:/mnt -v /mnt/scratch/tmp/tensorflow_datasets:/tensorflow_datasets keras-stable-diffusion bash
CUDA_VISIBLE_DEVICES=0 python cityscapes.py

```
