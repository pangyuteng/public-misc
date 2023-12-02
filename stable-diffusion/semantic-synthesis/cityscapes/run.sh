#!/bin/bash

export TFDS_DATA_DIR=/radraid/pteng-public/tmp/tensorflow_datasets

cd /cvibraid/cvib2/apps/personal/pteng/github/aigonewrong/stable-diffusion/semantic-synthesis/cityscapes

python3 cityscapes.py
python3 ddim.py