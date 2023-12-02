#!/bin/bash

export TFDS_DATA_DIR=/radraid/pteng-public/tmp/tensorflow_datasets
export CHECKPOINT_PATH=/cvibraid/cvib2/apps/personal/pteng/github/aigonewrong/stable-diffusion/semantic-synthesis/cityscapes
export TEMP_PATH=/cvibraid/cvib2/apps/personal/pteng/github/aigonewrong/stable-diffusion/semantic-synthesis/cityscapes/tmp
#python3 cityscapes.py
python3 ddim.py