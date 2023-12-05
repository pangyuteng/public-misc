#!/bin/bash
echo $1 $2

export CODE_FOLDER=$1
export TFDS_DATA_DIR=$2

cd $CODE_FOLDER

python3 cityscapes.py
python3 ddim.py