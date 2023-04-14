#!/bin/bash
echo $1
cd /mnt/hd1/aigonewrong/stable-diffusion/ct
if [ "$1" = "vqvae" ]; then
    python vqvae.py
else
    python diffusion.py
fi
