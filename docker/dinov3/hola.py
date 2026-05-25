import os
import torch
import numpy as np

from transformers import pipeline
from transformers.image_utils import load_image
"""
wget https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Felis_silvestris_silvestris_small_gradual_decrease_of_quality_-_JPEG_compression.jpg/250px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality_-_JPEG_compression.jpg
"""

image = load_image("cat.jpg")

feature_extractor = pipeline(
    model="/opt/.huggingface/hub/models--facebook--dinov3-convnext-tiny-pretrain-lvd1689m/snapshots/10d30274b4d445111e2d5bf75ac93bbd94db274b",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
features = np.array(features)
print(features.shape)

"""

docker run  -it -u $(id -u):$(id -g) --env-file=.env \
--shm-size=10g --gpus "device=0" \
-v /mnt:/mnt -w $PWD \
docker.io/pangyuteng/dinov3 bash 


"""
