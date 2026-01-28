import os
import torch

from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
print(features.shape)

"""

docker run  -it -u $(id -u):$(id -g) --env-file=.env \
--shm-size=10g --gpus "device=0" \
-v /mnt:/mnt -w $PWD \
docker.io/pangyuteng/dinov3 bash 


"""