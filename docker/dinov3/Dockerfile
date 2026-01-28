FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

WORKDIR /opt
RUN apt-get update && apt-get install vim git curl -yq
RUN git clone https://github.com/facebookresearch/dinov3.git
