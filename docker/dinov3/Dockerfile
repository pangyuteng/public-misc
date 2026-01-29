FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

WORKDIR /opt
RUN apt-get update && apt-get install vim git curl -yq
RUN git clone https://github.com/facebookresearch/dinov3.git
WORKDIR /opt/dinov3
RUN python setup.py install
RUN pip install "transformers[torch]"

ENV HF_HOME=/opt/.huggingface
RUN mkdir -p /opt/.huggingface && chmod 777 -R /opt/.huggingface
COPY cache_weights.py /opt
RUN --mount=type=secret,id=hf,env=HF_TOKEN python /opt/cache_weights.py

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
