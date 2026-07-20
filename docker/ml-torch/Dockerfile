FROM huggingface/transformers-pytorch-gpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN adduser --uid 1000 --disabled-password ubuntu
USER ubuntu