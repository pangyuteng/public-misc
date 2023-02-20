FROM tensorflow/tensorflow:2.9.1-gpu-jupyter 

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt 

WORKDIR /opt/
RUN git clone https://github.com/fchollet/stable-diffusion-tensorflow
WORKDIR /opt/stable-diffusion-tensorflow

# cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt
RUN python setup.py install
