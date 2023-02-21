FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

# cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt 

WORKDIR /opt/
RUN git clone https://github.com/fchollet/stable-diffusion-tensorflow
WORKDIR /opt/stable-diffusion-tensorflow
RUN pip install -r requirements.txt
RUN python setup.py install