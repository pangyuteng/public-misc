FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

# cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY warmup.py /tmp
RUN python /tmp/warmup.py