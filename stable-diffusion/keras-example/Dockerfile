#FROM tensorflow/tensorflow:2.9.1-gpu-jupyter 
#COPY requirements.txt .
#RUN pip install -r requirements.txt 


FROM tensorflow/tensorflow:2.10.0

RUN apt-get update && apt-get install git -yq
RUN python -m pip install --upgrade pip

WORKDIR /opt/
RUN git clone https://github.com/fchollet/stable-diffusion-tensorflow
WORKDIR /opt/stable-diffusion-tensorflow
RUN pip install -r requirements.txt
RUN python setup.py install

RUN pip uninstall tensorboard -y

COPY download_weights.py .
RUN python download_weights.py
