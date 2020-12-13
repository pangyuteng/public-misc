FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
