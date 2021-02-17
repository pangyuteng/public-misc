FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt --use-feature=2020-resolver
