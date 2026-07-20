FROM tensorflow/tensorflow:2.20.0-gpu-jupyter


RUN apt update && apt install libosmesa6 libglu1-mesa libglu1-mesa-dev -yq
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# per matplotlib warning adding below, and cache .json file.
ENV MPLCONFIGDIR=/opt/.config/matplotlib
RUN mkdir -p $MPLCONFIGDIR && chmod -R 777 /opt/.config && python -c "import matplotlib.pyplot"
