FROM python:3.10-bullseye

RUN apt-get update \
    && apt-get install -yq \
    ffmpeg libsm6 libxext6 build-essential make vim curl wget \
    wkhtmltopdf

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install vtk-osmesa --extra-index-url https://gitlab.kitware.com/api/v4/projects/13/packages/pypi/simple

WORKDIR /opt
RUN git clone https://github.com/neurolabusc/nii2mesh
RUN cd nii2mesh/src && make && mv /opt/nii2mesh/src/nii2mesh /usr/bin
