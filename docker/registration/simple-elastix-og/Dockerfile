FROM python:3.11-bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
  autotools-dev \
  build-essential \
  ca-certificates \
  cmake \
  git \
  wget \
  curl \
  vim
# RUN rm -rf /var/lib/apt/lists/*

# build SimpleElastix
RUN mkdir -p /opt/sources
WORKDIR /opt/sources
RUN git clone https://github.com/SuperElastix/SimpleElastix ; cd SimpleElastix ; git checkout 1eec022 

WORKDIR /opt/sources/SimpleElastix
RUN mkdir /build \
  && cd /build \
  && cd /build;cmake -D PYTHON_EXECUTABLE=`which python3` /opt/sources/SimpleElastix/SuperBuild \
  && cd /build;make -j 4 \  
  && cd /build/SimpleITK-build/Wrapping/Python;python3 Packaging/setup.py install \
  && rm -rf /build && rm -rf /opt/sources

#&& cd /build;make -j `nproc` \

# WORKDIR /opt
# COPY requirements.txt /opt
# RUN pip install -r requirements.txt

