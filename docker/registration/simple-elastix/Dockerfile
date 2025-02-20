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

WORKDIR /opt
COPY requirements.txt /opt
RUN pip install -r requirements.txt


# #
# # v2.4.1 cf120d6
# # https://github.com/SimpleITK/SimpleITK/releases/tag/v2.4.1
# #
# RUN git clone https://github.com/SimpleITK/SimpleITK.git ; cd SimpleITK ; git checkout cf120d6
# WORKDIR /opt/sources
# RUN mkdir SimpleITK-build \
#     && cd SimpleITK-build \
#     && cmake -D SimpleITK_USE_ELASTIX=ON ../SimpleITK/SuperBuild \
#     && cd /build;make -j 4

# make -j `nproc` 
