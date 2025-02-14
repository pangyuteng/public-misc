FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        ca-certificates \
        cmake \
        git \
        wget \
        curl \
        vim
RUN rm -rf /var/lib/apt/lists/*

# install elastix
RUN mkdir -p /opt/sources
WORKDIR /opt/sources
RUN mkdir -p /opt/sources/elastix
RUN wget --quiet --no-check-certificate https://github.com/SuperElastix/elastix/releases/download/4.9.0/elastix-4.9.0-linux.tar.bz2 -O elastix.tar.bz2 && \
    tar xjf elastix.tar.bz2 -C elastix && \
    rm /opt/sources/elastix.tar.bz2

ENV ELASTIX_LIBRARYDIR /opt/sources/elastix/lib
ENV LD_LIBRARY_PATH $ELASTIX_LIBRARYDIR:$LD_LIBRARY_PATH
ENV ELASTIX_PATH /opt/sources/elastix/bin
ENV PATH $ELASTIX_PATH:$PATH

RUN pip install pandas==1.1.5 pydicom==2.3.0 SimpleITK==2.1.1
