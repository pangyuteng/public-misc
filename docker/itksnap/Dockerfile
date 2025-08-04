FROM ubuntu:24.04
RUN apt-get update && apt-get upgrade -yq && apt-get install curl wget vim -yq

RUN	apt-get update && apt-get install -yq \
    vim curl \
    libc6 libxv1 libglu1-mesa xauth x11-utils xorg \
    libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    
RUN mkdir /opt/itksnap
COPY itksnap-4.4.0-alpha4-20250730-Linux-x86_64.tar.gz /tmp
RUN tar -xvf /tmp/itksnap-4.4.0-alpha4-20250730-Linux-x86_64.tar.gz -C /opt/itksnap --strip-components=1 \
    && rm /tmp/itksnap-4.4.0-alpha4-20250730-Linux-x86_64.tar.gz

#LD_LIBRARY_PATH
#PATH
## libcurl3