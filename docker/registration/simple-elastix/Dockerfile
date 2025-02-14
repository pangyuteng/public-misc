FROM python:3.8-bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
  wget \
  curl \
  vim

WORKDIR /opt
COPY requirements.txt /opt
RUN pip install -r requirements.txt

