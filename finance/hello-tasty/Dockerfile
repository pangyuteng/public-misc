FROM python:3.8-buster

RUN apt-get update -y; apt-get install -y curl wget vim

# Copy the package requirements
COPY requirements.txt /opt

WORKDIR /opt
# Install the package requirements
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . /opt/app

WORKDIR /opt/app
CMD tail -f /dev/null