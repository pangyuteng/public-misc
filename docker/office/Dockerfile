
FROM python:3.10-bullseye

WORKDIR /opt

RUN python3 -m pip install --upgrade pip
COPY requirements.txt /opt
RUN pip install -r /opt/requirements.txt

ENV MPLCONFIGDIR=/opt/.config
RUN mkdir -p ${MPLCONFIGDIR} && chmod -R 777 ${MPLCONFIGDIR}
