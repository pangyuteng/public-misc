#!/bin/bash

docker build -t pangyuteng/dcm:latest .
docker push pangyuteng/dcm:latest

#docker build -t pangyuteng/dcm:minimal -f Dockerfile.minimal .
#docker push pangyuteng/dcm:minimal
