## setup environment with docker, and run notebook

```
docker build -t tasty .
docker run -it -v ${PWD}:/opt -w /opt -p 8889:8888 tasty bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
```
gpu
```
docker run -it --gpus device=0 -v ${PWD}:/opt -w /opt -p 8889:8888 tasty bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
```
### convience jupyter lab up with docker-copmose

```
doocker-compose up --build
```




