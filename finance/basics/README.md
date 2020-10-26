## setup environment with docker, and run notebook

```
docker build -t tasty .
docker run -it -v ${PWD}:/opt -w /opt -p 8888:8888 tasty bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
```



### notebook

```
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

