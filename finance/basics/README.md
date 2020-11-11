## setup environment with docker, and run notebook

```
docker build -t tasty .
docker run -it -v ${PWD}:/opt -w /opt -p 8888:8888 tasty bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
```

### notebook

```
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```


### tweets

+ price-return
https://twitter.com/aigonewrong/status/1320605318691262464

+ realized-and-implied-volatilities
https://twitter.com/aigonewrong/status/1326626008439480320
