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

```
price-return.ipynb
https://twitter.com/aigonewrong/status/1320605318691262464
```

+ realized-and-implied-volatilities
```
realized-and-implied-volatilities.ipynb
https://twitter.com/aigonewrong/status/1326626008439480320
```

+ mean-reversion
```
mean-reversion.ipynb
https://twitter.com/aigonewrong/status/1329700527110643712
```

+ option pricing with black-schole-merton model
```
black-schole-merton-option-pricing.ipynb

```

+ poor man's back-testing
```

```
