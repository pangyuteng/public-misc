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

+ trading $BABA with mean-reversion signals.
```
mean-reversion-2020-11-11-BABA.ipynb
mean-reversion-2020-11-12-BABA.ipynb
https://twitter.com/aigonewrong/status/1326571891683782657
https://twitter.com/aigonewrong/status/1327090636210618368
```

+ option pricing with black-schole-merton model
```
https://twitter.com/aigonewrong/status/1330650128961572867
black-schole-merton-option-pricing.ipynb

```

+ poor man's backtesting - attempt 0
```
https://twitter.com/aigonewrong/status/1330650122548441089
poor-persons-option-backtest-attempt-0.ipynb
```

+ CBOE CNDR replica - attempt 0
```
https://twitter.com/aigonewrong/status/1335783839055024128
cboe-cndr-replica.ipynb
```


### TODOS:

+ transformers forecast mean reversion signals.
+ do wsb sentiment analysis again
+ backest with quantconnect
+ make website from above items
