
predict price and volatility trend for next 5 days.

```

# build container

cd finance
docker build -t tasty .


# with 1 gpu, device 0
cd finance/transformer
docker run -it -u $(id -u):$(id -g) --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -w /workdir -v $PWD:/workdir tasty:latest bash

# get historical data
python gen_data.py

# train transformer model
python train.py

# evaluate the model
python evaluate.py

# forecast next 5 day trend
python predict.py


```

sample output from `python predict.py`
```
next five day forecast: (executed on 2022-06-26 17:45:39 utc)
  ticker  last_price price_trend volatility_trend
0    SPY      390.08          up             down
1    QQQ      294.61          up               up
2    IWM      175.09        down             down
3    GLD      170.09          up             down
4   UVXY       14.39          up             down
5    TLT      112.56        down             down
6   TSLA      737.12          up             down
7   NVDA      171.26          up               up
8   NFLX      190.85          up             down
9    AMC       12.47        down               up

```