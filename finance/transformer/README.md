

[<img src="./static/tom-nn.png" width="300px">](https://www.tastytrade.com/shows/alpha-bytes/episodes/neural-networks-06-23-2022)


training a transformer with financial data.
```
primary objective. for shits and giggles

seconday objective.  quick example to show case how to:

    + transform price and volume as `X` (features from past 80 days) and `y` (price trend for the subsequent 5 days).
    + train and evaluate a vanilla transformer model using X & y s.
    + with the trained model, forecast next 5-day price trend for a basket of stocks.

misc objective.

    + how are we ** not ** going to work neural networks into tasty.
    + or if you want, tweak the code to attempt to predict volatility.

reading materials:

for traders: 
    a gentle intro on neural network by Julia Spina (@FinancePhoton)
    https://www.tastytrade.com/shows/alpha-bytes/episodes/neural-networks-06-23-2022

reference / for devs: 
    Timeseries classification with a Transformer model by Theodoros Ntakouris (@zarkopafilis)
    https://keras.io/examples/timeseries/timeseries_transformer_classification


```

instructions
```

# build container

cd finance
docker build -t tasty .

# with no gpu

cd finance/transformer
docker run -it -u $(id -u):$(id -g) \
    -w /workdir -v $PWD:/workdir tasty:latest bash

# with 1 gpu, device 0
cd finance/transformer
docker run -it -u $(id -u):$(id -g) \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 
    -w /workdir -v $PWD:/workdir tasty:latest bash

# get historical data
python gen_data.py

# train transformer model
python train.py

# evaluate the model
python evaluate.py

# trained date 2022-06-23
# train set: 63% accuracy
# test set: 51% accuracy

# forecast next 5 day trend
python predict.py


```

below shows sample output from `python predict.py`, where model predicts the price trend for June 23-30 for a basket of tickers.
```

historical data obtained (shape (7405, 10)) with last date being 2022-06-23 00:00:00

next five day forecast:
  ticker  pred_int pred_str  pred_prob direction
0    SPY         1       up   0.637715        up
1    QQQ         1       up   0.674343        up
2    IWM         1       up   0.713386        up
3    GLD         0     down   0.500847      down
4   UVXY         0     down   0.631274      down
5    TLT         0     down   0.620470      down
6   TSLA         1       up   0.699413        up
7   NVDA         1       up   0.666117        up
8   NFLX         1       up   0.670497        up
9    AMC         1       up   0.687950        up

```

