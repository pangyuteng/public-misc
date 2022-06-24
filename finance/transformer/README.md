

[<img src="./static/tom-nn.png" width="300px">](https://www.tastytrade.com/shows/alpha-bytes/episodes/neural-networks-06-23-2022)


training a transformer with financial data.
```
primary objective. for shits and giggles

seconday objective.  quick example to show case how to:

    + transform price and volume as `X` (features from past 80 days) and `y` (price trend for the subsequent 5 days).
    + train and evaluate a vanilla transformer model using X & y s.
    + with the trained model, forecast next 5-day price trend for a basket of stocks.

reading materials:

for traders: 
    a gentle intro to neural network by Julia Spina (@FinancePhoton)
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

next five day forecast: (executed on 2022-06-24 14:24:32 utc)
  ticker  last_price  pred_prob direction
0    SPY      387.01       0.62        up
1    QQQ      292.93       0.56        up
2    IWM      174.51       0.56        up
3    GLD      170.39       0.57      down
4   UVXY       14.25       0.54        up
5    TLT      113.95       0.52      down
6   TSLA      733.84       0.52      down
7   NVDA      169.65       0.54        up
8   NFLX      190.29       0.55        up
9    AMC       12.31       0.50      down

```

misc thoughts.
```

    + this model could be actually useful if you tweak the code to attempt to predict volatility.
    + an alternative to predicting up/down is to predict up/down/sideways.
    + as with all ml models, there are lots of hyperparameters that can be tuned, ideally you want to be retraining every so often, examining the data multiple times, and finally front-test prior actual deployment.
    + snr for future price return is low, so the 51% accuracy on test set (below that of train set) is quiet comforting & surprising. comforting and expected as this is indication of some overfitting, surprising as accuracy of 51% seems believable and counters the arguement that past price data provide no information to predict future price trend.
    + again, this is a "hello world" demo, do not use this as trading advice.

```
