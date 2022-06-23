
reference. https://keras.io/examples/timeseries/timeseries_transformer_classification

objective. quick example on using transformer to forecast next 5-day stock price.

```
docker run -it -v $PWD:/workdir -w /workdir -p 8888:8888 tasty bash

docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir tasty:latest bash


# get historical data
python gen_data.py

# train transformer model
python train.py

# forecast next 5 day trend
python predict.py
```

```
SPY [0.43828067 0.5617193 ]
QQQ [0.43545115 0.56454885]
IWM [0.41574058 0.58425945]
GLD [0.49746338 0.5025366 ]

above sample ouptut indicates price trend will go up for all tickets with 50 to 56% probability
```

