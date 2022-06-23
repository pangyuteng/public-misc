
reference. https://keras.io/examples/timeseries/timeseries_transformer_classification

objective. quick example on using transformer to forecast next 5-day stock price.

```
docker run -it -v $PWD:/workdir -w /workdir -p 8888:8888 tasty bash

docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -u $(id -u):$(id -g) -w /workdir -v $PWD:/workdir tasty:latest bash


# get historical data
python gen_data.py

# train transformer model
python train.py

# evaluate the model
python evaluate.py

train set: 60% accuracy
test set: 57% accuracy


# forecast next 5 day trend
python predict.py


```

below shows sample output from `python predict.py`, indicating price trend will go up for all ticker below with 51 to 60% confidence.

```
SPY [0.4309829 0.5690171]
QQQ [0.40523878 0.5947612 ]
IWM [0.38824543 0.61175454]
GLD [0.48685786 0.5131421 ]
TSLA [0.40803763 0.59196234]
NVDA [0.41636002 0.58364   ]
NFLX [0.4152975  0.58470243]
AMC [0.39998144 0.6000186 ]
```

