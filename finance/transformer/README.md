
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

train set: 63% accuracy
test set: 51% accuracy


# forecast next 5 day trend
python predict.py


```

below shows sample output from `python predict.py`, indicating price trend will go up for most tickers except for GLD,UVXY,TLT.

```
SPY [0.36587808 0.63412195]
QQQ [0.3270516 0.6729484]
IWM [0.28903788 0.7109621 ]
GLD [0.5021903 0.4978097]
UVXY [0.6297858  0.37021422]
TLT [0.6187992  0.38120082]
TSLA [0.30150697 0.69849306]
NVDA [0.33429468 0.6657053 ]
NFLX [0.33263874 0.6673613 ]
AMC [0.31235704 0.687643  ]
```

