
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

train set: 56% accuracy
test set: 55% accuracy


# forecast next 5 day trend
python predict.py


```

below shows sample output from `python predict.py`, indicating price trend will go up for SPY,QQQ,IWM (with 59% confidence) while GLD will likely go down (with lower confidence >50%).

```
SPY [0.4073227  0.59267724]
QQQ [0.4052036 0.5947964]
IWM [0.40801558 0.5919844 ]
GLD [0.502255   0.49774495]
```

