
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
