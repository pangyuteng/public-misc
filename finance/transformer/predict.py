
import numpy as np
import pandas as pd
from gen_data import get_latest_data
from train import get_model, checkpoint_filepath

symbols_list = ['SPY','QQQ','IWM','GLD','UVXY','TLT','TSLA','NVDA','NFLX','AMC']
x,rawdf = get_latest_data(symbols=symbols_list)

model = get_model()
model.load_weights(checkpoint_filepath)
y_hat = model.predict(x)

last_date = rawdf.index[-1]
mylist = []
for ticker,item in zip(symbols_list,y_hat):
    pred = np.argmax(item)
    pred_prob = item[pred]
    direction = 'up' if pred == 1 else 'down'
    mydict = dict(
        ticker=ticker,
        pred_int=pred,
        pred_str=direction,
        pred_prob=pred_prob,
        direction=direction
    )
    mylist.append(mydict)
df = pd.DataFrame(mylist)

# make prediction more readable
comment = f'\n\nhistorical data obtained (shape {rawdf.Close.shape}) with last date being {last_date}\n'
print(comment)
print("next five day forecast:")
print(df)

