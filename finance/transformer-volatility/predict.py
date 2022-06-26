import datetime
import numpy as np
import pandas as pd
from gen_data import get_latest_data
from train import get_model, checkpoint_filepath

tstamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S utc')
symbols_list = ['SPY','QQQ','IWM','GLD','UVXY','TLT','TSLA','NVDA','NFLX','AMC']
x,rawdf = get_latest_data(symbols=symbols_list)

model = get_model()
model.load_weights(checkpoint_filepath)
y_hat,v_hat = model.predict(x)

last_date = rawdf.index[-1]
mylist = []
for ticker,yitem,vitem in zip(symbols_list,y_hat,v_hat):
    close_price = np.around(rawdf[('Close',ticker)][-1],2)
    print(ticker,close_price,yitem,vitem)
    idx_price = np.argmax(yitem)
    pred_price = yitem[idx_price]
    idx_volatility = np.argmax(vitem)
    pred_volatility = vitem[idx_volatility]
    mydict = dict(
        ticker=ticker,
        last_price=close_price,
        price_prob=np.around(pred_price,2),
        volatility_prob=np.around(pred_volatility,2),
        price_trend='up' if idx_price == 1 else 'down',
        volatility_trend='up' if idx_volatility ==1 else 'down',
    )
    mylist.append(mydict)
df = pd.DataFrame(mylist)

# make prediction more readable
comment = f'\n\nhistorical data obtained (shape {rawdf.Close.shape}) with last date being {last_date}\n'
print(comment)
print(f"next five day forecast: (executed on {tstamp})")
print(df)


