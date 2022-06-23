import os
import sys
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import tensorflow_datasets as tfds
import tensorflow as tf
import time
import traceback

FEATURE_DIM = 5
def etl(history):
    df = pd.DataFrame()
    df['price'] = history.Close
    df['volume'] = history.Volume
    df.index = history.index
    df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))
    df['price'] = df.price
    df['hist_vol'] = df.log_ret.rolling(21).std()*np.sqrt(252)*100
    df = df.dropna()
    df['s_price']=df.price # use for label
    df['s_ret']=df.log_ret
    df['s_vol']=df.hist_vol
    df['s_volume']=df.volume
    df['s_month']=df.index.month.values
    df['s_day']=df.index.day.values
    data = df[['s_price','s_ret','s_vol','s_month','s_day']].values
    #scaler = MinMaxScaler() # same reasoning, also not a fan of minmax scaler
    # compute zscore and scale to -1 to 1, and use tanh as oppose to clip so value still makes some sense for fat tails.
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    transformed = scaler.transform(data)
    transformed = np.tanh(transformed)
    df['s_price']=transformed[:,0]
    df['s_ret']=transformed[:,1]
    df['s_vol']=transformed[:,2]
    df['s_month']=transformed[:,3]
    df['s_day']=transformed[:,4]
    mylist = [df.s_price.values,df.s_ret.values,df.s_vol.values,df.s_month.values,df.s_day.values]
    assert(len(mylist)==FEATURE_DIM)
    return np.stack(mylist,axis=-1)

look_back=80
look_forward=5
total_days = look_back+look_forward
def chunckify(arr):
    tmp_list = []
    for x in np.arange(total_days,arr.shape[0]-total_days,look_forward):
        tmp = arr[x:x+total_days]
        if tmp.shape != (total_days,FEATURE_DIM):
            continue        
        price_trend = 1 if tmp[-1,0]-tmp[-look_forward,0] > 0 else -1
        x,y = tmp[:look_back,1:],price_trend
        tmp_list.append((x,y))
    return tmp_list


def get_latest_data(symbols=['SPY','QQQ',]):
    
    final_list = []
    ticker_list = yf.Tickers(' '.join(symbols))
    history = ticker_list.history(period="max")
    for ticker in ticker_list.tickers:
        mydf = history[[('Close',ticker),('Volume',ticker)]]
        arr = etl(mydf)
        # get most recent Y
        arr = arr[-look_back:,1:]
        final_list.append(arr)
    x = np.array(final_list)    
    return x, history

#DEBUG = False
url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
def main():
    
    symbols_list = ['SPY','QQQ','IWM','GLD','UVXY','TLT','TSLA','NVDA','NFLX','AMC']
    #df=pd.read_csv(url)
    #symbols_list.extend(list(df.Symbol.values))
    final_list = []
    for x in np.arange(0,len(symbols_list),100):
        try:
            symbols = symbols_list[x:x+100]
            print(symbols)
            ticker_list = yf.Tickers(' '.join(symbols))
            history = ticker_list.history(period="max")
            for ticker in ticker_list.tickers:
                try:
                    mydf = history[[('Close',ticker),('Volume',ticker)]]
                    arr = etl(mydf)
                    if arr.shape[0] > total_days:
                        tmp_list = chunckify(arr)
                        final_list.extend(tmp_list)
                except:
                    traceback.print_exc()
                    raise ValueError()
            time.sleep(2)
        except:
            traceback.print_exc()
            raise ValueError()
    
    Y = np.array([x[1] for x in final_list])
    X = np.array([x[0] for x in final_list])
    np.save('X.npy', X)
    np.save('Y.npy', Y)    

if __name__ == '__main__':
    main()