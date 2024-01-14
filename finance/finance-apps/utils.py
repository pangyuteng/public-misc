
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

# perm issue with default cache dir, switch to /tmp
yf.set_tz_cache_location("/tmp/cache/location")

sector_list = ['XLE','XLF','XLU','XLI','XLK','XLV','XLY','XLP','XLB']
ticker_list = ['BTC-USD','SPY','QQQ','^VIX','^TNX','^IRX']
m2sl_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=M2SL&scale=left&cosd=1959-01-01&coed=2022-09-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2022-10-31&revision_date=2022-10-31&nd=1959-01-01'

def get_data(lookback=-2000,roll=200):

    # get historical daily price for SPY and ^VIX
    ticker_list.extend(sector_list)
    cool = ' '.join(ticker_list)

    df = yf.download(cool,period="max",interval="1d")
    df = df['Close']
    df=df.dropna()
    spy_ret = np.log(df['SPY']) - np.log(df['SPY'].shift(1))
    df['SPY_ret'] = spy_ret

    for x in ticker_list:
        if x.startswith('^'):
            pass
        else:
            price_ret=np.log(df[x]) - np.log(df[x].shift(1))
            df[x+"_ret"]=price_ret
    df = df.dropna()
    odf = df.copy()
    print(df.shape)
    print(df.index[0],df.index[-1])

    m2sl = pd.read_csv(m2sl_url)
    m2sl['Date'] = m2sl.DATE.apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
    m2sl = m2sl.set_index('Date')
    m2sl = m2sl.reindex(odf.index, method='ffill')
    m2sl = m2sl[['M2SL']]
    print(df.shape)
    print(m2sl.shape)
    df = odf.merge(m2sl,how='left',left_index=True,right_index=True)
    print(df.shape)

    for x in sector_list:
        df[f'{x}_corr'] = df[f"{x}_ret"].rolling(roll).corr(df['SPY_ret'])

    return df