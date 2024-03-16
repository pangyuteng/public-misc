import os
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

data = yf.download("btc-usd mstr aapl qqq tsla nvda smci",
    start="2020-07-28",end="2024-03-12")
df = data['Adj Close']
df = df.dropna()
for x in df.columns:
    tmp = df[x]/df['BTC-USD']
    tmp /= tmp[0]
    plt.plot(df.index,tmp,label=x)
plt.legend()
plt.grid(True)
plt.savefig('hohoho.png')

# https://twitter.com/PrestonPysh/status/1767642324954136958