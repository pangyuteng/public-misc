import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def str2float(x):
    try:
        return float(x)
    except:
        return np.nan

rawfname = 'raw-uc-salary.parquet.gzip'
fname = 'processed-uc-salary.parquet.gzip'
if not os.path.exists(rawfname):
    df_list = []
    csv_file_list = sorted([str(x) for x in Path(".").rglob("*.csv")])
    for csv_file in csv_file_list:
        item = pd.read_csv(csv_file)
        print(item.Year.unique())
        df_list.append(item)
    df = pd.concat(df_list)
    columns = ["Base Pay","Overtime Pay","Other Pay","Benefits","Total Pay","Pension Debt","Total Pay & Benefits"]
    for col in columns:
        df[col] = df[col].apply(lambda x: str2float(x))
    
    df.to_parquet(rawfname,compression='gzip')

if not os.path.exists(fname):
    df = pd.read_parquet(rawfname)
    print(df.shape)
    df.Year = df.Year.apply(lambda x: int(x))
    df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))
    df.to_parquet(fname,compression='gzip')

df = pd.read_parquet(fname)
df['optin']=False
optinlist = []
for employee_name in df['Employee Name'].unique():
    tmp = df[df['Employee Name']==employee_name]
    if len(tmp) > 5 and 2022 in tmp.Year:
        optinlist.append(employee_name)
print(len(optinlist))
    
