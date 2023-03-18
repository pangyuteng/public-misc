import os
from pathlib import Path
import numpy as np
import pandas as pd

def str2float(x):
    try:
        return float(x)
    except:
        return np.nan

fname = 'uclasalary.parquet.gzip'
if not os.path.exists(fname):
    df_list = []
    csv_file_list = [str(x) for x in Path(".").rglob("*.csv")]
    for csv_file in csv_file_list:
        df_list.append(pd.read_csv(csv_file))
    df = pd.concat(df_list)
    columns = ["Base Pay","Overtime Pay","Other Pay","Benefits","Total Pay","Pension Debt","Total Pay & Benefits"]
    for col in columns:
        df[col] = df[col].apply(lambda x: str2float(x))
    df.to_parquet(fname,compression='gzip')

df = pd.read_parquet(fname)
print(df.head())
print(df.columns)

print(df.shape)
df = df['Job Title'].apply(lamdba x: "Progammer Analyzst" in x)
print(df.shape)