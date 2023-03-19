import os
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

fname = 'uc-salary.parquet.gzip'
if not os.path.exists(fname):
    df_list = []
    csv_file_list = [str(x) for x in Path(".").rglob("*.csv")]
    for csv_file in csv_file_list:
        item = pd.read_csv(csv_file)
        item = item.iloc[:1000,:]
        df_list.append(item)
    df = pd.concat(df_list)
    columns = ["Base Pay","Overtime Pay","Other Pay","Benefits","Total Pay","Pension Debt","Total Pay & Benefits"]
    for col in columns:
        df[col] = df[col].apply(lambda x: str2float(x))
    df.to_parquet(fname,compression='gzip')

sns.set_theme(style="darkgrid")
df = pd.read_parquet(fname)

df["JobTitle"] = df["Job Title"].apply(lambda x: ''.join([i for i in x if not i.isdigit()]) )
df.Year = df.Year.apply(lambda x: int(x))
df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))
sns.set(rc={'figure.figsize':(20,20)})
#sns.lineplot(data=df, x="Year", y="TotalPay", hue="JobTitle")
sns.relplot(
    data=df,
    x="Year",
    y="TotalPay",
    hue="JobTitle",
    kind="line",
    facet_kws={"legend_out": True}
)
plt.savefig('time-salary.png')

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.3)
# plt.savefig('time-salary.png')