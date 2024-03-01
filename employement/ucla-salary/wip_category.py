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

def shrink_title(x):
    x = ''.join([i for i in x if not i.isdigit()])
    plist = [" ","-","/","_"]
    for p in plist:
        x = x.replace(p,"")
    return x

starting_year = int(sys.argv[1]) # 2011 or 2019 
rawfname = f'{starting_year}-raw-uc-salary.parquet.gzip'
fname = f'{starting_year}-processed-uc-salary.parquet.gzip'
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

    # job title update ??
    df = df[df.Year>=starting_year]

    df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))

    # remove level from job title, probably not great.
    df["JobTitle"] = df["Job Title"].apply(lambda x: shrink_title(x))

    df['StartingSalaryLevel'] = None
    min_year = df.Year.min()
    baseline_df = df[df.Year == min_year]
    salary_range = {
        '50k-80k':(50000,80000),
        '80k-120k':(80000,120000),
        '120k-250k':(120000,250000),
        '250k-400k':(250000,400000),
        '400k+':(400000,5000000),
    }

    print(baseline_df.TotalPay.min(),baseline_df.TotalPay.max())

    for salary_level,range_min_max in salary_range.items():
        min_val,max_val = range_min_max
        jobdf = baseline_df[(baseline_df.TotalPay>=min_val)&(baseline_df.TotalPay<max_val)]
        for x in jobdf.JobTitle.unique():
            df.loc[df.JobTitle==x,"StartingSalaryLevel"]=salary_level
            
        print(salary_level,jobdf.TotalPay.median(),len(jobdf))

    print(df.Year.unique())
    df.to_parquet(fname,compression='gzip')

df = pd.read_parquet(fname)
for salary_level in df.StartingSalaryLevel.unique():
    tmp = df[df.StartingSalaryLevel==salary_level]
    print(salary_level,len(tmp.JobTitle.unique()))
    print(tmp.JobTitle.unique()[:10])

for year in df.Year.unique():
    tmp = df[df.Year==year]
    print(year, tmp.TotalPay.median(),len(tmp))
    print(tmp.StartingSalaryLevel.unique())
    print("--")
sns.set(rc={'figure.figsize':(40,40)})
sns.relplot(
    data=df,
    x="Year",
    y="TotalPay",
    hue="StartingSalaryLevel",
    kind='line',
    facet_kws={"legend_out": True}
)
plt.savefig(f'{starting_year}-time-salary.png')
