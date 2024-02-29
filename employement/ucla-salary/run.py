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
    csv_file_list = sorted([str(x) for x in Path(".").rglob("*.csv")])
    for csv_file in csv_file_list:
        item = pd.read_csv(csv_file)
        print(item.Year.unique())
        df_list.append(item)
    df = pd.concat(df_list)
    columns = ["Base Pay","Overtime Pay","Other Pay","Benefits","Total Pay","Pension Debt","Total Pay & Benefits"]
    for col in columns:
        df[col] = df[col].apply(lambda x: str2float(x))
    df.to_parquet(fname,compression='gzip')

sns.set_theme(style="darkgrid")
df = pd.read_parquet(fname)
print(df.shape)

df.Year = df.Year.apply(lambda x: int(x))
df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))
# remove level from job title.
df["JobTitle"] = df["Job Title"].apply(lambda x: ''.join([i for i in x if not i.isdigit()]) )
df["JobTitle"] = df["Job Title"].apply(lambda x: x.replace(' ',''))

# weed out part time rolls via removing total comp to be less than 60k
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
    jobdf = baseline_df[(baseline_df.TotalPay>min_val)&(baseline_df.TotalPay<max_val)]
    for x in jobdf.JobTitle.unique():
        df.loc[df.JobTitle==x,"StartingSalaryLevel"]=salary_level
        
    print(salary_level,jobdf.TotalPay.median(),len(jobdf))

print(df.Year.unique())

sns.set(rc={'figure.figsize':(20,20)})
sns.relplot(
    data=df,
    x="Year",
    y="TotalPay",
    hue="StartingSalaryLevel",
    kind="line",
    facet_kws={"legend_out": True}
)
plt.savefig('time-salary.png')
