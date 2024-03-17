# run
import merge_csvs

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def categorize(title):
    title = title.lower()
    # i give up, manually categorize.
    manager_code = "manager"
    prof_code = "professor"
    other_code = "other"

    if '-exec ' in title:
        title = manager_code
    elif 'mgr ' in title:
        title = manager_code
    elif 'manager' in title:
        title = manager_code
    elif ' mgr' in title:
        title = manager_code
    elif 'supervisor' in title:
        title = manager_code
    elif 'supv' in title:
        title = manager_code
    elif 'supvr' in title:
        title = manager_code
    elif 'coach' in title:
        title = manager_code
    elif title.endswith(' prof'):
        title = prof_code
    elif 'prof ' in title:
        title = prof_code
    elif 'prof-' in title:
        title = prof_code
    elif 'professor' in title:
        title = prof_code
    elif 'dean' in title:
        title = prof_code
    else:
        title = other_code
    return title

rawfname = 'raw-uc-salary.parquet.gzip'
df = pd.read_parquet(rawfname)
df.Year = df.Year.apply(lambda x: int(x))
df['JobCategory']= df['Job Title'].apply(lambda x: categorize(x))
df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))

kwargs =dict( 
    x="Year",
    y="TotalPay",
    hue="JobCategory",
    kind="line",
    errorbar=('ci', 95),
    facet_kws={"legend_out": True}
)
sns.set(rc={'figure.figsize':(20,20)})
sns.relplot(data=df,**kwargs)
plt.savefig('time-salary-all.png')
plt.close()

# ?? df = df[(df.Year >= 2017)&(df.TotalPay>50000)]

mylist = []
for jobcategory in ["manager","professor","other"]:
    for year in sorted(df.Year.unique()):
        tmpbaseline = df[(df.Year==year-1)&(df.JobCategory==jobcategory)]
        tmp = df[(df.Year==year)&(df.JobCategory==jobcategory)]
        baseline_mean = tmpbaseline.TotalPay.mean()
        item_mean = tmp.TotalPay.mean()
        item_std = tmp.TotalPay.std()
        prct_change_from_prior_year = 100*(item_mean-baseline_mean)/baseline_mean
        myitem = dict(
            Year=int(year),
            JobCategory=jobcategory,
            TotalPayMean=np.round(item_mean,1),
            TotalPayStd=np.round(item_std,1),
            PrctChangeFromPriorYear=np.round(prct_change_from_prior_year,1),
        )
        mylist.append(myitem)
mydf = pd.DataFrame(mylist)
mydf.to_csv("salary_summary.csv",index=False)


kwargs =dict( 
    x="Year",
    y="PrctChangeFromPriorYear",
    hue="JobCategory",
    kind="line",
    facet_kws={"legend_out": True}
)
sns.set(rc={'figure.figsize':(20,20)})
plt.title('percent change from prior year filter by salary>=50K')
sns.relplot(data=mydf,**kwargs)
plt.savefig('time-salary-prctchange.png')
