# run
import merge_csvs
# run
import cluster

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cluster import adjust_weights

iskmeans = True

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("word2vec.json","r") as f:
    word2vec = json.loads(f.read())

rawfname = 'raw-uc-salary.parquet.gzip'
df = pd.read_parquet(rawfname)
df.Year = df.Year.apply(lambda x: int(x))
if iskmeans:
    df['Job Title']= df['Job Title'].apply(lambda x: adjust_weights(x))
else:
    df['JobCategory']= df['Job Title'].apply(lambda x: adjust_weights(x,category=True))
df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))

if iskmeans:
    print(df.columns)
    X = np.array(df['Job Title'].apply(lambda x: word2vec[x]).tolist())
    print(X.shape)
    Y = model.predict(X)
    print(Y.shape)
    df['JobCategory']=Y

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
df = df[df.TotalPay>=50000] # assume 50k is full time?
df = df[df.Year>=2020]
sns.relplot(data=df,**kwargs)
plt.savefig('time-salary-filtered.png')

if not iskmeans:
    mylist = []
    for jobcategory in ["manager","professor","other"]:
        for year in [2020,2021,2022]:
            tmpbaseline = df[(df.Year==year-1)&(df.JobCategory==jobcategory)]
            tmp = df[(df.Year==year)&(df.JobCategory==jobcategory)]
            baseline_mean = tmpbaseline.TotalPay.mean()
            item_mean = tmp.TotalPay.mean()
            item_std = tmp.TotalPay.std()
            prct_change_from_prior_year = 100*(item_mean-baseline_mean)/baseline_mean
            myitem = dict(
                Year=year,
                JobCategory=jobcategory,
                TotalPayMean=np.round(item_mean,1),
                TotalPayStd=np.round(item_std,1),
                PrctChangeFromPriorYear=np.round(prct_change_from_prior_year,1),
            )
            mylist.append(myitem)
    mydf = pd.DataFrame(mylist)
    mydf.to_csv("salary_summary.csv",index=False)