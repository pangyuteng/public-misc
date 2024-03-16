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

iskmeans = False

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
df = df[df.Year>=2020]
sns.relplot(data=df,**kwargs)
plt.savefig('time-salary-filtered.png')

if not iskmeans:
    mylist = []
    for jobcategory in ["manager","professor","other"]:
        for year in [2020,2021,2022]:
            tmp = df[(df.Year==year)&(df.JobCategory==jobcategory)]
            myitem = dict(
                Year=year,
                JobCategory=jobcategory,
                TotalPayMean=tmp.TotalPay.mean(),
                TotalPayStd=tmp.TotalPay.std(),
            )
            mylist.append(myitem)
    mydf = pd.DataFrame(mylist)
    mydf.to_csv("salary_summary.csv")