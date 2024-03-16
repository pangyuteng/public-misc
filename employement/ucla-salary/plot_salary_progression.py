import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cluster import adjust_weights

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("word2vec.json","r") as f:
    word2vec = json.loads(f.read())

rawfname = 'raw-uc-salary.parquet.gzip'
df = pd.read_parquet(rawfname)
df.Year = df.Year.apply(lambda x: int(x))
df['Job Title']= df['Job Title'].apply(lambda x: adjust_weights(x))
df["TotalPay"] = df["Total Pay & Benefits"].apply(lambda x: int(x))
print(df.columns)
X = np.array(df['Job Title'].apply(lambda x: word2vec[x]).tolist())
print(X.shape)
Y = model.predict(X)
print(Y.shape)
df['JobCategory']=Y

sns.set(rc={'figure.figsize':(20,20)})
sns.relplot(
    data=df,
    x="Year",
    y="TotalPay",
    hue="JobCategory",
    kind="line",
    facet_kws={"legend_out": True}
)
plt.savefig('time-salary.png')