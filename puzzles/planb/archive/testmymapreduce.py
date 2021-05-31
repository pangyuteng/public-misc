import time
import hashlib
import pandas as pd
import itertools

d = {}

# transform each row
def transform(row):
    l = row.iloc[0]
    
    v=l.strip('\n').split(" ")
    b=int(v[0])
    i=int(v[2])
    o=int(v[3])

    tmp = {}
    for x in range(4+i,4+i+o):
        tmp[v[x]]=(b,float(v[x]))

    for y in range(4,4+i):
        p = d.pop(v[y],None)
    return tmp

# https://gist.github.com/FrancisBehnen/df215330de29e6969dec8d69658e2621

df = pd.read_csv('data_test',header=None)
df['dict'] = df.apply(transform,axis=1)

for n,row in df.iterrows():
    d.update(row['dict'])

m = hashlib.sha256()
m.update(str(d).encode('utf-8'))
print(len(d))
print(m.digest().hex())
