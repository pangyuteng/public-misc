import pandas as pd

d = {}

# transform each row
def transform(row):
    l = row.iloc[0]
    v=l.strip('\n').split(" ")
    b=int(v[0])
    i=int(v[2])
    o=int(v[3])
    for x in range(4+i,4+i+o):
        d[v[x+o]]=(b,float(v[x]))
    for y in range(4,4+i):
        del d[v[y]]

df = pd.read_csv('data_test',header=None)
df.apply(transform,axis=1)

print(str(d)[-256:])
import hashlib
m = hashlib.sha256()
m.update(str(d).encode('utf-8'))
print(len(d))
print(m.digest().hex())
