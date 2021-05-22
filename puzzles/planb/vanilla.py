import hashlib

d={}
with open('data_test','r') as f:
    for l in f:
        v=l.strip('\n').split(" ")
        b=int(v[0])
        i=int(v[2])
        o=int(v[3])
        # for x in range(4+i,4+i+o): d[str(v[x+o])]=(b,float(v[x]))
        #                                    ^^^ x+o seems odd. 
        for x in range(4+i,4+i+o):
            d[str(v[x])]=(b,float(v[x]))
        for y in range(4,4+i):
            p = d.pop(str(v[y]),None)

print(len(d))
m = hashlib.sha256()
m.update(str(d).encode('utf-8'))
print(m.digest().hex())

