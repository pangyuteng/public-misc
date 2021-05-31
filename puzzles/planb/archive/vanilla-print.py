import sys
d={}                    # d will be ~60GB in ram
yo=False
with open('data_test','r') as f:    # 200GB csv file
    for l in f:                 # ~700,000 lines
        v=l.split(" ")             # v contains thousands of items
        b=int(v[0])
        i=int(v[2])
        o=int(v[3])
        print('forx')
        print(v)
        print(b,i,o,'@@',4+i,4+i+o)
        for x in range(4+i,4+i+o):
            d[v[x+o]]=(b,float(v[x]))
            print(x+o)
            print((b,float(v[x])))

        print('fory')
        print(4,4+i)
        for y in range(4,4+i):
            print(v[y])
            del d[v[y]]
            yo = True
        if yo:
            sys.exit(0)  
print(str(d)[-256:])
import hashlib
m = hashlib.sha256()
m.update(str(d).encode('utf-8'))
print(len(d))
print(m.digest().hex())

