import sys
import pandas as pd
from pandarallel import pandarallel
import itertools

pandarallel.initialize()

fname = sys.argv[1]

m_push_df = pd.DataFrame([],columns=['op','uid','mykey','myval'])
m_pop_df = pd.DataFrame([],columns=['op','uid','mykey'])


# `map`: transform each row
def transform_row(row):
    op_list = []

    l = row.iloc[0]
    
    v=l.strip('\n').split(" ")
    b=int(v[0])
    i=int(v[2])
    o=int(v[3])

    # to push
    for x in range(4+i,4+i+o):
        op_list.append(dict(
            op='push',
            uid=int(row.name),
            mykey=v[x+o],
            myval=(b,float(v[x])),
        ))

    # to pop
    for y in range(4,4+i):
        op_list.append(dict(
            op='pop',
            uid=int(row.name),
            mykey=v[y],
        ))

    return op_list

# gather all operations 
keep = 'last'
def process(chunk):
    global m_push_df
    global m_pop_df

    p = chunk.parallel_apply(transform_row,axis=1)
    op_list = list(itertools.chain(*p)) # flatten list of list.
    op_df=pd.DataFrame(op_list)
    
    # get `push` then remove duplicates
    push_df = op_df[op_df.op == 'push'].copy()
    m_push_df = pd.concat([m_push_df,push_df])
    m_push_df.drop_duplicates(subset=['mykey'],keep=keep,inplace=True) 
    
    # get `pop` then remove duplicates
    pop_df = op_df[op_df.op == 'pop'].copy()
    m_pop_df = pd.concat([m_pop_df,pop_df])
    m_pop_df.drop_duplicates(subset=['mykey'],keep=keep,inplace=True)


if __name__ == '__main__':
    chunksize = 80000
    with pd.read_csv(fname, 
        chunksize=chunksize,
        header=None) as reader:
        # todo make below parallelized
        for chunk in reader:
            process(chunk)

    # `filter`
    df = m_push_df.merge(m_pop_df,how='left',on=['mykey'])
    df['todel'] = False

    def markdel(row):
        # if `pop` is found and `pop` operation occured post `push` operation, mark as delete
        if row.op_y == 'pop' and row.uid_y >= row.uid_x:
            row.todel = True
        return row

    df = df.parallel_apply(markdel,axis=1)
    df=df[df.todel==False]
    df.sort_values(['uid_x'],axis=0,ascending=True)

    # finally transform to dict
    df=df[['mykey','myval_x']].copy()
    df.drop_duplicates(subset=['mykey','myval_x'],keep=keep,inplace=True) 
    ds = df.myval_x
    ds.index = df.mykey
    d = ds.to_dict()

    print(str(d)[-256:])
    import hashlib
    m = hashlib.sha256()
    m.update(str(d).encode('utf-8'))
    print(len(d))
    print(m.digest().hex())
