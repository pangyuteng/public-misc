
import pandas as pd
import itertools

fname = 'data_test'

master_df = pd.DataFrame()
m_push_df = pd.DataFrame([],columns=['op','uid','mykey','myval'])
m_pop_df = pd.DataFrame([],columns=['op','uid','mykey'])

# transform each row
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
            mykey=str(v[x]),
            myval=(b,float(v[x])),
        ))
    # to pop
    for y in range(4,4+i):
        op_list.append(dict(
            op='pop',
            uid=int(row.name),
            mykey=str(v[y]),
        ))

    return op_list

# map-reduce
def process(chunk):
    global m_push_df
    global m_pop_df

    p = chunk.apply(transform_row,axis=1)
    op_list = list(itertools.chain(*p))
    op_df=pd.DataFrame(op_list)
    
    # get then remove duplicates
    push_df = op_df[op_df.op == 'push'].copy()
    push_df.drop_duplicates(subset=['mykey'],keep='last',inplace=True)
    # note keep is set to `last`, repecting vanilla implementation of keeping only last val
    
    pop_df = op_df[op_df.op == 'pop'].copy()
    pop_df.drop_duplicates(subset=['mykey'],keep='last',inplace=True)

    # iterim merge
    m_push_df = pd.concat([m_push_df,push_df])
    m_push_df.drop_duplicates(subset=['mykey'],keep='last',inplace=True) 
    
    # iterm merge
    m_pop_df = pd.concat([m_pop_df,pop_df])
    m_pop_df.drop_duplicates(subset=['mykey'],keep='last',inplace=True)
        

chunksize = 500 #** 6
with pd.read_csv(fname, 
    chunksize=chunksize,
    header=None) as reader:

    # todo make below parallelized
    for chunk in reader:
        process(chunk)

# final processing.


print(m_push_df.shape,m_pop_df.shape)
