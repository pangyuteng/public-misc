
import sys
import multiprocessing
import pandas as pd
#from pandarallel import pandarallel
import itertools
#pandarallel.initialize()

fname = sys.argv[1]

m_push_df = pd.DataFrame([],columns=['op','uid','mykey','myval'])
m_pop_df = pd.DataFrame([],columns=['op','uid','mykey'])


# `map`: transform each row
def transform_row(row):
    op_list = []
    row_index,myline = row  
    v=myline.split(" ")
    b=int(v[0])
    i=int(v[2])
    o=int(v[3])

    # to push
    for x in range(4+i,4+i+o):
        op_list.append(dict(
            op='push',
            uid=row_index,
            mykey=v[x+o],
            myval=(b,float(v[x])),
        ))

    # to pop
    for y in range(4,4+i):
        op_list.append(dict(
            op='pop',
            uid=row_index,
            mykey=v[y],
        ))

    return op_list

# gather all operations 
keep = 'last'
def myprocess(chunk):

    p = [] 
    for x in chunk:
        p.append(transform_row(x))

    op_list = list(itertools.chain(*p)) # flatten list of list.
    op_df=pd.DataFrame(op_list)
    #print('myprocess',op_df.shape)
    return op_df

def mymerge(op_df):

    global m_push_df
    global m_pop_df

    # get `push` then remove duplicates
    push_df = op_df[op_df.op == 'push'].copy()
    m_push_df = pd.concat([m_push_df,push_df])
    m_push_df.drop_duplicates(subset=['mykey'],keep=keep,inplace=True) 
    
    # get `pop` then remove duplicates
    pop_df = op_df[op_df.op == 'pop'].copy()
    m_pop_df = pd.concat([m_pop_df,pop_df])
    m_pop_df.drop_duplicates(subset=['mykey'],keep=keep,inplace=True)
    
    #print('mymerge',m_push_df.shape,m_pop_df.shape,)

if __name__ == '__main__':

    # https://stackoverflow.com/questions/4047789/parallel-file-parsing-multiple-cpu-cores
    numthreads = 8
    numlines = 10000
    lines = open(fname).readlines()
    lines = [(n,l) for n,l in enumerate(lines)]
    # create the process pool
    with multiprocessing.Pool(processes=numthreads) as pool:

        # map the list of lines into a list of result dicts
        result_list = pool.map(myprocess, 
            (lines[line:line+numlines] for line in range(0,len(lines),numlines) ) )

        for x in result_list:
            mymerge(x)
    print('merge done')
    print(m_push_df.shape,m_pop_df.shape,)
    # `filter`
    df = m_push_df.merge(m_pop_df,how='left',on=['mykey'])
    df=df[df.op_y!='pop']

    # final "reduce"
    df.sort_values(['uid_x'],axis=0,ascending=True)
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
