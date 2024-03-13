import pandas as pd
import json

json_file = 'job_list.json'
json_dict_file = 'job_dict.json'

if not os.path.exists(json_file):
    rawfname = 'raw-uc-salary.parquet.gzip'
    df = pd.read_parquet(rawfname)
    job_list = df['Job Title'].unique().tolist()
    print('job_list',len(job_list))
    with open(json_file,'w') as f:
        f.write(json.dumps(job_list,indent=True))

with open(json_file,'r') as f:
    job_list = json.loads(f.read())

with open(json_file,'w') as f:
    f.write(json.dumps(job_list,indent=True))

