# https://twitter.com/100trillionUSD/status/1396109728468684805

import random
import string
import numpy as np

seed = 21
line_num = 1000
fname = 'data_test'
max_x_per_line = 5000
max_y_per_line = 100
max_lim = 1e4

random.seed(seed)
np.random.seed(seed)

# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_char(y):
   return ''.join(random.choice(string.ascii_letters) for x in range(y))

with open(fname,'w') as f:
    for x in range(line_num):
        b = np.random.randint(1000) # not used
        y_len = np.random.randint(1,high=max_y_per_line)
        x_len = np.random.randint(1,high=max_x_per_line)
        tmp = [b,None,y_len,x_len]
        y_list = np.random.randint(0,high=max_lim,size=y_len)
        x_list = np.random.randint(0,high=max_lim,size=x_len)
        tmp.extend(y_list)
        tmp.extend(x_list)
        myline = ' '.join([str(x) for x in tmp])+'\n'
        f.write(myline)


