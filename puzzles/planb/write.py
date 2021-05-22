# https://twitter.com/100trillionUSD/status/1396109728468684805

import random
import string
import numpy as np

seed = 21
line_num = 100000
fname = 'data_test'
max_num_per_line = 1000
max_char_per_line = 20

random.seed(seed)
np.random.seed(seed)

# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_char(y):
   return ''.join(random.choice(string.ascii_letters) for x in range(y))

with open(fname,'w') as f:
    for x in range(line_num):
        b = np.random.randint(1000) # not used
        o = np.random.randint(1,high=max_num_per_line) # num of float elements
        i = np.random.randint(1,high=max_char_per_line) # offset
        tmp = [b,None,i,o]
        char_list = [random_char(3) for _ in range(i)]
        num_list = np.random.rand(o)
        tmp.extend(char_list)
        tmp.extend(num_list)
        myline = ' '.join([str(x) for x in tmp])+'\n'
        f.write(myline)



