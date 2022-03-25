# https://stackoverflow.com/questions/33824359/read-file-line-by-line-with-asyncio

import hashlib
import asyncio
import aiofiles

d={}
async def main():
    async with aiofiles.open('data_test','r') as f:
        async for l in f:
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

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

m = hashlib.sha256()
m.update(str(d).encode('utf-8'))
print(len(d))
print(m.digest().hex())

