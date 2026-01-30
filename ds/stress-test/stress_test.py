import os
import aiohttp
import asyncio
import time

user = os.environ["USER"]
password = os.environ["PASSWORD"]
url = "http://10.9.146.69:5003/batch_markup_service?reader_initials=MML&source=scan&experiment_id=10158&patient_id=KiTS-00091"
url = "http://10.9.146.69:5003/batch_markup_service?reader_initials=PYT&source=scan&experiment_id=99902142&patient_id=ABC-001"

async def myrequest(session):
    async with session.get(url,auth=aiohttp.BasicAuth(user, password)) as response:
        html = await response.text()
        return html
async def main():
    async with aiohttp.ClientSession() as session:
        my_list = [myrequest(session) for x in range(100)]
        out = await asyncio.gather(*my_list)
        #print(out)

start_time = time.time()
asyncio.run(main())
end_time = time.time()
print(f'duration {end_time-start_time}sec')

"""

docker run -it -u $(id -u):$(id -g) -v /cvibraid:/cvibraid registry.cvib.ucla.edu/qia:prod bash
bash run.sh

run.sh
```
#!/bin/bash
export MYAUTH=xxx
export USER=xxx
export PASSWORD=xxx
python stress_test.py
```

curl -H "Authorization: Basic ${MYAUTH}" "http://10.9.146.69:5003/batch_markup_service?reader_initials=PYT&source=scan&experiment_id=99902142&patient_id=ABC-001"

"""