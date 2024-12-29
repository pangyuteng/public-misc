

```


https://opentimestamps.org

docker run -it \
    -v /mnt:/mnt \
    -v ${PWD}/bitcoin.conf:/root/.bitcoin/bitcoin.conf \
    pangyuteng/ml:office-latest bash

# tstamp the document

ots stamp hello-world.txt

# 
# wait for 24 hrs prior verify.
#   donate btc to server 
#   verify document tstamp, use flag `--no-bitcoin`**
#   or add `/root/.bitcoin/bitcoin.conf` when container is started

ots --no-bitcoin verify hello-world.txt.ots -f hello-world.txt
ots verify hello-world.txt.ots -f hello-world.txt

```

```
**
"Could not connect to Bitcoin node: Cookie file unusable"


from https://petertodd.org/2016/opentimestamps-announcement
"""
However, the client does come with a number of example timestamps which you can try verifying immediately. You’ll need a local Bitcoin Core node (a pruned node is fine) with the rpcuser and rpcpassword options set in ~/.bitcoin/bitcoin.conf to allow the OpenTimestamps client to connect to your node via the RPC interface. Once that’s setup, let’s try verifying examples/hello-world.txt:
"""

```