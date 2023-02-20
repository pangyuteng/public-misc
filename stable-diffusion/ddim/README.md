
```
docker run --runtime=nvidia -it -w $PWD -v /mnt:/mnt keras-stable-diffusion bash

# NOTE0. no `-u` flag since flower dataset needs `/` write perm
# NOTE1. set hard memory - no balloning device if you are using VM!

```
