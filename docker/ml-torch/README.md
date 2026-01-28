

general-ml purpose container 

TODO: make a torch version as well

```
docker pull pangyuteng/ml
```

```
docker build -t pangyuteng/ml:latest .
docker push pangyuteng/ml:latest

```


--gpus all --shm-size=1g
https://github.com/aws/sagemaker-python-sdk/issues/937