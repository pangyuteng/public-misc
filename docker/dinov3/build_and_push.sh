# docker build --secret id=hf,type=file,src=.env -t pangyuteng/dinov3 .
docker build --secret id=hf,env=HF_TOKEN -t pangyuteng/dinov3 .
docker push pangyuteng/dinov3
