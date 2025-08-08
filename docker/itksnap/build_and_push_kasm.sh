docker build \
    --build-arg GROUPID=$(id -g) \
    --build-arg GROUPID2=1474381808 \
    -t pangyuteng/itksnap:kasm \
    -f Dockerfile.kasm .
docker push pangyuteng/itksnap:kasm

# --build-arg GROUPID=$(id -g) \

# docker build \
#   --build-arg GROUPID=$(id -g) \
#   --build-arg USERID=$(id -u) \
#   --build-arg USERNAME=kasm-user \
#   -t pangyuteng/itksnap:kasm-pteng \
#   -f Dockerfile.kasm.user . 

# docker push pangyuteng/itksnap:kasm-pteng