docker build \
    --build-arg GROUPID=$(id -g) \
    -t pangyuteng/itksnap:kasm -f Dockerfile.kasm .
docker push pangyuteng/itksnap:kasm

# --build-arg GROUPID=$(id -g) \
