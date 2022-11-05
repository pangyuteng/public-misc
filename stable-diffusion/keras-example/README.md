
```

https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion

docker build -t keras-stable-diffusion .

docker run --runtime=nvidia -it -u $(id -u):$(id -g) -w $PWD -v /mnt:/mnt keras-stable-diffusion bash

```