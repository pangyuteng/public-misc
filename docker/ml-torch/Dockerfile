FROM huggingface/transformers-pytorch-gpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libosmesa6-dev -yq

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get install -yq libgles2-mesa-dev
RUN adduser --uid 1000 --disabled-password ubuntu
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow
USER ubuntu