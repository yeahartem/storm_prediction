FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app
RUN ln -s "$CONDA_PREFIX/lib/libnvrtc.so.11.8.89" "$CONDA_PREFIX/lib/libnvrtc.so"

ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
COPY environments/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && rm /app/requirements.txt
 
CMD ["bash"]