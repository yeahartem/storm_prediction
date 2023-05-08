FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

USER root

RUN curl -sL https://github.com/mamba-org/micromamba-releases/releases/download/1.4.2-2/micromamba-linux-64.tar.bz2 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/root//micromamba \
    CONDA_PREFIX=/root/micromamba \
    PATH=/root/micromamba/bin:$PATH

COPY environments/environment.yml /app/environment.yml
RUN micromamba create -y -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
 && micromamba clean -qya
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

CMD tail -f /dev/null