FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

COPY environments/environment.yml /app/environment.yml
RUN micromamba create -qy -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
 && micromamba clean -qya

RUN ln -s "$CONDA_PREFIX/lib/libnvrtc.so.11.8.89" "$CONDA_PREFIX/lib/libnvrtc.so"

CMD ["bash"]