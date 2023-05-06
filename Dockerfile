FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
COPY environments/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && rm /app/requirements.txt
 
CMD ["bash"]