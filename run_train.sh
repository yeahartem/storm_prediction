#!/bin/bash

sudo add-apt-repository ppa:jonathonf/gcc
sudo apt-get update
sudo apt install gcc-7
/opt/conda/bin/python3 src/regression/train.py