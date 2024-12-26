#! /bin/bash
apt-get update -yqq
apt-get install -yqq \
    libgl1 \
    libglib2.0-0 \

pip install torch==2.0.1 torchvision==0.15.2 matplotlib==3.5.0