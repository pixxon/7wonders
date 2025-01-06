FROM ubuntu:24.04

USER root

RUN apt update && \
    apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    libpython3-dev \
    g++ \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install torch torchvision opencv-python albumentations --break-system-packages && \
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --break-system-packages

USER ubuntu
