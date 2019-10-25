#!/bin/bash

echo "clone mmdetection ..."
#git clone https://github.com/open-mmlab/mmdetection.git
echo "clone OK !"

echo "Create new python environment ..."
yes | conda create -n skypool_lmc python=3.7
echo "OK !"

source activate skypool_lmc
echo "Install pytorch and torchvision ..."
yes | conda install pytorch torchvision -c pytorch
echo "OK !"

echo "Install requirments ..."
pip install mmcv Cython pandas tqdm
yes | conda install nccl==2.0
echo "OK !"

export CUDA_HOME="/usr/local/cuda"
export PATH="$PATH:$CUDA_HOME/bin"
export LD_LIBRART_PATH="$CUDA_HOME/lib64"

cd mmdetection
pip install -v -e .




