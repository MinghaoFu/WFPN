# Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution

This repository is for WFPN introduced in the following paper

Minghao Fu, Dongyang Zhang, Min Lei, Kun He, Changyu Li, and Jie Shao, "Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution", BMVC 2022, [Paper](...)

This code is modified from [EDSR(Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch) and tested on Ubuntu 16.04.6 LTS environment (Python 3.7.11, Pytorch 1.7.1, CUDA 9.0, cuDNN 7.0.5) with TITAN RTX GPU.

## Training

### Dataset
We select DIV2K and Flickr2K as our training data. If you want to train with DIV2K or Flicklr2K only you could set 
### Setting model hyperparammeter in ./configs

Basic training and testing configs could be adjusted in configs/plain.json.

## Run
You could change param of --opt to select training model and config file.
> CUDA_VISIBLE_DEVICES=0 python main.py --opt wfpn





