# Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution

This repository is for WFPN introduced in the following paper

**Minghao Fu**, Dongyang Zhang, Min Lei, Kun He, Changyu Li, and Jie Shao, *Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution*, BMVC 2022, [[Paper]](https://bmvc2022.mpi-inf.mpg.de/0615.pdf)

This code is tested on Ubuntu 16.04.6 LTS environment (Python 3.7.11, Pytorch 1.7.1, CUDA 9.0, cuDNN 7.0.5) with TITAN RTX GPU.

## Run

### Dataset
We select [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) as our training datasets.

### Configs
Basic training and testing configs could be adjusted in *configs/plain.json*. For specifc model, you could customize it in *configs/{model}.json*. 



