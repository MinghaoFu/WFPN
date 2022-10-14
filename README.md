# Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution

This repository is for WFPN introduced in the following paper

**Minghao Fu**, Dongyang Zhang, Min Lei, Kun He, Changyu Li, and Jie Shao, *Wide Feature Projection with Fast and Memory-Economic Attention for Efficient Image Super-Resolution*, BMVC 2022, [[Paper]](...)

This code is modified from [EDSR(Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch) and tested on Ubuntu 16.04.6 LTS environment (Python 3.7.11, Pytorch 1.7.1, CUDA 9.0, cuDNN 7.0.5) with TITAN RTX GPU.

## Run

### Dataset
We select [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) as our training datasets.

### Setting hyperparameter in configs
Basic training and testing configs could be adjusted in *configs/plain.json*. For specifc model, you could customize it in *configs/{model}.json*.

### Training
You could change --opt to select training model and config file.
> CUDA_VISIBLE_DEVICES=0 python main.py --opt {model}

For instance, if you want to train our WFPN, you only need to set **{model}** as **wfpn**.

### Testing
You just need to set **test_only: true** in *configs/{model}*.json and then run the identical shell.

---
The definations of parameters in *configs* are the same as [EDSR(Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch). Thanks for their pioneering work.




