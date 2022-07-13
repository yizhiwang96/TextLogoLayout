# TextLogoLayout

This is the official Pytorch implementation of the paper:

Aesthetic Text Logo Synthesis via Content-aware Layout Inferring. CVPR 2022.

Paper: [arxiv](https://arxiv.org/abs/2204.02701)
Supplementary: [link](./dataset/intro/CVPR_22_textlogolayout_SM.pdf)

## Demo
Our model takes glyph images and their corresponding texts as input and synthesizes aesthetic layouts for them automatically.

English Results:
<div align=center>
	<img src="dataset/intro/demo_eng_res.jpg"> 
</div>

Chinese Results:
<div align=center>
	<img src="dataset/intro/demo_chn_res.jpg"> 
</div>

## Dataset
### TextLogo3K Dataset
We construct a text logo dataset named as TextLogo3K by collecting data from Tencent Video, one of the leading online video platforms in China.
The dataset consists of 3,470 carefully-selected text logo images that are extracted from the posters/covers of the movies, TV series, and comics. 

<div align=center>
	<img src="dataset/intro/textlogo3k_logos.jpg"> 
</div>

We manually annotate the bounding box, pixel-level mask, and category for each character in those text logos.

<div align=center>
	<img src="dataset/intro/textlogo3k_annos.jpg"> 
</div>

Download link: [Google Drive](https://drive.google.com/drive/folders/1FofGxAbpXp2Jjfz-mROsqwpOvL8SKpuE?usp=sharing), [PKU Disk](https://disk.pku.edu.cn:443/link/7201CADEA4E0A3B977D71228B5CCABE8) (Password: 1VEn)

Please download it, unzip it, and put the folder 'TextLogo3K' under './dataset/'.

**Please note that this dataset CAN ONLY be used for academic purpose.**

In addition to the layout synthesis problem addressed in our paper, our dataset can also be adopted in many tasks, such as (1) **text detection/segmentation**, (2) **texture transfer**, (3) **artistic text recognition**, and (4) **artistic font generation**.

### English Dataset
The English dataset we used is from TextSeg (Rethinking Text Segmentation: A Novel Dataset and A Text-Specific Refinement Approach, CVPR 2021).
Please follow the instructions in its [homepage](https://github.com/SHI-Labs/Rethinking-Text-Segmentation) to request the dataset.

We will update the preprocessing script for English Dataset soon, some details about the implementation:
(1) In the English dataset, we view a word as an element, which is resized into 64x128 (heightxwidth).
(2) We utilize the word embeddings from [GloVe](https://github.com/stanfordnlp/GloVe).
(3) The hyper-parameter `loss_ol_w` in `options.py` is set according to the experimental results. Some results demonstrated in our paper are from the setting of `loss_ol_w` = `5`.

## Installation

### Requirement

- **python 3.8**
- **Pytorch 1.9.0** (it may work on some lower or higher versions, but not tested)

Please use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to build the environment:
```shell
conda create -n tll python=3.8
source activate tll
```
Install pytorch via the [instructions](https://pytorch.org/get-started/locally/).
- Others
```shell
conda install tensorboardX scikit-image jieba
```

## Training and Testing

### Training 
To train our model:
```shell
python train.py --experiment_name base_model 
```
The training log will be written in `./experiments/base_model/logs`, which can be visualized by Tensorboard.
The checkpoints will be saved in `./experiments/base_model/checkpoints`.
All hyper-parameters can be found in `options.py`.

Our code supports multi-gpu training, if your single GPU's memory is not enough, check `multi_gpu` in `options.py` is `True` and run:
```shell
CUDA_VISIBLE_DEVICES=0,1,2...,n python train.py --experiment_name base_model 
```
### Pretrained models
Our trained checkpoints (at epoch 600) can be found in [Google Drive](https://drive.google.com/drive/folders/1wpfvpv37ja2e5zpUvfU_YT1AQHhaUZN5?usp=sharing) and [PKU Disk](https://disk.pku.edu.cn:443/link/B793615E0997A4B82CC3B74E22C3CAB5). We find checkpoints at different steps may give different styles, it is encouraged to train the model by yourself and test more checkpoints.

### Testing 
To test our model on TextLogo3K testing dataset:
```shell
python test.py --experiment_name base_model --test_sample_times 10 --test_epoch 600
```
The results will be saved in `./experiments/base_model/results`.

### Testing on your own data
(This function is being developed, will be upgraded soon)
To test our model on your own cases:
First, download the Chinese embeddings from [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors), i.e., sgns.baidubaike.bigram-char, put it under './dataset/Embeddings'.

Then, generate the data from input texts and font files:
```shell
python gen_data.py --input_text 你好世界 --ttf_path ./dataset/ttfs/FZShengSKSJW.TTF --output_dir ./dataset/YourDataSet/
```
Last, use our model to infer:
```shell
python test.py --experiment_name base_model --test_sample_times 10 --test_epoch 500 --data_name YourDataSet --mode test
```
The results will be written to `./experiments/base_model/results/500/YourDataSet/`

## Acknowledgment

- [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- [TextSeg](https://github.com/SHI-Labs/Rethinking-Text-Segmentation)

## Citation

If you use this code or find our work is helpful, please consider citing our work:
```
@inproceedings{wang2021aesthetic,
  title={Aesthetic Text Logo Synthesis via Content-aware Layout Inferring},
  author={Wang, Yizhi and Pu, Gu and Luo, Wenhan and Wang, Yexin ans Xiong, Pengfei and Kang, Hongwen and Wang, Zhonghao and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
