# CPSL: Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation (CVPR 2022, official Pytorch implementation)

![Teaser](docs/overview.png)

### [Paper](https://arxiv.org/abs/2203.09744)

## Abstract
>Domain adaptive semantic segmentation aims to learn a model with the supervision of source domain data, and produce satisfactory dense predictions on unlabeled target domain. One popular solution to this challenging task is self-training, which selects high-scoring predictions on target samples as pseudo labels for training. However, the produced pseudo labels often contain much noise because the model is biased to source domain as well as majority categories. To address the above issues, we propose to directly explore the intrinsic pixel distributions of target domain data, instead of heavily relying on the source domain. Specifically, we simultaneously cluster pixels and rectify pseudo labels with the obtained cluster assignments. This process is done in an online fashion so that pseudo labels could co-evolve with the segmentation model without extra training rounds. To overcome the class imbalance problem on long-tailed categories, we employ a distribution alignment technique to enforce the marginal class distribution of cluster assignments to be close to that of pseudo labels. The proposed method, namely Class-balanced Pixel-level Self-Labeling (CPSL), improves the segmentation performance on target domain over state-of-the-arts by a large margin, especially on long-tailed categories.

## Installation
Install dependencies:
```bash
pip install -r requirements.txt
```
## Data Preparation 
Download [Cityscapes](https://www.cityscapes-dataset.com/), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/downloads/).

We expect dataset folder to be like
```
└── dataset
    ├── cityscapes
    │   ├── annotations
    │   ├── gtFine
    │   └── leftImg8bit
    ├── GTA5
    │   ├── images
    │   ├── labels
    │   └── split.mat
    ├── SYNTHIA
    │   ├── GT
    │   ├── RGB
    └── └── meta.json

```

## Pretrained Models
| backbone  | warmed-up models (41.4 mIoU) | Stage1 self-labeling (55.5 mIoU) |  Stage2 KD-1 (59.4 mIoU) | Stage3 KD-2 (60.8 mIoU) |
|:----------:|:------:|:----:|:----:|:----:|
| ResNet101 | [model] | [model](https://drive.google.com/file/d/1NMcdUUBzwdhosGdiIFJZUll_p9nQ-i5O/view?usp=sharing) | [model](https://drive.google.com/file/d/1OkKjlDRRzOYLWUETHLo55YMqKf0O8ySy/view?usp=sharing) | [model](https://drive.google.com/file/d/1ImTM4aBk0STMXCmJiQs1HWmEBKeL4xs4/view?usp=sharing) | 



## Traning




