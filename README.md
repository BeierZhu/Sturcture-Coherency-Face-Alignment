# Structure-Coherent Deep Feature Learning for Robust Face Alignment
This repository contains the code for "Structure-Coherent Deep Feature Learning for Robust Face Alignment"  

# Introduction
We propose a structure-coherent deep feature learning method for face alignment. Unlike most existing face alignment methods which overlook the facial structure cues, we explicitly exploit the relation among facial landmarks to make the detector robust to hard cases such as occlusion and large pose. Specifically, we leverage a landmark-graph relational network to enforce the structural relationships among landmarks. We consider the facial landmarks as structural graph nodes and carefully design the neighborhood to passing features among the most related nodes. Our method dynamically adapts the weights of node neighborhood to eliminate distracted information from noisy nodes, such as occluded landmark point. Moreover, different from most previous works which only tend to penalize the landmarks absolute position during the training, we propose a relative location loss to enhance the information of relative location of landmarks. This relative location supervision further regularizes the facial structure. Our approach considers the inter- actions among facial landmarks and can be easily implemented on top of any convolutional backbone to boost the performance. Extensive experiments on three popular benchmarks, including WFLW, COFW and 300W, demonstrate the effectiveness of the proposed method. In particular, due to explicit structure modeling, our approach is especially robust to challenging cases resulting in impressive low failure rate on COFW and WFLW datasets. 

# Requirements

* python >= 3.6 
* pytorch >=1.0.1

# Before Start

We provide processed WFLW dataset here [Google Drive](https://drive.google.com/drive/folders/1WKRgeqz8I3blqq7V49VarQpuQGofzfIS?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1j8HQxgU4ActNptfISUCNNw) (pwd: hzwf). Upzip download files.  
Remember to change the `train_root`, `train_source`, `val_root` and `val_source` 
to your image directory and annotation file, repectively.

Acknowledgement: we use the code from [PFLD](https://github.com/guoqiangqi/PFLD/blob/master/data/SetPreparation.py) to preprare the WLFW dataset.


# Train from scratch
To train the L-GRN with ResNet34 Backbone

```bash
python main.py --config experiments/WLFW/config_resnet.yaml --expname experiments/WLFW/exp_resnet --train
```
To compare with the fully-connect layer with ResNet34 Backbone

```bash
python main.py --config experiments/WLFW/config_resnet_fc.yaml --expname experiments/WLFW/exp_resnet_fc/ --train
```

# Evaluate
Here's an example to evaluate L-GRN with ResNet34 Backbone

```bash
python main.py --config experiments/WLFW/config_resnet.yaml --expname experiments/WLFW/exp_resnet/ --load_path experiments/WLFW/exp_resnet/checkpoint_best.pth.tar --evaluate
```

# Citation 
If you find our paper or this project helps your research, please kindly consider citing our paper in your publications.

```bash
@ARTICLE{9442331,
  author={Lin, Chunze and Zhu, Beier and Wang, Quan and Liao, Renjie and Qian, Chen and Lu, Jiwen and Zhou, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Structure-Coherent Deep Feature Learning for Robust Face Alignment}, 
  year={2021},
  volume={30},
  number={},
  pages={5313-5326},
  doi={10.1109/TIP.2021.3082319}}
```
