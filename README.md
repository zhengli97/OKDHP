# Online Knowledge Distillation for Efficient Pose Estimation (OKDHP)

This is the pytorch implementation for the OKDHP.
This repository is extended based on [FPD](https://github.com/yuanyuanli85/Fast_Human_Pose_Estimation_Pytorch).

Email: lizheng1@stu.hznu.edu.cn

## Abstract

In this work, we investigate a novel Online Knowledge Distillation 
framework by distilling Human Pose structure knowledge in a 
one-stage manner to guarantee the distillation efficiency, termed OKDHP.
Specifically, OKDHP trains a single multi-branch network
and acquires the predicted heatmaps from each, which are
then assembled by a Feature Aggregation Unit (FAU) as
the target heatmaps to teach each branch in reverse. 
Instead of simply averaging the heatmaps, FAU which consists
of multiple parallel transformations with different receptive
fields, leverages the multi-scale information, thus obtains
target heatmaps with higher-quality. Specifically, the pixelwise 
Kullback-Leibler (KL) divergence is utilized to minimize the
discrepancy between the target heatmaps and the
predicted ones, which enables the student network to learn
the implicit keypoint relationship. Besides, an unbalanced
OKDHP scheme is introduced to customize the student networks 
with different compression rates. The effectiveness of
our approach is demonstrated by extensive experiments on
two common benchmark datasets, MPII and COCO.

## Training 

In this code, you can reproduce the experiment results in the paper, including MPII and COCO.

- Running 4-Stack OKDHP with three branches on MPII dataset. 

(Running based on one NVIDIA RTX 3090 GPU)

~~~
sh start.sh
~~~
