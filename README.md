# [Adversarial Learning For Semi-Supervised Semantic Segmentation](https://openreview.net/references/pdf?id=HkTgsG-CW "Open Review")
## Introduction
This is a submission for ICLR 2018 Reproducibility Challenge. The central theme of the work by the authors is to incorporate adversarial training for semantic-segmentation task which enables the segmentation-network to learn in a semi-supervised fashion on top of the traditional supervised learning. The authors claim significant improvement in the performance (measured in terms of mean IoU) of segmentation network after the supervised-training is extended with adversarial and semi-supervised training.

## Scope
 My plan is to reproduce the improvement in the performance of the segmentation network (Resnet-101) by including adversarial and semi-supervised learning scheme over the baseline supervised training and document my experience along the way. The authors have used two datasets, PASCAL VOC 12 (extended version) and Cityscapes, to demonstrate  the benefits of their proposed training scheme. I will focus on PASCAL VOC 12 dataset for this work. Specifically, the target for this work is to reproduce the following table from the paper.

 | Method | &emsp; &emsp; &emsp; Data Amount <br> 1/2 &emsp; &emsp; &emsp; full |
 | --- | --- |
 | Baseline (Resnet-101) | 69.8 &emsp; &emsp; &emsp;73.6  |
 |Baseline + Adversarial Training| 72.6 &emsp; &emsp; &nbsp;  74.9|
 |Baseline + Adversarial Training + <br> Semi-supervised Learning|73.2 &emsp; &emsp; &nbsp;  NA|

## Results Reproduced
Following table summarizes the results I have been able to reproduce for the full dataset. For the full dataset, only the performance of the adversarial training on top of baseline can be evaluated.


| Method (Full Dataset) | Original | Challenge |
| --- | --- | --- |
| Baseline (Resnet-101) | 73.6  | 69.98 |
|Baseline + Adversarial Training|  74.9| 70.97 |
|Baseline + Adversarial Training + <br> Semi-supervised Learning|NA| NA|

Following table summarized the results that I was able to reproduce for the semi-supervised training where half of the training data is reserved for semi-supervised training with unlabeled data. 

| Method (1/2 Dataset) | Original | Challenge |
| --- | --- | --- |
| Baseline (Resnet-101) | 69.8  | 67.84 |
|Baseline + Adversarial Training|  72.6| 68.89 |
|Baseline + Adversarial Training + <br> Semi-supervised Learning|73.2| 69.05|

|With Densenet and UperNet, we get better results|
