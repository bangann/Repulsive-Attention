## Repulsive Attention: Rethinking Multi-head Attention as Bayesian Inference
This is the implementation for our paper [Repulsive Attention: 
Rethinking Multi-head Attention as Bayesian Inference](https://www.aclweb.org/anthology/2020.emnlp-main.17.pdf).

In this repository, we provide code and pretrained model for ELectra experiment in our paper.



## Introduction

The neural attention mechanism plays an important role in many natural
language processing applications. In particular, the use of multi-head 
attention extends single-head attention by allowing a model to jointly
attend information from different perspectives. 
we provide a novel understanding of multi-head attention from a Bayesian inference perspective. 

Based on the recently developed particle-optimization sampling techniques,
we propose a approach that explicitly improves the repulsiveness
in multi-head attention and consequently strengthens model's expressive power.

## Citation
If you find this useful in your research, please consider citing:

    @inproceedings{an2020repulsive,
    title={Repulsive Attention: Rethinking Multi-head Attention as Bayesian Inference},
    author={An, Bang and Lyu, Jie and Wang, Zhenyi and Li, Chunyuan and Hu, Changwei and Tan, Fei and Zhang, Ruiyi and Hu, Yifan and Chen, Changyou},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    pages={236--255},
    year={2020}}
