## Repulsive Attention: Rethinking Multi-head Attention as Bayesian Inference
This is the implementation for our paper "Repulsive Attention: 
Rethinking Multi-head Attention as Bayesian Inference" 

In this repository, we provide codes and pretrained models for our experiments containing 
following four tasks. Please refer to sub-directories for detailed instructions.

* **Self-attentive Sentence Classification**
* **Transformer-based Neural Translation**
* **Language Representation Learning**
* **Graph-to-Text Generation**




## Introduction

The neural attention mechanism plays an important role in many natural
language processing applications. In particular, the use of multi-head 
attention extends single-head attention by allowing a model to jointly
attend information from different perspectives. 
we provide a novel understanding of multi-head attention from a Bayesian inference perspective. 

Based on the recently developed particle-optimization sampling techniques,
we propose a non-parametric approach that explicitly improves the diversity
in multi-head attention and consequently strengthens model's expressive power.
