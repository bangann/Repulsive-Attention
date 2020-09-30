# Self-attentive Sentence Classification with Repulsive Attention
This is the code for Self-attentive Sentence Classification with Bayesian Repulsive Attention modeling. It is also one of the experiments of our paper "Repulsive Attention: Rethinking Multi-head Attention as Bayesian Inference".

## Requirements and Installation
This project is based on pytorch. Before inplementation, please make sure you have
the following dependencies.

* [PyTorch](http://pytorch.org/) version == 1.0.0
* Python version >= 3.6
* torchtext==0.4.0
* torchvision==0.2.1


If you have problems with spacy.en-core-web-sm, please try the following comand:

```
$ python -m spacy download en
```


## How to run
```
python train_svgd.py --corpus=yelp --attention_hops=30 --bayesian_method=svgd --optimizer=sgd --learning_rate=0.06 --alpha=1 --stepsize=1 --save_dir_name=yelp_svgd


python train_svgd.py --corpus=age --attention_hops=30 --bayesian_method=svgd --fc_dim=2000 --alpha=0.01 --stepsize=1 --save_dir_name=age_svgd


python train_svgd.py --corpus=snli --attention_hops=30 --bayesian_method=svgd --fc_dim=4000 --attention_unit=150 --alpha=0.005 --stepsize=1 --save_dir_name=snli_svgd

```
