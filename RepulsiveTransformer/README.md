# Transformer with Repulsive Attention

This is the code for Transformer with proposed repulsive attention. The code is based on open-sourced [fairseq (v0.6.0)](https://github.com/pytorch/fairseq/tree/v0.6.0). Follow [this link](https://fairseq.readthedocs.io/) for a detailed document about the original code base.


## Requirements
* Python version >= 3.6
* PyTorch version >= 0.4.0
* For more requirements, please refer to [fairseq](https://github.com/pytorch/fairseq/tree/v0.6.0)

To install fairseq:
```
pip install -r requirements.txt
python setup.py build develop
```

## Dataset
Please refer to the [fairseq preprocessing repo](https://github.com/zhuohan123/macaron-net/tree/master/translation/macaron-scripts/data-preprocessing) to get and tokenize the IWSLT14 De-En and WMT14 En-De datasets. 
Or use the data provided in the last section.

## How to run

We trained the Transformer-small model with the IWSLT14 De-En dataset on one GPU, and 
the Transformer-base model with the WMT14 En-De dataset on four GPU.
The scripts for training and testing is located at `rTransformer-scripts` folder. 
```
# IWSLT14 De-En
## To train the model
$ CUDA_VISIBLE_DEVICES=0 ./rTransformer-scripts/train/train-iwslt14-svgd.sh
## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./rTransformer-scripts/test/test-iwslt14.sh $DATA_DIR/checkpoint_best.pt

# WMT14 En-De base
## To train the model
$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./rTransformer-scripts/train/train-wmt14-base-svgd.sh
## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./rTransformer-scripts/test/test-wmt14-base.sh $DATA_DIR/checkpoint_best.pt

```

## Pre-trained Models
We provide the pre-processed datasets and pre-trained models as follows:

Description | Dataset | Model 
---|---|---
Transformer-small-DMA | [IWSLT14 De-En](https://drive.google.com/file/d/1f7SlKFwG4PaVVsZsmQC-LhuX3O1U-H6w/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/file/d/1RwpowMpJYBxrA0-BTqu_cGjdo7JBuquJ/view?usp=sharing)
Transformer-base-DMA | [WMT14 En-De]() | [download (.tbz2)](https://drive.google.com/file/d/1o6BMMo3RMh90JcHb3tOonZywho9XjGeM/view?usp=sharing)