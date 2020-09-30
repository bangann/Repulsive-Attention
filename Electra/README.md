# ELECTRA with Repulsive Attention

This is the code for ELECTRA with the proposed repulsive attention. The code is modified 
form the official [ELECTRA](https://github.com/google-research/electra) repo. Please refer to it for 
more details.

## Requirements
* Python 3
* [TensorFlow](https://www.tensorflow.org/) 1.15 
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/) and [SciPy](https://www.scipy.org/) 


## Dataset

We use the [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) released by Aaron Gokaslan and Vanya Cohen in our paper. 
Please refer to [ELECTRA](https://github.com/google-research/electra) for data processing details.

## How to run
To pretrain a ELECTRA model with diversified attention:
```
python3 run_pretraining_svgd.py --data-dir $DATA_DIR --model-name <NAME> --hparams '{"stepsize": 0.01, "d_kernel_weight": 0.1}' 
```
To fin-tune a pretrained model on [GLUE](https://gluebenchmark.com/), please follow [ELECTRA](https://github.com/google-research/electra):
```
python3 run_finetuning.py --data-dir $DATA_DIR  --model-name <NAME> --hparams '{"model_size": "small", "task_names": ["cola"], "num_trials": 50}'
```

## Pretrained Models

We provide the pre-trained models as follow:

| Model | Download |
| --- | --- |
| ELECTRA-Small-DMA | [link](https://drive.google.com/file/d/1dYQazMD06bLKGan-D8vwbXAnlU8pIJ0R/view?usp=sharing)|
