# Graph-to-Text Generation with Repulsive Attention

This is the code for Graph-to-Text Generation with the proposed repulsive attention, which is modified from the official source code of the paper, [Text Generation from Knowledge Graphs with Graph Transformers](https://github.com/rikdz/GraphWriter).

## Requirements
* Python version >= 3.6
* PyTorch version >= 0.4.0

Please refer to the [official repo](https://github.com/rikdz/GraphWriter) for detailed reqirments.

## How to run
Training:
```
python train.py  -save <DIR> -bayesian_method svgd -stepsize 0.1 -d_kernel_weigh 0.01 
```
To generate, use:
```
python generator.py -save <SAVED MODEL>
```
To evaluate, run:
```
python eval.py <GENERATED TEXTS> <GOLD TARGETS>
```