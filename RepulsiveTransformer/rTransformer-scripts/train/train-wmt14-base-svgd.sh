#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH

model=transformer
PROBLEM=wmt14_en_de
ARCH=transformer_wmt_en_de
DATA_PATH=data-bin/wmt14_en_de_joined_dict/
OUTPUT_PATH=log/$PROBLEM/$ARCH/svgd_0.01_0.1
mkdir -p $OUTPUT_PATH

# Assume training on 4 P40 GPUs. Change the --max-tokens and --update-freq to match your hardware settings.

nohup python -u train.py $DATA_PATH \
  --seed 1 \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr 0.0007 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
  --max-tokens 4096 --save-dir $OUTPUT_PATH \
  --update-freq 4 --no-progress-bar --log-interval 50 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 10 --save-interval 10 \
  --bayesian_method svgd --d_kernel_weight 0.01 --stepsize 0.1 \
| tee -a $OUTPUT_PATH/train_log.txt &

