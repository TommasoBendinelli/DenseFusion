#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train.py --dataset tommaso\
  --dataset_root ./datasets/tommaso/tommaso_preprocessed