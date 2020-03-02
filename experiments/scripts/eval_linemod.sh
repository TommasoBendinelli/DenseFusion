#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model trained_models/linemod/pose_model_9_0.012956139583687484.pth\
  --refine_model trained_models/linemod/pose_refine_model_95_0.007274364822843561.pth