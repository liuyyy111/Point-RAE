#!/usr/bin/env bash
set -x
GPU_DEVICES=$1

CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
    python main.py --config "cfgs/pretrain/pretrain.yaml" \
    --exp_name "exp_name"
