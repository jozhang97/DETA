#!/usr/bin/env bash

set -x

EXP_DIR=exps/public/improved_baseline
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine --two_stage \
    --num_feature_levels 5 --num_queries 900 \
    --dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
    --epochs 50 --lr_drop 40 \
    ${PY_ARGS}
