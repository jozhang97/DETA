#!/usr/bin/env bash

set -x

EXP_DIR=exps/public/deta_swin_ft
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine --two_stage \
    --num_feature_levels 5 --num_queries 900 \
    --dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 \
    --assign_first_stage --assign_second_stage \
    --epochs 24 --lr_drop 20 \
    --lr 5e-5 --lr_backbone 5e-6 --batch_size 1 \
    --backbone swin \
    --bigger \
    ${PY_ARGS}
