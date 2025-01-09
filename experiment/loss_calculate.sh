#!/bin/bash

# 赋值给变量
TRUE_DATASET="../data/LixC12/dataset/test.extxyz"
PREDICTED_DATASET="../experiment/tests/MACE_model/infer_test.extxyz"
OUTPUT_DIR="../experiment/plot/loss_distribution"
ENERGY_WEIGHT=1.0
FORCES_WEIGHT=100.0

# 运行 Python 脚本
python3 ../mace4LiC/loss_out.py \
    --true_dataset "$TRUE_DATASET" \
    --predicted_dataset "$PREDICTED_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --energy_weight "$ENERGY_WEIGHT" \
    --forces_weight "$FORCES_WEIGHT"
