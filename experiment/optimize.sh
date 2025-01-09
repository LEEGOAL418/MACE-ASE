#!/bin/bash

# 设置模型路径、初始结构路径、输出目录等变量
MODEL_PATH="/home/user/Desktop/MACE-ASE/experiment/MACE_model_0106.model"
INIT_STRUCTURE_PATH="/home/user/Desktop/MACE-ASE/structoptimize/initialcell/02_AB.extxyz"
OUTPUT_DIR="/home/user/Desktop/MACE-ASE/structoptimize"

# 选择优化器，这里使用 LBFGS 作为默认优化器
OPTIMIZER="CG"

# 设置最大力收敛标准
FMAX=0.01

# 调用 Python 脚本进行结构优化
python3 ../mace4LiC/structure_optimize.py \
    --model_path "$MODEL_PATH" \
    --init_structure_path "$INIT_STRUCTURE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --optimizer "$OPTIMIZER" \
    --fmax "$FMAX" \
    --constrain_c_atoms
