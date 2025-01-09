#!/bin/bash

# 模型路径
MODEL_PATH="/home/user/Desktop/MACE-code/experiment/MACE_model_0106.model"
# 输出目录（每个文件单独一个输出目录，以免互相覆盖）
OUTPUT_DIR="/home/user/Desktop/MACE-code/data/LixC12/raw/output_ibrion2"
# 选择优化器，这里示例使用 CG
OPTIMIZER="CG"

# 设置最大力收敛标准
FMAX=0.01

# 将数字从 02 遍历到 20（带零填充），例如 "02" "03" "04" ... "20"
for i in $(seq -w 2 20); do
    # 拼接构型文件路径
    INIT_STRUCTURE_PATH="/home/user/Desktop/MACE-code/data/LixC12/raw/dataset_ibrion2/dataset/${i}_AA.extxyz"

    # 调用 Python 脚本进行结构优化
    python3 /home/user/Desktop/MACE-code/mace4LiC/structure_optimize.py \
        --model_path "$MODEL_PATH" \
        --init_structure_path "$INIT_STRUCTURE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --optimizer "$OPTIMIZER" \
        --fmax "$FMAX" \
        --constrain_c_atoms

    # 可选：执行完单个文件后，打印提示信息
    echo "Finished optimizing structure for ${i}_AB.extxyz"
done
