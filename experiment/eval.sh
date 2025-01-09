#!/bin/bash
# eval.sh
# 划分数据集并评估训练集、验证集和测试集

set -e  # 如果任何命令失败，脚本将立即退出

# 设置变量
MODEL="MACE_model_0106.model"  # 或者 "MACE_model_stage2.model"
OUTPUT_DIR="./MACE_model"
SEED=2024

VALID_INDICES_FILE="./valid_indices_${SEED}.txt"
ORIGINAL_TRAIN_FILE="../data/LixC12/dataset/train.extxyz"
TEST_INPUT_FILE="../data/LixC12/dataset/test.extxyz"

TRAIN_SPLIT_FILE="${OUTPUT_DIR}/train_split.extxyz"
VALID_SPLIT_FILE="${OUTPUT_DIR}/valid_split.extxyz"

TRAIN_EVAL_OUTPUT="${OUTPUT_DIR}/infer_train.extxyz"
VALID_EVAL_OUTPUT="${OUTPUT_DIR}/infer_valid.extxyz"
TEST_EVAL_OUTPUT="${OUTPUT_DIR}/infer_test.extxyz"

VALID_FILE_PY="../mace4LiC/valid_file.py"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=== 步骤 1: 划分数据集 ==="

# 运行 valid_file.py 来划分训练集和验证集
python3 "$VALID_FILE_PY" \
    --valid_indices="$VALID_INDICES_FILE" \
    --train_file="$ORIGINAL_TRAIN_FILE" \
    --output_dir="$OUTPUT_DIR"

echo "数据集划分完成："
echo "训练集（去除验证集后的）：$TRAIN_SPLIT_FILE"
echo "验证集：$VALID_SPLIT_FILE"

echo "=== 步骤 2: 评估训练集 ==="
# 评估训练集
mace_eval_configs \
    --configs="$TRAIN_SPLIT_FILE" \
    --model="$MODEL" \
    --output="$TRAIN_EVAL_OUTPUT" \
    --batch_size 16 \
    --device "cuda" \
    --default_dtype "float64"

echo "训练集评估完成，结果保存在 $TRAIN_EVAL_OUTPUT"

echo "=== 步骤 3: 评估验证集 ==="
# 评估验证集
mace_eval_configs \
    --configs="$VALID_SPLIT_FILE" \
    --model="$MODEL" \
    --output="$VALID_EVAL_OUTPUT" \
    --batch_size 1 \
    --device "cuda" \
    --default_dtype "float64"

echo "验证集评估完成，结果保存在 $VALID_EVAL_OUTPUT"

echo "=== 步骤 4: 评估测试集 ==="
# 评估测试集
mace_eval_configs \
    --configs="$TEST_INPUT_FILE" \
    --model="$MODEL" \
    --output="$TEST_EVAL_OUTPUT" \
    --batch_size 1 \
    --device "cuda" \
    --default_dtype "float64"

echo "测试集评估完成，结果保存在 $TEST_EVAL_OUTPUT"

echo "=== 步骤 5: 评估完成 ==="
echo "所有评估结果已生成："
echo "训练集预测：$TRAIN_EVAL_OUTPUT"
echo "验证集预测：$VALID_EVAL_OUTPUT"
echo "测试集预测：$TEST_EVAL_OUTPUT"
