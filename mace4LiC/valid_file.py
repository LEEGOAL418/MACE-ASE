#!/usr/bin/env python3
# valid_file.py

import os
import sys
import argparse
from ase.io import read, write
import numpy as np

def read_valid_indices(file_path):
    try:
        with open(file_path, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        return indices
    except Exception as e:
        print(f"Error reading valid indices from {file_path}: {e}")
        sys.exit(1)

def split_train_valid(train_file, valid_indices, output_dir):
    try:
        # 读取所有训练集构型
        train_structures = read(train_file, index=':')
        total_structures = len(train_structures)
        print(f"Total structures in original training set: {total_structures}")

        # 验证索引的有效性
        valid_indices = np.array(valid_indices)
        if valid_indices.size == 0:
            print("Warning: No valid indices found in the validation indices file.")
            # 全部作为训练集
            write(os.path.join(output_dir, 'train_split.extxyz'), train_structures, format='extxyz')
            print(f"All structures are assigned to the training set.")
            return
        if valid_indices.max() >= total_structures or valid_indices.min() < 0:
            print("Error: Some indices in valid_indices_2024.txt are out of bounds.")
            sys.exit(1)

        # 提取验证集构型
        valid_structures = [train_structures[i] for i in valid_indices]
        print(f"Extracted {len(valid_structures)} structures for validation set.")

        # 提取训练集构型（去除验证集）
        all_indices = set(range(total_structures))
        training_indices = sorted(all_indices - set(valid_indices))
        training_structures = [train_structures[i] for i in training_indices]
        print(f"Remaining {len(training_structures)} structures assigned to the training set.")

        # 保存划分后的训练集和验证集
        train_split_path = os.path.join(output_dir, 'train_split.extxyz')
        valid_split_path = os.path.join(output_dir, 'valid_split.extxyz')

        write(train_split_path, training_structures, format='extxyz')
        write(valid_split_path, valid_structures, format='extxyz')

        print(f"Training split saved to {train_split_path}")
        print(f"Validation split saved to {valid_split_path}")

    except Exception as e:
        print(f"Error splitting train and valid sets: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="划分训练集和验证集，并保存到指定路径。")
    parser.add_argument('--valid_indices', type=str, required=True, help="验证集索引文件路径（valid_indices_2024.txt）")
    parser.add_argument('--train_file', type=str, required=True, help="原始训练集文件路径（train.extxyz）")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录路径，用于保存划分后的训练集和验证集")
    
    args = parser.parse_args()
    
    # 定义输出文件路径
    train_split_path = os.path.join(args.output_dir, 'train_split.extxyz')
    valid_split_path = os.path.join(args.output_dir, 'valid_split.extxyz')
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.valid_indices):
        print(f"Error: Validation indices file not found: {args.valid_indices}")
        sys.exit(1)
    if not os.path.isfile(args.train_file):
        print(f"Error: Original training set file not found: {args.train_file}")
        sys.exit(1)
    
    # 读取验证集索引
    valid_indices = read_valid_indices(args.valid_indices)
    print(f"Number of validation indices read: {len(valid_indices)}")
    
    # 划分训练集和验证集
    split_train_valid(args.train_file, valid_indices, args.output_dir)

if __name__ == "__main__":
    main()
