#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from ase.io import read
import pandas as pd  # 用于保存 CSV 文件

# -------------------- 必要的自定义损失函数及辅助函数（参考loss.py） -------------------- #

def weighted_mean_squared_error_energy(ref, pred):
    """
    计算能量的加权均方误差。
    
    参数：
        ref (dict): 包含 'energy'、'weight' 和 'num_atoms' 的参考值。
        pred (dict): 包含 'energy' 的预测值。
        
    返回：
        torch.Tensor: 加权能量损失。
    """
    energy_diff = ref["energy"] - pred["energy"]
    weighted_energy_diff = ref["weight"] * energy_diff / ref["num_atoms"]
    return torch.mean(torch.square(weighted_energy_diff))

def mean_squared_error_forces(ref, pred):
    """
    计算力的均方误差。
    
    参数：
        ref (dict): 包含 'forces' 的参考值。
        pred (dict): 包含 'forces' 的预测值。
        
    返回：
        torch.Tensor: 力损失。
    """
    forces_diff = ref["forces"] - pred["forces"]
    return torch.mean(torch.square(forces_diff))

class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=100.0) -> None:
        super().__init__()
        self.energy_weight = torch.tensor(energy_weight, dtype=torch.get_default_dtype())
        self.forces_weight = torch.tensor(forces_weight, dtype=torch.get_default_dtype())

    def forward(self, ref: dict, pred: dict) -> tuple:
        """
        计算总损失、能量损失和力损失。
        
        参数：
            ref (dict): 包含 'energy'、'forces'、'weight' 和 'num_atoms' 的参考值。
            pred (dict): 包含 'energy' 和 'forces' 的预测值。
            
        返回：
            tuple: (总损失, 能量损失, 力损失)
        """
        energy_loss = weighted_mean_squared_error_energy(ref, pred)
        forces_loss = mean_squared_error_forces(ref, pred)
        total_loss = self.energy_weight * energy_loss + self.forces_weight * forces_loss
        return total_loss, energy_loss, forces_loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item():.3f}, "
            f"forces_weight={self.forces_weight.item():.3f})"
        )

# -------------------- 处理计算结果 -------------------- #

def compute_loss(ref, pred, loss_fn):
    """
    使用损失函数计算所有的损失部分。
    
    参数：
        ref (dict): 包含参考能量、力、权重和原子数的字典。
        pred (dict): 包含预测能量和力的字典。
        loss_fn: 损失函数对象。
        
    返回：
        tuple: (能量损失的标量值, 总损失的标量值, 力损失的标量值)
    """
    total_loss, energy_loss, forces_loss = loss_fn(ref, pred)
    return total_loss.item(), energy_loss.item(), forces_loss.item()

# -------------------- 参数解析 -------------------- #

def parse_arguments():
    """
    解析命令行参数。
    
    返回：
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(description='Compute and plot loss metrics for atomic structures.')
    
    parser.add_argument('--true_dataset', type=str, required=True,
                        help='Path to the true dataset (.extxyz file).')
    parser.add_argument('--predicted_dataset', type=str, required=True,
                        help='Path to the predicted dataset (.extxyz file).')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the CSV file and plots. Defaults to the current directory.')
    parser.add_argument('--energy_weight', type=float, default=1.0,
                        help='Weight for the energy loss component. Default is 1.0.')
    parser.add_argument('--forces_weight', type=float, default=100.0,
                        help='Weight for the forces loss component. Default is 100.0.')
    
    return parser.parse_args()

# -------------------- 主函数 -------------------- #

def main():
    args = parse_arguments()
    
    true_dataset_path = args.true_dataset
    predicted_dataset_path = args.predicted_dataset
    output_dir = args.output_dir
    energy_weight = args.energy_weight
    forces_weight = args.forces_weight
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化损失函数
    loss_fn = WeightedEnergyForcesLoss(energy_weight, forces_weight)
    
    # 读取真实数据集
    print("Reading true dataset from:", true_dataset_path)
    true_atoms_list = read(true_dataset_path, index=':', format='extxyz')
    print(f"Number of structures in the true dataset: {len(true_atoms_list)}")
    
    # 读取预测数据集
    print("Reading predicted dataset from:", predicted_dataset_path)
    pred_atoms_list = read(predicted_dataset_path, index=':', format='extxyz')
    print(f"Number of structures in the predicted dataset: {len(pred_atoms_list)}")
    
    # 检查两个数据集的结构数量是否一致
    if len(true_atoms_list) != len(pred_atoms_list):
        raise ValueError("The number of structures in the true and predicted datasets do not match.")
    
    # 用于存储结果的列表
    atom_counts = []
    total_losses = []
    energy_losses = []
    forces_losses = []
    
    # 同时遍历真实和预测数据集
    for idx, (true_atoms, pred_atoms) in enumerate(zip(true_atoms_list, pred_atoms_list)):
        # 获取当前结构的原子数
        num_atoms = len(true_atoms)
        
        # 提取参考能量和力，并转换为默认数据类型（通常是 float32）
        ref_energy = torch.tensor(true_atoms.get_potential_energy(), dtype=torch.get_default_dtype())
        ref_forces = torch.tensor(true_atoms.get_forces(), dtype=torch.get_default_dtype())
        
        # 提取预测能量和力
        # 预测的能量存储在 pred_atoms.info['MACE_energy']
        # 力存储在 pred_atoms.arrays['MACE_forces']
        pred_energy = torch.tensor(pred_atoms.info['MACE_energy'], dtype=torch.get_default_dtype())
        pred_forces = torch.tensor(pred_atoms.arrays['MACE_forces'], dtype=torch.get_default_dtype())
        
        # 准备参考和预测的字典，包括权重和原子数
        ref = {
            'energy': ref_energy,
            'forces': ref_forces,
            'weight': torch.tensor(1.0, dtype=torch.get_default_dtype()),  # 默认权重设为1.0
            'num_atoms': torch.tensor(num_atoms, dtype=torch.get_default_dtype())  # 原子数
        }
        pred = {
            'energy': pred_energy,
            'forces': pred_forces
        }
        
        # 计算所有损失（输出标量值！）
        total_loss, energy_loss, forces_loss = compute_loss(ref, pred, loss_fn)
        
        # 存储结果
        atom_counts.append(num_atoms)
        total_losses.append(total_loss)
        energy_losses.append(energy_loss)
        forces_losses.append(forces_loss)
        
        # 每处理100个结构或最后一个结构时打印进度
        if (idx + 1) % 100 == 0 or (idx + 1) == len(true_atoms_list):
            print(f"Processed {idx + 1}/{len(true_atoms_list)} structures.")
    
    # 将列表转换为 NumPy 数组，便于后续处理
    atom_counts = np.array(atom_counts)
    total_losses = np.array(total_losses)
    energy_losses = np.array(energy_losses)
    forces_losses = np.array(forces_losses)
    
    # -------------------- 保存结果为 CSV 文件 -------------------- #
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'Number_of_Atoms': atom_counts,
        'Total_Loss': total_losses,
        'Energy_Loss': energy_losses,
        'Forces_Loss': forces_losses
    })
    
    # 保存为 CSV 文件，保存在输出目录下
    csv_output_path = os.path.join(output_dir, 'loss_results.csv')
    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')  # 使用 'utf-8-sig' 以确保 Excel 正确识别编码
    print(f"Results have been saved to {csv_output_path}")
    
    # -------------------- 绘图 -------------------- #
    
    # 定义损失类型及其对应的数据
    loss_types = {
        'Total_Loss': total_losses,
        'Energy_Loss': energy_losses,
        'Forces_Loss': forces_losses
    }
    
    # 为每种损失类型创建并保存散点图
    for loss_name, loss_values in loss_types.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(atom_counts, loss_values, alpha=0.6, label=loss_name.replace('_', ' '))
        plt.xlabel('Number of Atoms')
        plt.ylabel(loss_name.replace('_', ' '))
        plt.title(f'{loss_name.replace("_", " ")} vs. Number of Atoms')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # 保存为 PNG 格式，保存在输出目录下
        png_path = os.path.join(output_dir, f'{loss_name}.png')
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Plot saved as {png_path}")

if __name__ == "__main__":
    main()
