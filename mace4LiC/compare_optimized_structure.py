import os
import numpy as np
from ase.io import read
from structure_optimize import optimize_structure


def calculate_errors(final_true, final_pred):
    """
    计算最终构型的坐标误差（平均绝对误差）、能量误差（meV/atom）和力的平均绝对误差（MAE）。

    :param final_true: 文件中的最终构型（ASE Atoms 对象）
    :param final_pred: 优化后的最终构型（ASE Atoms 对象）
    """
    # 获取原子数
    num_atoms = len(final_true)

    # 坐标误差（每个原子的位置误差的平均绝对值）
    true_positions = final_true.get_positions()
    pred_positions = final_pred.get_positions()
    position_abs_error = np.abs(true_positions - pred_positions)  # 逐原子的绝对误差
    mean_position_abs_error = np.mean(position_abs_error)  # 平均绝对误差

    # 能量误差（meV/atom）
    true_energy = final_true.get_potential_energy()
    pred_energy = final_pred.get_potential_energy()
    energy_error = abs(pred_energy - true_energy) * 1000 / num_atoms  # 转换为 meV/atom

    # 力的平均绝对误差（MAE，eV/Å）
    true_forces = final_true.get_forces()
    pred_forces = final_pred.get_forces()
    force_mae = np.mean(np.abs(true_forces - pred_forces)) * 1000

    # 输出结果
    print("=== 比较结果 ===")
    print(f"原子坐标的平均绝对误差: {mean_position_abs_error:.6f} Å")
    print(f"能量误差: {energy_error:.6f} meV/atom")
    print(f"力的平均绝对误差: {force_mae:.6f} meV/Å")


def main():
    # 参数配置
    model_path = "/home/mect-a/桌面/MACE-code/experiment/parameter_test_20241212.model"  # 替换为实际的 MACE 模型路径
    init_structure_path = "/home/mect-a/桌面/MACE-code/data/LixC12/raw/dataset_ibrion2/dataset/19_AA.extxyz"  # 初始数据文件
    output_dir = "/home/mect-a/桌面/MACE-code/structoptimize/optimizedcell"  # 输出文件夹

    # 读取初始文件（如 02_AB.extxyz）
    all_structures = read(init_structure_path, index=":")  # 读取文件中所有构型
    initial_conf = all_structures[0]  # 初始构型
    final_true_conf = all_structures[-1]  # 文件中的最终构型

    # 优化初始构型
    print("开始结构优化...")
    optimize_structure(
        model_path=model_path,
        init_structure_path=init_structure_path,
        output_dir=output_dir,
        optimizer="LBFGS",  # 可根据需要修改
        fmax=0.01,  # 最大力收敛条件
        constrain_c_atoms=True  # 是否限制碳原子
    )

    # 读取优化后的最终构型
    optimized_structure_path = os.path.join(output_dir, "optimizedcell", "19_AA_LBFGS_relaxed.extxyz")
    final_pred_conf = read(optimized_structure_path)

    # 比较优化结果与文件中的最终构型
    print("比较优化结果与参考构型...")
    calculate_errors(final_true_conf, final_pred_conf)


if __name__ == "__main__":
    main()