#!/usr/bin/env python3

import os
import argparse
import numpy as np
from ase.io import read, write
from structure_optimize import optimize_structure
from ase.calculators.vasp import Vasp
from ase import Atoms

def calculate_errors(final_true: Atoms, final_pred: Atoms):
    """
    计算最终构型的坐标误差（平均绝对误差）、能量归一化误差（eV/atom）和力的最大误差（矩阵中对应位置的最大绝对误差）。

    :param final_true: 文件中的最终构型（ASE Atoms 对象）
    :param final_pred: 优化后的最终构型（ASE Atoms 对象）
    :return: mean_position_abs_error, energy_error_per_atom, max_force_error
    """
    # 获取原子数
    num_atoms = len(final_true)

    # 坐标误差（每个原子的位置误差的平均绝对值）
    true_positions = final_true.get_positions()
    pred_positions = final_pred.get_positions()
    position_abs_error = np.abs(true_positions - pred_positions)  # 逐原子的绝对误差
    mean_position_abs_error = np.mean(position_abs_error)  # 平均绝对误差

    # 能量误差（eV/atom）
    true_energy = final_true.get_potential_energy()
    pred_energy = final_pred.get_potential_energy()
    energy_error_per_atom = abs(pred_energy - true_energy) / num_atoms  # 归一化为 eV/atom

    # 力的最大误差（矩阵中对应位置的最大绝对误差）
    true_forces = final_true.get_forces()
    pred_forces = final_pred.get_forces()
    force_diff = np.abs(true_forces - pred_forces)
    max_force_error = np.max(force_diff)  # 最大绝对误差

    # 输出结果
    print("=== 比较结果 ===")
    print(f"原子坐标的平均绝对误差: {mean_position_abs_error:.6f} Å")
    print(f"能量归一化误差: {energy_error_per_atom:.6f} eV/atom")
    print(f"力的最大绝对误差: {max_force_error:.6f} eV/Å")

    return mean_position_abs_error, energy_error_per_atom, max_force_error

def calculate_vasp_energy(atoms: Atoms, vasp_executable: str = 'vasp_std'):
    """
    使用 ASE 的 VASP 计算器计算单点能量。

    :param atoms: ASE Atoms 对象
    :param vasp_executable: VASP 执行命令，默认为 'vasp_std'
    :return: 计算得到的能量（eV）
    """
    # 配置 VASP 计算器
    calc = Vasp(
        xc='PBE',              # 交换-相关函数
        encut=400,             # 截断能量
        ismear=0,              # 填充方法
        sigma=0.05,            # 电子态填充宽度
        lreal='Auto',          # 投影在实空间中
        ediff=1e-5,            # 能量收敛标准
        nelm=800,              # 电子自洽迭代最大步数
        nelmin=4,              # 电子自洽迭代最小步数
        gga='PBE',             # GGA近似，PBE泛函
        vdw='d3_zero',         # DFT-D3 Grimme with zero-damping function
        lwav=False,            # 不输出波函数
        lcharg=False,          # 不输出电荷密度
        # 其他必要的 VASP 参数，可以根据需要进行调整
    )
    atoms.calc = calc

    # 计算能量
    energy = atoms.get_potential_energy()
    return energy

def main():
    parser = argparse.ArgumentParser(description="批量优化结构并计算误差。")
    parser.add_argument("--model_path", required=True, help="Path to the MACE model file.")
    parser.add_argument("--data_dir", required=True, help="Directory containing .extxyz data files.")
    parser.add_argument("--output_dir", required=True, help="Root directory for optimized results.")
    parser.add_argument("--vasp_executable", default="vasp_std", help="VASP executable command (default: vasp_std).")
    parser.add_argument("--optimizer", choices=["LBFGS", "CG"], default="LBFGS", help="Optimization algorithm to use (default: LBFGS).")
    parser.add_argument("--fmax", type=float, default=0.01, help="Maximum force convergence criteria (default: 0.01 eV/Å).")
    parser.add_argument("--constrain_c_atoms", action='store_true', help="Apply constraints on C atoms to move only along z direction.")
    args = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    model_path = args.model_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    vasp_executable = args.vasp_executable
    optimizer = args.optimizer
    fmax = args.fmax
    constrain_c_atoms = args.constrain_c_atoms

    # 创建输出目录及子目录
    initial_structures_dir = os.path.join(output_dir, "initial_structures")
    reference_final_structures_dir = os.path.join(output_dir, "reference_final_structures")
    optimizedcell_dir = os.path.join(output_dir, "optimizedcell")
    os.makedirs(initial_structures_dir, exist_ok=True)
    os.makedirs(reference_final_structures_dir, exist_ok=True)
    os.makedirs(optimizedcell_dir, exist_ok=True)

    # 初始化日志文件
    log_file = os.path.join(output_dir, "error_logs.txt")
    with open(log_file, 'w') as log:
        log.write("filename\tmean_pos_error(A)\tenergy_error(eV/atom)\tmax_force_error(eV/Å)\tvasp_energy_error(eV)\n")

    # 遍历数据目录中的所有 .extxyz 文件
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".extxyz"):
            file_path = os.path.join(data_dir, filename)
            base_name = os.path.splitext(filename)[0]
            print(f"\n=== 处理文件: {filename} ===")

            # 读取所有构型
            try:
                all_structures = read(file_path, index=":")
            except Exception as e:
                print(f"错误: 无法读取文件 {filename}: {e}")
                continue

            if len(all_structures) < 2:
                print(f"警告: 文件 {filename} 包含的构型数少于2，跳过。")
                continue

            initial_conf = all_structures[0]  # 第一个构型
            final_true_conf = all_structures[-1]  # 最后一个构型

            # 保存初始和参考最终构型
            initial_save_path = os.path.join(initial_structures_dir, f"{base_name}_initial.extxyz")
            final_true_save_path = os.path.join(reference_final_structures_dir, f"{base_name}_reference_final.extxyz")
            try:
                write(initial_save_path, initial_conf)
                write(final_true_save_path, final_true_conf)
                print(f"保存初始构型到: {initial_save_path}")
                print(f"保存参考最终构型到: {final_true_save_path}")
            except Exception as e:
                print(f"错误: 无法保存构型文件: {e}")
                continue

            # 执行结构优化
            print("开始结构优化...")
            try:
                optimize_structure(
                    model_path=model_path,
                    init_structure_path=initial_save_path,
                    output_dir=output_dir,
                    optimizer=optimizer,
                    fmax=fmax,
                    constrain_c_atoms=constrain_c_atoms
                )
            except Exception as e:
                print(f"错误: 结构优化失败: {e}")
                continue

            # 构建优化后结构文件路径
            optimized_structure_filename = f"{base_name}_{optimizer}_relaxed.extxyz"
            optimized_structure_path = os.path.join(optimizedcell_dir, optimized_structure_filename)
            if not os.path.exists(optimized_structure_path):
                print(f"错误: 优化后的结构文件未找到: {optimized_structure_path}")
                continue

            # 读取优化后的最终构型
            try:
                final_pred_conf = read(optimized_structure_path)
                print(f"读取优化后的构型: {optimized_structure_path}")
            except Exception as e:
                print(f"错误: 无法读取优化后的构型: {e}")
                continue

            # 计算 VASP 单点能量
            print("计算优化后构型的 VASP 单点能量...")
            try:
                vasp_energy = calculate_vasp_energy(final_pred_conf, vasp_executable=vasp_executable)
                print(f"VASP 计算的单点能量: {vasp_energy:.6f} eV")
            except Exception as e:
                print(f"错误: VASP 计算失败: {e}")
                vasp_energy = None

            # 计算误差
            print("计算误差...")
            try:
                mean_pos_error, energy_error_per_atom, max_force_error = calculate_errors(final_true_conf, final_pred_conf)
            except Exception as e:
                print(f"错误: 计算误差失败: {e}")
                mean_pos_error = energy_error_per_atom = max_force_error = None

            # 计算 VASP 能量误差
            if vasp_energy is not None:
                try:
                    reference_energy = final_true_conf.get_potential_energy()
                    vasp_energy_error = abs(vasp_energy - reference_energy)  # eV
                    print(f"VASP 能量误差: {vasp_energy_error:.6f} eV")
                except Exception as e:
                    print(f"错误: 获取参考能量失败: {e}")
                    vasp_energy_error = 'N/A'
            else:
                vasp_energy_error = 'N/A'

            # 记录结果到日志文件
            try:
                with open(log_file, 'a') as log:
                    log.write(f"{filename}\t{mean_pos_error:.6f}\t{energy_error_per_atom:.6f}\t{max_force_error:.6f}\t{vasp_energy_error}\n")
                print(f"记录结果到日志文件: {log_file}")
            except Exception as e:
                print(f"错误: 无法写入日志文件: {e}")

            print("处理完成。")

    if __name__ == "__main__":
        main()
