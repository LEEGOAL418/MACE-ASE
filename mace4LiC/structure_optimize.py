import os
import time
import argparse
import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.io.trajectory import Trajectory
from mace.calculators.mace import MACECalculator
from ase.constraints import FixedLine
import csv

def compute_mae(refr_config, pred_config):
    """
    计算原子坐标的 MAE (Mean Absolute Error)
    
    参数：
        ref_config :参考构型
        pred_config :预测构型
    
    返回：
        torch.Tensor: 计算得到的 MAE
    """
    refr_positions = refr_config.arrays["positions"]
    pred_positions = pred_config.arrays["positions"]
    mae_coords = np.abs(refr_positions - pred_positions)
    total_mae = np.mean(mae_coords)
    return total_mae

def optimize_structure(
    model_path,
    init_structure_path,
    output_dir,
    optimizer="LBFGS",
    fmax=0.01,
    constrain_c_atoms=False,
    result_csv_path="optimization_results.csv"
):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取初始结构的基础文件名（不带后缀）
    base_name = os.path.splitext(os.path.basename(init_structure_path))[0]

    # 动态生成输出文件路径
    traj_file_path = os.path.join(output_dir, "trajectory", f"{base_name}_{optimizer}_traj.traj")
    relaxed_structure_path = os.path.join(output_dir, "optimizedcell", f"{base_name}_{optimizer}_relaxed.extxyz")
    steps_output_path = os.path.join(output_dir, "steps", f"{base_name}_{optimizer}_steps.extxyz")

    # 确保各子目录存在
    os.makedirs(os.path.dirname(traj_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(relaxed_structure_path), exist_ok=True)
    os.makedirs(os.path.dirname(steps_output_path), exist_ok=True)

    # 加载 MACE 模型
    calculator = MACECalculator(model_path=model_path, device='cuda', dtype='float64')

    # 读取初始结构文件（如果有多个构型，则取第一个和最后一个）
    all_confs = read(init_structure_path, index=':')
    init_conf = all_confs[0]
    refr_conf = all_confs[-1]
    
    print(f"Total number of configurations in file: {len(all_confs)}")
    print("Using the first configuration (index=0) for optimization and the last one for reference.")

    # 为初始结构设置 MACE 计算器
    init_conf.calc = calculator

    # 施加约束,目前仅限定z方向自由度（如果指定）
    if constrain_c_atoms:
        carbon_indices = [atom.index for atom in init_conf if atom.symbol == 'C']
        if carbon_indices:
            constraints = FixedLine(indices=carbon_indices, direction=[0, 0, 1])
            init_conf.set_constraint(constraints)
            print("Applied constraints on carbon atoms to move only along z direction.")
        else:
            print("No carbon atoms found to apply constraints.")

    # 打印初始能量
    initial_energy = init_conf.get_potential_energy()
    print(f"Initial Energy: {initial_energy:.6f} eV")

    # 记录优化开始时间
    start_time = time.time()

    # 创建 .traj 文件记录优化过程
    traj = Trajectory(traj_file_path, 'w', init_conf)

    # 选择优化器
    if optimizer == "LBFGS":
        dyn = LBFGS(init_conf, trajectory=traj)
        print("Using LBFGS optimizer.")
    elif optimizer == "CG":
        dyn = SciPyFminCG(init_conf, trajectory=traj)
        print("Using SciPy CG optimizer.")
    else:
        raise ValueError("Invalid optimizer selected. Choose 'LBFGS' or 'CG'.")

    # 运行优化
    dyn.run(fmax=fmax)

    # 记录优化结束时间
    end_time = time.time()

    # 打印优化总时间
    total_time = end_time - start_time
    print(f"Optimization completed in {total_time:.2f} seconds.")

    # 打印优化后的能量
    pred_conf = init_conf
    optimized_energy = pred_conf.get_potential_energy()
    print(f"Optimized Energy: {optimized_energy:.6f} eV")

    # 计算优化后的坐标与参考结构的 MAE
    mae = compute_mae(refr_conf, pred_conf)
    print(f"Coordinates MAE: {mae:.6f} Å")

    # 输出优化后的结构到文件
    write(relaxed_structure_path, pred_conf)
    print(f"Relaxed structure saved to: {relaxed_structure_path}")

    # 将轨迹文件中的所有优化步骤保存到一个 .extxyz 文件
    steps = read(traj_file_path, index=':')
    with open(steps_output_path, 'w') as f:
        for step, atoms in enumerate(steps):
            write(f, atoms, format='extxyz')
    print(f"Optimization steps saved to: {steps_output_path}")

    # 记录优化结果到 CSV
    results = {
        "filename": base_name,
        "steps": len(steps),
        "time(s)": total_time,
        "final_energy(eV)": optimized_energy,
        "Coordinates MAE(A)": mae
    }

    # 使用 csv.writer 处理 CSV 文件
    file_exists = os.path.exists(result_csv_path)
    with open(result_csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "steps", "time(s)", "final_energy(eV)", "Coordinates MAE(A)"])

        # 如果文件不存在，则写入标题行
        if not file_exists:
            writer.writeheader()

        # 写入结果
        writer.writerow(results)
    
    print(f"Results saved to {result_csv_path}")

    print("Optimization finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize atomic structures using MACE and ASE.")
    parser.add_argument("--model_path", required=True, help="Path to the MACE model file.")
    parser.add_argument("--init_structure_path", required=True, help="Path to the initial structure file.")
    parser.add_argument("--output_dir", required=True, help="Root directory for output files.")
    parser.add_argument(
        "--optimizer",
        choices=["LBFGS", "CG"],
        default="LBFGS",
        help="Optimization algorithm to use (default: LBFGS)."
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Maximum force convergence criteria (default: 0.01 eV/Å)."
    )
    parser.add_argument(
        "--constrain_c_atoms",
        action='store_true',
        help="Apply constraints on C atoms to move only along z direction."
    )
    parser.add_argument(
        "--result_csv_path",
        default="optimization_results.csv",
        help="Path to save the optimization results in CSV format."
    )

    args = parser.parse_args()

    optimize_structure(
        model_path=args.model_path,
        init_structure_path=args.init_structure_path,
        output_dir=args.output_dir,
        optimizer=args.optimizer,
        fmax=args.fmax,
        constrain_c_atoms=args.constrain_c_atoms,
        result_csv_path=args.result_csv_path
    )
