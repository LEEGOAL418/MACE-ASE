import os
import csv
from ase.io import read

def extract_optimization_info():
    output_root = "/home/user/Desktop/MACE-code/data/LixC12/raw/output_ibrion2"
    steps_dir = os.path.join(output_root, "steps")
    csv_output = os.path.join(output_root, "optimization_summary.csv")

    if not os.path.isdir(steps_dir):
        print(f"错误: steps 目录不存在: {steps_dir}")
        return

    files = os.listdir(steps_dir)
    steps_files = [f for f in files if f.endswith("_steps.extxyz")]

    if not steps_files:
        print(f"在 {steps_dir} 目录中未找到任何以 '_steps.extxyz' 结尾的文件。")
        return

    print(f"在 {steps_dir} 目录中找到 {len(steps_files)} 个待处理的 .extxyz 文件。")

    with open(csv_output, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'steps', 'final_energy (eV)'])

        for steps_file in steps_files:
            steps_path = os.path.join(steps_dir, steps_file)
            try:
                atoms = read(steps_path, index=':')
                steps_count = len(atoms)
                final_energy = atoms[-1].get_potential_energy()
                writer.writerow([steps_file, steps_count, final_energy])
                print(f"处理完成: {steps_file} | 步数: {steps_count} | 最终能量: {final_energy:.6f} eV")
            except Exception as e:
                print(f"处理文件 {steps_path} 时出错: {e}")

    print(f"所有信息已保存到 CSV 文件: {csv_output}")

if __name__ == "__main__":
    extract_optimization_info()
