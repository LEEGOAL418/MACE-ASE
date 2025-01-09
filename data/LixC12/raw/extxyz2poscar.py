import os
from ase.io import read, write

def convert_extxyz_to_poscar():
    # 定义输出根目录
    output_root = "/home/user/Desktop/MACE-ASE/data/LixC12/raw/output_ibrion2"
    
    # 定义 optimizedcell 目录路径（去掉方括号）
    optimizedcell_dir = os.path.join(output_root, "optimizedcell")
    
    if not os.path.isdir(optimizedcell_dir):
        print(f"错误: optimizedcell 目录不存在: {optimizedcell_dir}")
        return
    
    files = os.listdir(optimizedcell_dir)
    extxyz_files = [f for f in files if f.endswith("_relaxed.extxyz")]
    
    if not extxyz_files:
        print(f"在 {optimizedcell_dir} 目录中未找到任何以 '_relaxed.extxyz' 结尾的文件。")
        return
    
    print(f"在 {optimizedcell_dir} 目录中找到 {len(extxyz_files)} 个待转换的 .extxyz 文件。")
    
    # 定义 POSCAR 保存目录
    poscar_dir = os.path.join(output_root, "POSCAR")
    os.makedirs(poscar_dir, exist_ok=True)
    
    for extxyz_file in extxyz_files:
        extxyz_path = os.path.join(optimizedcell_dir, extxyz_file)
        try:
            atoms = read(extxyz_path)
        except Exception as e:
            print(f"读取文件 {extxyz_path} 时出错: {e}")
            continue

        poscar_filename = extxyz_file.replace("_relaxed.extxyz", "_POSCAR")
        poscar_path = os.path.join(poscar_dir, poscar_filename)

        try:
            write(poscar_path, atoms, format='vasp')
            print(f"成功转换: {extxyz_path} -> {poscar_path}")
        except Exception as e:
            print(f"转换文件 {extxyz_path} 到 POSCAR 时出错: {e}")

    print("所有文件转换完成。")

if __name__ == "__main__":
    convert_extxyz_to_poscar()
