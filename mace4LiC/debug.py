import ase.io
import pdb
file_path = '/share/project/hezhang/workspace/mect/mace/data/LixC12/raw/dataset_ibrion1/2_AA_0.extxyz'
atoms_list = ase.io.read(file_path, index=":")
pdb.set_trace()
print(atoms_list)