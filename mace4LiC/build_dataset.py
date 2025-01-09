import os, glob
import ase.io 
import pandas as pd

def merge_files(paths, outpath, metapath):
    meta = []
    all_confs = []
    for path in paths:
        confs = ase.io.read(path, index=":")
        meta.append(dict(
            path=path,
            num_confs=len(confs),
            num_atoms=len(confs[0].get_atomic_numbers()),
            atoms=confs[0].get_atomic_numbers(),
        ))
        all_confs.extend(confs)
    ase.io.write(outpath, all_confs)
    meta = pd.DataFrame.from_records(meta)
    meta.to_csv(meta_path, index=None)

data_dir = '/share/project/hezhang/workspace/mect/mace/data/LixC12'

train_aa_files = []
train_ab_files = []
for i in range(2, 7):
    for j in range(11):
        train_aa_files.append(f'{data_dir}/raw/dataset_ibrion1/{i}_AA_{j}.extxyz')
        train_ab_files.append(f'{data_dir}/raw/dataset_ibrion1/{i}_AB_{j}.extxyz')

for i in range(2, 7):
    train_aa_files.append(f'{data_dir}/raw/dataset_ibrion2/{i:02d}_AA.extxyz')
    train_ab_files.append(f'{data_dir}/raw/dataset_ibrion2/{i:02d}_AB.extxyz')

test_aa_files = []
test_ab_files = []
for i in range(7, 21):
    test_aa_files.append(f'{data_dir}/raw/dataset_ibrion2/{i:02d}_AA.extxyz')
    test_ab_files.append(f'{data_dir}/raw/dataset_ibrion2/{i:02d}_AB.extxyz')

train_files = train_aa_files + train_ab_files
test_files = test_aa_files + test_ab_files
print(len(train_files), len(test_files))

train_path = os.path.join(data_dir, 'dataset/train.extxyz')
meta_path = os.path.join(data_dir, 'dataset/train_meta.csv')
merge_files(train_files, train_path, meta_path)

test_path = os.path.join(data_dir, 'dataset/test.extxyz')
meta_path = os.path.join(data_dir, 'dataset/test_meta.csv')
merge_files(test_files, test_path, meta_path)
