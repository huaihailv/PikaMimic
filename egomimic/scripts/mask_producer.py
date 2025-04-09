import h5py
import random

# 创建HDF5文件并写入数据
hdf5_file_path = "/home/lvhuaihai/EgoMimic/datasets/ego.hdf5"

with h5py.File(hdf5_file_path, "a") as f:
    # 创建 mask group
    mask_group = f.create_group("mask")
    
    # 创建 train 和 valid 数据集
    train_data = []
    valid_data = []

    # 生成并按比例分配字符串
    for i in range(100):
        demo_str = f"demo_{i}"
        if random.random() < 0.8:  # 80% 分配到 train
            train_data.append(demo_str)
        else:  # 20% 分配到 valid
            valid_data.append(demo_str)

    # 创建数据集并写入字符串
    mask_group.create_dataset("train", data=[s.encode('utf-8') for s in train_data])
    mask_group.create_dataset("valid", data=[s.encode('utf-8') for s in valid_data])

print(f"HDF5 file '{hdf5_file_path}' created successfully with train and valid datasets.")