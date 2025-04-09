import h5py
import numpy as np

def resample_to_80_frames(data, target_length=80):
    current_length = data.shape[0]
    
    if current_length < target_length:
        # 如果当前长度小于目标长度，重复最后一帧
        last_frame = data[-1:]
        padded_data = np.tile(last_frame, (target_length - current_length, 1) + (1,) * (data.ndim - 1)).squeeze()
        return np.concatenate([data, padded_data], axis=0)
    
    # 计算均匀抽帧的间隔
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    resampled_data = data[indices]
    
    return resampled_data

def process_hdf5_file(file_path, output_path):
    print(f"start process {file_path}")
    with h5py.File(file_path, 'r') as data:
        # 假设数据存储在 'observations/images/cam_high' 中
        import cv2
        data_num = data['head_pose'].shape[0]
        
        image_data = data['img_front']
        left = data['left_pose']
        right = data['right_pose']
        # qpos_data = np.concatenate([left, right], axis=1)
        head_data = data['head_pose'][:]
        
        # 抽帧到400帧
        resampled_image_data = resample_to_80_frames(image_data)
        resampled_left_data = resample_to_80_frames(left)
        resampled_right_data = resample_to_80_frames(right)
        resampled_action_data = resample_to_80_frames(head_data)
        
        # 创建一个新的HDF5文件来保存抽帧后的数据
        with h5py.File(output_path, 'w') as resampled_data:
            resampled_data.create_dataset('img_front', data=resampled_image_data)
            resampled_data.create_dataset('left_pose', data=resampled_left_data)
            resampled_data.create_dataset('right_pose', data=resampled_right_data)
            resampled_data.create_dataset('head_pose', data=resampled_action_data)

        print(f"resample_finish on {output_path}")

def generate_output_path(file_path, index):
    from pathlib import Path
    # 获取路径和文件名
    file_name = Path(file_path).name
    file_path = Path(file_path).parent
    
    # 生成新的文件名
    new_base_name = f"{file_path}/episode_{index}_80.hdf5"
    # 返回新的文件路径
    return f"{new_base_name}"

import os
# 递归搜索指定路径下的所有符合条件的文件
def find_episode_files(root_dir):
    # 用于存储匹配文件的路径
    episode_files = []

    # 遍历根目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 遍历当前文件夹中的每个文件
        for filename in filenames:
            # 检查文件名是否匹配 'data_xx' 模式
            if filename.startswith('data_'):
                # 获取文件的绝对路径
                file_path = os.path.join(dirpath, filename)
                episode_files.append(file_path)
    return episode_files
    
# 示例调用
file_path = "/home/lvhuaihai/EgoMimic/datasets/human_manipulation_dat"
files_list = find_episode_files(file_path)
print(files_list)
for index, file_list in enumerate(files_list):
    output_path = generate_output_path(file_list, index)
    process_hdf5_file(file_list, output_path)