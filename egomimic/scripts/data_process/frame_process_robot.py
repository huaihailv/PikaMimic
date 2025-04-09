import h5py
import numpy as np
import os
from pathlib import Path

def resample_to_400_frames(data, target_length=400):
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

def process_hdf5_file(file_path, output_path, target_len=400):
    print(f"start process {file_path}")
    with h5py.File(file_path, 'r') as data:
        # 假设数据存储在 'observations/images/cam_high' 中
        import cv2
        data_num = data['observations/images/cam_high'].shape[0]
        
        image_data = data['observations/images/cam_high']
        left_image_data = data['observations/images/cam_left_wrist']
        right_image_data = data['observations/images/cam_right_wrist']
        qpos_data = data['observations/qpos'][:]
        action_data = data['action'][:]
        
        # 抽帧到400帧
        resampled_image_data = resample_to_400_frames(image_data, target_len)
        resampled_left_image_data = resample_to_400_frames(left_image_data, target_len)
        resampled_right_image_data = resample_to_400_frames(right_image_data, target_len)
        resampled_qpos_data = resample_to_400_frames(qpos_data, target_len)
        resampled_action_data = resample_to_400_frames(action_data, target_len)
        
        # 创建一个新的HDF5文件来保存抽帧后的数据
        with h5py.File(output_path, 'w') as resampled_data:
            resampled_data.create_dataset('observations/images/cam_high', data=resampled_image_data)
            resampled_data.create_dataset('observations/images/cam_left_wrist', data=resampled_left_image_data)
            resampled_data.create_dataset('observations/images/cam_right_wrist', data=resampled_right_image_data)
            resampled_data.create_dataset('observations/qpos', data=resampled_qpos_data)
            resampled_data.create_dataset('action', data=resampled_action_data)
        print(f"resample_finish on {output_path}")

def generate_output_path(file_path, index, target_len):
    # 获取路径和文件名
    file_name = Path(file_path).name
    file_path = Path(file_path).parent

    # 生成新的目录名
    new_dir_name = f"{file_path.parent}/{file_path.name}_{target_len}"
    new_base_name = f"{new_dir_name}/{file_name}"
    
    # 创建新的目录
    os.makedirs(new_dir_name, exist_ok=True)
    
    # 返回新的文件路径
    return f"{new_base_name}"

# 递归搜索指定路径下的所有符合条件的文件
def find_episode_files(root_dir):
    # 用于存储匹配文件的路径
    episode_files = []

    # 遍历根目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 遍历当前文件夹中的每个文件
        for filename in filenames:
            # 检查文件名是否匹配 'data_xx' 模式
            if filename.startswith('episode_'):
                # 获取文件的绝对路径
                file_path = os.path.join(dirpath, filename)
                episode_files.append(file_path)
    return episode_files
    
# 示例调用
file_path = "/home/lvhuaihai/EgoMimic/datasets/groceries_cobot_260115_100"
files_list = find_episode_files(file_path)
for index, file_list in enumerate(files_list):
    output_path = generate_output_path(file_list, index, target_len=400)
    process_hdf5_file(file_list, output_path)