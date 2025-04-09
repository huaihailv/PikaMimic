import os
import h5py
import numpy as np

# 设置路径
path = r'lvhuaihai/2025.02.10.10.20'

# 设置帧率
fps = 25

# 遍历目录中的所有 HDF5 文件
for filename in os.listdir(path):
    if filename.endswith(".hdf5"):
        file_path = os.path.join(path, filename)
        with h5py.File(file_path, 'r') as hdf5_file:
            try:
                # 假设图像数据存储在 'observations/images/cam_high' 中
                video_frames = np.asarray(hdf5_file['observations']['images']['cam_high'])
                num_frames = video_frames.shape[0]
                video_length = num_frames / fps
                print(f"文件: {filename}, 视频长度: {video_length:.2f} 秒")
            except KeyError as e:
                print(f"文件: {filename}, 错误: {e}")
            except Exception as e:
                print(f"文件: {filename}, 发生错误: {e}")