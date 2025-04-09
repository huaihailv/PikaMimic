import h5py
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from scipy.spatial.transform import Rotation as Rot
import json
# from simarUtils import *
# import torchvision

# data = h5py.File("/home/lvhuaihai/EgoMimic/datasets/groceries_human.hdf5", "r")
# data = h5py.File("/home/lvhuaihai/EgoMimic/datasets/groceries_robot.hdf5", "r")

# data = h5py.File("/home/lvhuaihai/EgoMimic/datasets/human_manipulation_dat/data_20241221-161430/data_20241221-161430.h5", "r")
import cv2
import numpy as np
import time

# 模拟tensor作为一组二进制图片帧front_img_1_masked
# video_frames = np.asarray(data['data']['demo_1']['obs']['front_img_1']) # (662, )


# import pdb; pdb.set_trace()
height, width, layers = 480, 640, 3

# print(video_frames.shape)

# print(height, width, layers)
# print(video_frames.shape)

# 创建视频写入器
for i in range(100):
    data = h5py.File(f"/home/lvhuaihai/EgoMimic/datasets/cobot/data_groceries_cobot/episode_{i}.hdf5", "r")
    # video_frames = np.asarray(data['data'][f'demo_{i}']['obs']['front_img_1'])
    # video_frames = np.asarray(data['img_front'])
    # print(data['observations']['images'].keys())
    # video_frames = np.asarray(data['observations']['images']['cam_left_wrist'])
    video_frames = np.asarray(data['observations']['images']['cam_high'])
    import os
    os.makedirs("/home/lvhuaihai/robot_videos", exist_ok=True)
    # mp4_name = f"/home/lvhuaihai/robot_videos/episode_{i}_left.mp4"
    mp4_name = f"/home/lvhuaihai/robot_videos/episode_{i}_high.mp4"
    
   
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    fps = 25  # 设置帧率
    video = cv2.VideoWriter(mp4_name, fourcc, fps, (width, height))
    # video_frames = video_frames.astype(np.uint8)
    for binary_image in video_frames:
        frame = np.frombuffer(binary_image, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        video.write(frame)
    video.release()
    print(f"Video saved as {mp4_name}.")