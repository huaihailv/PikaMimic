import h5py
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from scipy.spatial.transform import Rotation as Rot
import json
# from simarUtils import *
import torchvision

data = h5py.File("/home/lvhuaihai/EgoMimic/datasets/aloha_fruit.hdf5", "r")

import cv2
import numpy as np
import time

# 模拟tensor作为一组二进制图片帧
# video_frames = data['observations']['images']['cam_high'] # (662, )
video_frames = np.asarray(data['data']['demo_0']['obs']['front_img_1'])
video_frames = video_frames.astype(np.uint8)
# frame = cv2.imdecode(video_frames[0], cv2.IMREAD_COLOR)
print(video_frames.shape)
height, width, layers = video_frames[0].shape
print(video_frames[0].shape)

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
fps = 25  # 设置帧率
video = cv2.VideoWriter('output_video_high_aloha0.mp4', fourcc, fps, (width, height))

for binary_image in video_frames:
    print(binary_image.shape)
    # 使用 cv2.imdecode 解码
    # frame = np.frombuffer(binary_image, np.uint8)
    # frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    video.write(binary_image)

video.release()
print("Video saved as 'output_video_high_aloha0.mp4'.")