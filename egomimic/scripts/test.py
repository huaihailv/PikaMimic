import cv2
import h5py
data = h5py.File(f"/home/lvhuaihai/EgoMimic/datasets/data_cobot/2024.12.30.17.35/episode_12.hdf5", "r")
print(data.keys())
print(data['action'])
print(data['observations'].keys())
print(data['observations']['images'].keys())
print(data['observations']['images']['cam_high'].shape)
for i in range(data['observations']['images']['cam_high'].shape[0]):
    print(cv2.imdecode(data['observations']['images']['cam_high'][i], cv2.IMREAD_COLOR).shape)