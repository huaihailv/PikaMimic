from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.stream_id import StreamId

import os

import h5py
import numpy as np


def build_camera_matrix(provider, pose_t):
    T_world_device = pose_t.transform_world_device
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    calib = device_calibration.get_camera_calib(rgb_stream_label)
    rgb_camera_calibration = calibration.get_linear_camera_calibration(
        480,
        640,
        133.25430222 * 2,
        rgb_stream_label,
        calib.get_transform_device_camera(),
    )

    # rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
    T_world_rgb_camera = (T_world_device @ T_device_rgb_camera).to_matrix()
    return T_world_rgb_camera


def undistort_to_linear(provider, stream_ids, raw_image, camera_label="rgb"):
    camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    warped = calibration.get_linear_camera_calibration(
        480, 640, 133.25430222 * 2, camera_label, calib.get_transform_device_camera()
    )
    warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
    warped_rot = np.rot90(warped_image, k=3)
    return warped_rot


def reproject_point(pose, provider):
    ## cam_matrix := extrinsics
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    # point_pose_camera = cam_matrix @ pose_hom
    # print(point_pose_camera)
    calib = device_calibration.get_camera_calib(rgb_stream_label)
    T_device_sensor = device_calibration.get_transform_device_sensor(rgb_stream_label)
    point_position_camera = T_device_sensor.inverse() @ pose

    warped = calibration.get_linear_camera_calibration(
        480, 640, 133.25430222 * 2, "rgb", calib.get_transform_device_camera()
    )
    point_position_pixel = warped.project(point_position_camera)
    return point_position_pixel


def split_train_val_from_hdf5(hdf5_path, val_ratio):
    with h5py.File(hdf5_path, "a") as file:
        demo_keys = [key for key in file["data"].keys() if "demo" in key]
        num_demos = len(demo_keys)
        num_val = int(np.ceil(num_demos * val_ratio))

        indices = np.arange(num_demos)
        np.random.shuffle(indices)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = [f"demo_{i}" for i in train_indices]
        val_mask = [f"demo_{i}" for i in val_indices]

        file.create_dataset("mask/train", data=np.array(train_mask, dtype="S"))
        file.create_dataset("mask/valid", data=np.array(val_mask, dtype="S"))


def slam_to_rgb(provider):
    """
        Get slam camera to rgb camera transform
        provider: vrs data provider
    """
    device_calibration = provider.get_device_calibration()

    slam_id = StreamId("1201-1")
    slam_label = provider.get_label_from_stream_id(slam_id)
    slam_calib = device_calibration.get_camera_calib(slam_label)
    slam_camera = calibration.get_linear_camera_calibration(
        480, 640, 133.24530222 * 2, slam_label, slam_calib.get_transform_device_camera()
    )
    T_device_slam_camera = (
        slam_camera.get_transform_device_camera()
    )  # slam to device frame

    rgb_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_id)
    rgb_calib = device_calibration.get_camera_calib(rgb_label)
    rgb_camera = calibration.get_linear_camera_calibration(
        480, 640, 133.24530222 * 2, rgb_label, rgb_calib.get_transform_device_camera()
    )
    T_device_rgb_camera = (
        rgb_camera.get_transform_device_camera()
    )  # rgb to device frame

    transform = T_device_rgb_camera.inverse() @ T_device_slam_camera

    return transform
