import numpy as np
import pytransform3d.rotations as rotations
import h5py
import cv2
import os

class Adaptor:

    def pose6D2quat(self, pose:np.ndarray):
        
        column_1 = pose[:3]
        column_2 = pose[3:]

        R = np.column_stack((column_1, column_2, np.cross(column_1, column_2)))

        # quat = rotations.quaternion_from_matrix(R)
        euler = rotations.euler_from_matrix(R, 0,1,2, extrinsic=True)
        return euler

    def read_data_in_batches(self, hdf5_file, batch_size=100):
        """
        逐批读取数据的生成器。按需读取数据块而非一次性加载。
        """
        total_data = len(hdf5_file['qpos'])
        for i in range(0, total_data, batch_size):
            batch_qpos = hdf5_file['qpos'][i:i + batch_size]
            batch_images = hdf5_file['images']['cam_high'][i:i + batch_size]
            yield batch_qpos, batch_images
            
    def qpos_2_ee_pose(self, qpos:h5py.Group, i):
        def extract_pose(group):
            xyz = group['xyz'][:,i]                # shape: (3,)
            angle = group['space_angle'][:,i]      # shape: (4,)
            gripper = group['gripper'][i]        # shape: () or (1,)
            gripper = np.expand_dims(gripper, axis=0) if gripper.ndim == 0 else gripper
            return np.concatenate([xyz, angle, gripper])  # shape: (8,)

        pose_L = extract_pose(qpos['eef_pose_L'])
        pose_R = extract_pose(qpos['eef_pose_R'])

        return np.concatenate([pose_L, pose_R])

    def package_ee_pose_action(self, data_num:int, ee_pose:np.ndarray, action_chunk:int):
        '''
        ee_pose: <number , 6>
        '''
        action_xyz = np.zeros((data_num,action_chunk,14))
        # print(f"ee_pose size:{ee_pose.size}")
        for i in range(data_num):
            if(i < data_num-action_chunk):
                action_xyz[i][0:action_chunk] = ee_pose[i+1:i+1+action_chunk]
            else:
                action_xyz[i][0:data_num-i-1] = ee_pose[i+1:data_num]
                for j in range(data_num-i-1, action_chunk):
                    action_xyz[i][j] = ee_pose[-1]
        return action_xyz
    
    def package_joint_pos_action(self,data_num:int,joint_pos:np.ndarray,action_chunk:int):
        '''
        joint_positions: <number, 7>
        '''
        action_joint = np.zeros((data_num,action_chunk,14))
        for i in range(data_num):
            if(i < data_num-action_chunk):
                action_joint[i][0:action_chunk] = joint_pos[i+1:i+1+action_chunk]
            else:
                action_joint[i][0:data_num-i-1] = joint_pos[i+1:data_num]
                for j in range(data_num-i-1, action_chunk):
                    action_joint[i][j] = joint_pos[-1]                
        return action_joint

    def qpos_2_joint_positions(self, qpos:np.ndarray):

        l_joint_pos = qpos[50:56]
        r_joint_pos = qpos[0:6]
        l_gripper_pos = np.array([qpos[60]])
        r_gripper_pos = np.array([qpos[10]])

        l_pos = np.concatenate((l_joint_pos,l_gripper_pos))
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos))

        return np.concatenate((l_pos,r_pos))


    def cam_high_2_front_img(self, cam:np.ndarray):
        '''
        input: (59606,)
        output: (480,640,3)
        '''
        frame = cv2.imdecode(cam, cv2.IMREAD_COLOR)
        
        return frame
    
    def resample_to_100_frames(self, data, target_length=100):
        current_length = data
        
        if current_length < target_length:
            # 如果当前长度小于目标长度，重复最后一帧
            len = target_length - current_length
            padded_data = np.tile([current_length-1], len)
            # padded_data = np.tile(last_frame, (target_length - current_length, 1) + (1,) * (data.ndim - 1))
            return np.concatenate([np.linspace(0, current_length - 1, current_length, dtype=int), padded_data], axis=0)
        
        # 计算均匀抽帧的间隔
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        
        return indices


    def rdt2ego_add_demo(self, rdt_data:h5py.Group, ego_data:h5py.Group, action_chunk:int, video_path:str):
        '''
        egoMinic demo Structure:
        demo_i
            actions_joints_act
            actions_xyz_act
            obs
                ee_pose
                front_img_1
                front_img_1_line
                joint_positions
                right_wrist_img

        rdt data Structure:
        observations
            images
                cam_high
                cam_left_wrist
                cam_right_wrist
            qpos
            qvel
        '''

        total_frame = len(rdt_data['eef_pose_L/gripper'])
        data_num = 400
        print(f"current file data numbers:{total_frame}")

        camera_f = self.read_mp4_to_numpy_array(os.path.join(video_path, "camera_c.mp4"))
        camera_l = self.read_mp4_to_numpy_array(os.path.join(video_path, "camera_l.mp4"))
        camera_r = self.read_mp4_to_numpy_array(os.path.join(video_path, "camera_r.mp4"))
        
        
        # create a new demo group
        obs_group = ego_data.create_group('obs')

        # create 'ee_pose' dataset with compression and chunking
        ee_pose = np.zeros((data_num, 14))

        # create 'front_img_1' dataset with compression and chunking
        front_img = np.zeros((data_num, 480, 640, 3))
        right_wrist_img = np.zeros((data_num, 480, 640, 3))
        left_wrist_img = np.zeros((data_num, 480, 640, 3))

        # create joint position dataset
        joint_pos = np.zeros((data_num, 14))

        prev_ee = None
        
        indices = self.resample_to_100_frames(total_frame, 400)
        for j in range(len(indices)):
            i = indices[j]
            curr_ee = self.qpos_2_ee_pose(rdt_data, i)
            # curr_joint = self.qpos_2_joint_positions(rdt_data, i)

            if prev_ee is None:
                ee_pose[j] = np.zeros_like(curr_ee)
            else:
                ee_pose[j] = curr_ee - prev_ee
            prev_ee = curr_ee

            front_img[j] = cv2.resize(camera_f[i], (640, 480), interpolation=cv2.INTER_LINEAR)
            left_wrist_img[j] = camera_l[i]
            right_wrist_img[j] = camera_r[i]

        # create actions
        action_xyz = self.package_ee_pose_action(data_num, ee_pose, action_chunk)

        # create obs
        obs_group.create_dataset(
            "front_img_1", data=front_img)
        obs_group.create_dataset(
            "joint_positions", data=joint_pos)
        obs_group.create_dataset(
            "right_wrist_img", data=right_wrist_img)
        obs_group.create_dataset(
            "left_wrist_img", data=left_wrist_img)
        obs_group.create_dataset(
            "ee_pose", data=ee_pose)
        print("obs_finish")
        
        # create actions datasets with compression
        ego_data.create_dataset("actions_xyz_act", data=action_xyz)
        # ego_data.create_dataset("actions_joints_act", data=action_joints_pos)
        print("actions_finish")
        

    # 递归搜索指定路径下的所有符合条件的文件
    def find_episode_files(self,root_dir):
        # 用于存储匹配文件的路径
        episode_dirs = []

        # 遍历根目录下的所有文件和文件夹
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 遍历当前目录下的每个子目录
            for dirname in dirnames:
                # 检查目录名是否匹配 'episode_xx' 模式
                if dirname.startswith('episode'):
                    # 获取目录的绝对路径
                    dir_path = os.path.join(dirpath, dirname)
                    episode_dirs.append(dir_path)
        return episode_dirs
    
    def read_mp4_to_numpy_array(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Convert list of frames to a NumPy array
        frames_array = np.array(frames)
        
        return frames_array
    
    def rdt2ego(self, ego_data_path:str, action_chunk:int, rdt_data_path:str):
        # get filename of all episode
        files_list = self.find_episode_files(rdt_data_path)

        # create a new dataset to store the new data
        ego_data:h5py.File = h5py.File(ego_data_path, 'w')
        data = ego_data.create_group("data")

        for i in range(len(files_list)):
            # print(f"start save in episode_{i}.hdf5 with action chunk: {action_chunk}")
            print("start to handle {}_th file: ".format(i) + files_list[i])
            ego_group = data.create_group(f"demo_{i}")

            dir_path = files_list[i]
            for filename in os.listdir(dir_path):
                if filename.endswith('.hdf5'):
                    hdf5_file = os.path.join(dir_path, filename)
                    break
            rdt_data:h5py.File = h5py.File(hdf5_file, 'r')
            video_path = dir_path
            self.rdt2ego_add_demo(rdt_data, ego_group, action_chunk, video_path)
            ego_group.attrs["num_samples"] = 400

            rdt_data.close()

        mask_group = ego_data.create_group("mask")
        
        # 创建 train 和 valid 数据集
        train_data = []
        valid_data = []

        all_indices = list(range(len(ego_data['data'])))

        import random
        train_indices = random.sample(all_indices, 10)
        valid_indices = list(set(all_indices) - set(train_indices))  # 剩下的分配到 valid

        # 按照选定的索引填充 train 和 valid 数据
        for i in train_indices:
            demo_str = f"demo_{i}"
            train_data.append(demo_str)

        for i in valid_indices:
            demo_str = f"demo_{i}"
            valid_data.append(demo_str)

        # 创建数据集并写入字符串
        mask_group.create_dataset("train", data=[s.encode('utf-8') for s in train_data])
        mask_group.create_dataset("valid", data=[s.encode('utf-8') for s in valid_data])

        ego_data.close()
        

# 1. num_samples is corresponding to the fixed length of each demo
# 2. action chunk setting
# 3. rename the file according to the pairs
# 4. indices division

if __name__ == "__main__":
    adaptor = Adaptor()
    action_chunk = 50
    adaptor.rdt2ego(ego_data_path=f"/share/project/lvhuaihai/lvhuaihai/EgoMimic/datasets/pika_build_blocks_1000pairs_400frame_{action_chunk}ac.hdf5", \
        action_chunk = action_chunk, \
        rdt_data_path="/share/project/lvhuaihai/robot_data/pika/build_blocks")
    # 路径名字：demo name; pairs number; action chunk; fixed length of each demo;
    # num_samples = fixed length of each demo
    # self.resample_to_100_frames修改参数为fixed length of each demo
    