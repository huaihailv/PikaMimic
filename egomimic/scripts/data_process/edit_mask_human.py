import numpy as np
# import pytransform3d.rotations as rotations
import h5py
import cv2
import os

class Adaptor:

    # def pose6D2quat(self, pose:np.ndarray):
        
    #     column_1 = pose[:3]
    #     column_2 = pose[3:]

    #     R = np.column_stack((column_1, column_2,np.cross(column_1, column_2)))

    #     quat = rotations.quaternion_from_matrix(R)
    #     return quat

    def read_data_in_batches(self, hdf5_file, batch_size=100):
        """
        逐批读取数据的生成器。按需读取数据块而非一次性加载。
        """
        total_data = len(hdf5_file['qpos'])
        for i in range(0, total_data, batch_size):
            batch_qpos = hdf5_file['qpos'][i:i + batch_size]
            batch_images = hdf5_file['images']['cam_high'][i:i + batch_size]
            yield batch_qpos, batch_images
            
    def qpos_2_ee_pose(self, qpos:np.ndarray, i):
        
        l_ee_trans = qpos['left_pose'][i , :3]

        r_ee_trans = qpos['right_pose'][i , :3]
        
        return np.concatenate((l_ee_trans, r_ee_trans))

    def package_ee_pose_action(self, data_num:int, ee_pose:np.ndarray):
        '''
        ee_pose: <number , 6>
        '''
        action_xyz = np.zeros((data_num,100,6))
        # print(f"ee_pose size:{ee_pose.size}")
        for i in range(data_num):
            if(i < data_num-100):
                action_xyz[i][0:100] = ee_pose[i+1:i+101]
            else:
                action_xyz[i][0:data_num-i-1] = ee_pose[i+1:data_num]
                for j in range(data_num-i-1, 100):
                    action_xyz[i][j] = ee_pose[-1]
        return action_xyz
    
    def package_joint_pos_action(self,data_num:int,joint_pos:np.ndarray):
        '''
        joint_positions: <number, 12>
        '''
        action_joint = np.zeros((data_num,100,12))
        for i in range(data_num):
            if(i < data_num-100):
                action_joint[i][0:100] = joint_pos[i:i+100]
            else:
                action_joint[i][0:data_num-i-1] = joint_pos[i+1:data_num]
                for j in range(data_num-i-1, 100):
                    action_joint[i][j] = joint_pos[-1]                
        return action_joint

    def qpos_2_joint_positions(self, qpos:np.ndarray, i):

        l_ee_trans = qpos['left_pose'][i , :6]

        r_ee_trans = qpos['right_pose'][i , :6]
        
        return np.concatenate((l_ee_trans, r_ee_trans))


    def cam_high_2_front_img(self, cam:np.ndarray):
        '''
        input: (59606,)
        output: (480,640,3)
        '''
        frame = cv2.imdecode(cam, cv2.IMREAD_COLOR)
        
        return frame
    

    def human2ego_add_demo(self, human_data:h5py.Group, ego_data:h5py.Group):
        '''
        egoMinic demo Structure:
        data
            demo_i
                actions_joints_act
                actions_xyz_act
                obs
                    ee_pose
                    front_img_1
                    front_img_1_line
                    joint_positions
                    right_wrist_img

        human data Structure:
        observations
            images
                cam_high
                cam_left_wrist
                cam_right_wrist
            qpos
            qvel
        '''

        total_frame = len(human_data['head_pose'])
        data_num = total_frame // 5 if total_frame % 5 == 0 else total_frame // 5 + 1  
        print(f"current file data numbers:{data_num}")

        # create a new demo group
        obs_group = ego_data.create_group('obs')

        # create 'ee_pose' dataset with compression and chunking
        ee_pose = np.zeros((data_num, 6))

        # create 'front_img_1' dataset with compression and chunking
        front_img = np.zeros((data_num, 480, 640, 3))
        
        # create joint position dataset
        joint_pos = np.zeros((data_num, 12))
        j = 0
        for i in range (0,data_num):
            if i % 100 == 0:
                print("query:{}".format(i))
            if i % 5 == 0:
                ee_pose[i]  = self.qpos_2_ee_pose(human_data, i)
                front_img[i] = human_data['img_front'][i]
                joint_pos[i] = self.qpos_2_joint_positions(human_data, i)

        # create actions
        action_xyz = self.package_ee_pose_action(data_num, ee_pose)
        action_joints_pos = self.package_joint_pos_action(data_num, joint_pos)
        print("start save in ego_video.hdf5")
        # create obs
        obs_group.create_dataset(
            "front_img_1", data=front_img, compression="gzip", compression_opts=1)
        obs_group.create_dataset(
            "joint_positions", data=joint_pos, compression="gzip", compression_opts=1)
        obs_group.create_dataset(
            "ee_pose", data=ee_pose, compression="gzip", compression_opts=1)
        print("obs_finish")
        
        # create actions datasets with compression
        ego_data.create_dataset("actions_xyz_act", data=action_xyz, compression="gzip", compression_opts=1)
        ego_data.create_dataset("actions_joints_act", data=action_joints_pos, compression="gzip", compression_opts=1)
        print("actions_finish")
        

    # 递归搜索指定路径下的所有符合条件的文件
    def find_episode_files(self,root_dir):
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
    
    def human2ego(self, ego_data_path:str,human_data_path:str="/home/lvhuaihai/EgoMimic/datasets/human_groceries"):
        # get filename of all episode #
        # files_list = self.find_episode_files(human_data_path)

        # create a new dataset to store the new data        
        ego_data:h5py.File = h5py.File(ego_data_path, 'a')
        # data = ego_data.create_group("data")

        # for i in range(len(files_list)):
            
        #     print(f"start to handle {i}_th file: " + files_list[i])
        #     ego_group = data.create_group(f"demo_{i}")

        #     human_data:h5py.File = h5py.File(files_list[i], 'r')

        #     self.human2ego_add_demo(human_data, ego_group)
        #     ego_group.attrs["num_samples"] = human_data['head_pose'].shape[0] // 5 if human_data['head_pose'].shape[0] % 5 == 0 else human_data['head_pose'].shape[0] // 5 + 1
        #     human_data.close()

        del ego_data['mask']
        mask_group = ego_data.create_group("mask")
        
        # # 创建 train 和 valid 数据集
        # train_data = []
        # valid_data = []

        # 生成索引
        all_indices = list(range(len(ego_data['data'])))

        # 随机挑选 4/5 个索引分配到 train，剩下的分配到 valid
        import random
        train_indices = random.sample(all_indices, 100)
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
        

if __name__ == "__main__":
    adaptor = Adaptor()
    adaptor.human2ego("/home/lvhuaihai/EgoMimic/datasets/human_groceries.hdf5")