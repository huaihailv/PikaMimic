import argparse
import numpy as np
import time
import os
from PIL import Image
import torch
import robomimic.utils.obs_utils as ObsUtils
from torchvision.utils import save_image
import cv2
import torchvision
import logging


from egomimic.configs import config_factory
from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.real_utils import *
import matplotlib.pyplot as plt
from egomimic.algo.act import ACT
import pickle

import cv2
# import pyarrow as pa
import os
from pathlib import Path

model_path = os.getenv("ROBOTIC_MODEL_NAME_OR_PATH", "/home/lvhuaihai/EgoMimic/experiment_results/PikaMimic/stackbasket_agilex_pika/models/model_epoch_epoch=399.ckpt")
norm_stats_path = os.getenv("NORM_STATS_PATH", "/home/lvhuaihai/EgoMimic/experiment_results/PikaMimic/stackbasket_agilex_pika/ds1_norm_stats.pkl")
device = torch.device("cuda")

CAM_KEY = "front_img_1"

def main():
    np.random.seed(101)
    torch.manual_seed(101)
    modelwrapper = ModelWrapper.load_from_checkpoint(model_path, datamodule=None)
    norm_stats = open(norm_stats_path, "rb")
    norm_stats = pickle.load(norm_stats)

    arm = "both"
    modelwrapper.eval() 
    model = modelwrapper.model
    import pdb
    pdb.set_trace()
    
    for rollout_id in range(1000):
        frames={}
        
        update_observation_window(args, config, ros_operator)
        
        frames['image_left']   = observation_window[-1]['images'][config['image_left'][0]]
        frames['image_right']  = observation_window[-1]['images'][config['image_right'][0]]
        frames['image_center'] = observation_window[-1]['images'][config['image_center'][0]]
    
        frames['image_center'] = cv2.resize(frames['image_center'], (640, 480), interpolation=cv2.INTER_LINEAR)
        qpos = torch.from_numpy(np.concatenate([joints['jointstate_left'], joints['jointstate_right']], axis = -1))
        qpos = qpos.unsqueeze(0).to(device)
        data = {
            "obs": {
                "right_wrist_img": (
                    torch.from_numpy(frames["image_right"][None, None, :].copy())
                ).to(torch.uint8),
                "left_wrist_img": (
                    torch.from_numpy(frames["image_left"][None, None, :].copy())
                ).to(torch.uint8),
                "pad_mask": torch.ones((1, 100, 1)).to(device).bool(),
                "joint_positions": qpos[..., :].reshape((1, 1, -1)),
            },
            "type": torch.tensor([0]),
        }   

        if CAM_KEY == "front_img_1":
            data["obs"][CAM_KEY] = torch.from_numpy(
                frames["image_center"][None, None, :].copy()
            ).to(torch.uint8)    

        input_batch = model.process_batch_for_training(
            data, "actions_joints_act"
        )
        # right
        input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)/255.0
        input_batch["obs"]["left_wrist_img"] = input_batch["obs"]["left_wrist_img"].permute(0, 3, 1, 2)/255.0
        input_batch["obs"][CAM_KEY] = input_batch["obs"][CAM_KEY].permute(0, 3, 1, 2)/255.0

        input_batch = ObsUtils.normalize_batch(input_batch, normalization_stats=norm_stats, normalize_actions=False)
        info = model.forward_eval(input_batch, unnorm_stats=norm_stats)

        all_actions = info["actions_joints_act"].detach().float().to("cpu").numpy()[0]

        for i in range(50):
            left_action = all_actions[i,:7]
            right_action = all_actions[i,7:]


            time.sleep(0.05)



if __name__ == "__main__":
    main()