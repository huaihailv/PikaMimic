import argparse
import numpy as np
import time
import os

import torch
import robomimic.utils.obs_utils as ObsUtils
from torchvision.utils import save_image
import cv2
# from agilex_env import get_obs, ros_obs_init

from egomimic.configs import config_factory
from egomimic.pl_utils.pl_model import ModelWrapper


from egomimic.scripts.evaluation.real_utils import *

import pickle


# NORM_STATS = to_torch(NORM_STATS, torch.device("cuda"))
CAM_KEY = "front_img_1"

def eval_real(model, rollout_dir, norm_stats, arm="both"):
    device = torch.device("cuda")
    # max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    num_rollouts = 100
    for rollout_id in range(num_rollouts):
        
        # args, config, ros_operator = ros_obs_init()
        # obs, states = get_obs(args, config, ros_operator)
        
        import h5py
        data = h5py.File("/home/lvhuaihai/EgoMimic/datasets/cobot_groceries_100pairs_400frame_50ac.hdf5", "r")
        data_obs = data['data']['demo_0']['obs']
        obs = {}
        obs["top"] = data_obs['front_img_1'][100]
        obs["left_wrist"] = data_obs['left_wrist_img'][100]
        obs["right_wrist"] = data_obs['right_wrist_img'][100]

        
        states = data_obs['joint_positions'][100]
        ee_pose = data_obs['ee_pose'][100]
        
        qpos = np.array(states)
        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
        ee_pose = np.array(ee_pose)
        ee_pose = torch.from_numpy(ee_pose).float().unsqueeze(0).to(device)

        with torch.inference_mode():
            for t in range(1000):
                print(f"t:{t}")

                data = {
                    "obs": {
                        "right_wrist_img": (
                            torch.from_numpy(obs["right_wrist"][None, None, :])
                        ).to(torch.uint8),
                        "front_img_1": (
                            torch.from_numpy(obs["top"][None, None, :])
                            ).to(torch.uint8),
                        "left_wrist_img": (
                            torch.from_numpy(obs["left_wrist"][None, None, :])
                        ).to(torch.uint8),
                        "pad_mask": torch.ones((1, 50, 1)).to(device).bool(),
                        "joint_positions": qpos[..., :].reshape((1, 1, -1)),
                        "ee_pose": ee_pose[..., :].reshape((1, 1, -1)),
                    },
                    "type": torch.tensor([0]),
                }

                # postprocess_batch
                input_batch = model.process_batch_for_training(
                    data, "actions_joints_act"
                )

                # right
                input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)/255.0

                # left
                input_batch["obs"]["left_wrist_img"] = input_batch["obs"]["left_wrist_img"].permute(0, 3, 1, 2)/255.0

                # breakpoint()
                input_batch["obs"][CAM_KEY] = input_batch["obs"][CAM_KEY].permute(0, 3, 1, 2)
                input_batch["obs"][CAM_KEY] /= 255.0
                input_batch = ObsUtils.normalize_batch(input_batch, normalization_stats=norm_stats, normalize_actions=False)
                info = model.forward_eval(input_batch, unnorm_stats=norm_stats)

                all_actions = info["actions_joints_act"].cpu().numpy() # 'actions_xyz_act'
                
                print(all_actions.shape)
                
                import pdb
                pdb.set_trace()

    return


def main(args):
    """
    Train a model using the algorithm.
    """
    # first set seeds
    np.random.seed(101)
    torch.manual_seed(101)

    model = ModelWrapper.load_from_checkpoint(args.eval_path, datamodule=None)
    norm_stats = os.path.join(os.path.dirname(os.path.dirname(args.eval_path)), "ds1_norm_stats.pkl")
    norm_stats = open(norm_stats, "rb")
    norm_stats = pickle.load(norm_stats)
    
    arm = "right"
    if model.model.ac_dim == 14:
        arm = "both"

    model.eval()
    rollout_dir = os.path.dirname(os.path.dirname(args.eval_path))
    rollout_dir = os.path.join(rollout_dir, "rollouts")
    if not os.path.exists(rollout_dir):
        os.mkdir(rollout_dir)

    if not args.debug:
        rollout_dir = None

    eval_real(model.model, rollout_dir, norm_stats, arm=arm)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--eval-path",
        type=str,
        default="/home/lvhuaihai/EgoMimic/experiment_results/PikaMimic/stackbasket_agilex_pika/models/model_epoch_epoch=6999.ckpt",
        help="(optional) path to the model to be evaluated",
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)
