import os
import time
import math
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import dill
import hydra
import pathlib
from collections import deque
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common import pytorch_util
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
import diffusion_policy.globals as globals
from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.model.diffusion_model import DiffusionModel
from diffusion_policy import utils

from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# ROS2 stuff
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.duration import Duration
import builtin_interfaces.msg

import rclpy.time
import rosbag2_py
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ROS2 messages
from sensor_msgs.msg import JointState
from haptx_interfaces.msg import BiotacNormalized
from ros2_to_rlds_msgs.msg import Float64array

from diffusion_policy.evaluation.eval import EvalDexNex, EvalMixin

class MageHandInference:
    """
    assumes just 1 obs history. No stacking needed
    """
    def __init__(self,
                 policy: DiffusionModel,
                 debug,
                 analytics,
                 task_id,
                 ) -> None:
        self.policy = policy
        self.debug = debug
        self.analytics = analytics
        self.task_id = torch.tensor([[task_id]], device=globals.CONFIG.device) # 2d #type:ignore
        
        # save a handle
        self.batch_loader = globals.DEFAULT_BATCH_LOADER
        
        # obs dict, not normalized, cpu numpy
        self.obs_dict = None
        
        # save n obs steps
        self.n_obs_steps = globals.CONFIG.models.models.actor.model.model.n_obs_steps # type:ignore
                
        # make the scheduler, hard-coded
        self.noise_scheduler = DDIMScheduler(
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=100,
            prediction_type="epsilon"
        )
        self.noise_scheduler.set_timesteps(globals.CONFIG.num_inference_steps) # type:ignore
        
    def set_policy(self, policy):
        self.policy = policy
        self.original_policy_noise_scheduler = self.policy.noise_scheduler
        
    def save_data(self,
        obs_dict,
    ):  
        """
        obs dict, not normalized, cpu numpy
        """
        # iterate over items
        for key, val in obs_dict.items():
            # numpy it
            val = np.array(val)
            
            # see if it's only 1 dimensional
            if len(val.shape) == 1:
                # add batch dim AND traj dim
                obs_dict[key] = val.reshape(1, 1, -1)
            elif len(val.shape) == 2:
                # add batch dim
                obs_dict[key] = val.reshape(1, val.shape[0], val.shape[1])
                
        self.obs_dict = obs_dict
        
    def infer(self):
        
        # check that we have states and observations
        if self.obs_dict is None:
            print("No obs dict yet")
            return None, None
        
        # get observation
        obs_dict_np = self.obs_dict
            
        # replace the policy's scheduler for inference
        self.policy.noise_scheduler = self.noise_scheduler
        
        # run inference
        action, all_actions = self.RunInference(obs_dict_np)
        
        # put the original back in for training
        self.policy.noise_scheduler = self.original_policy_noise_scheduler
        
        # assert(not np.isnan(all_actions).any())
        return action, all_actions
    
    def norm_gpu_obs(self, obs_dict_np):
        # wrap in a data dict, which is what the batch loader expects
        dd = {"obs": obs_dict_np}
        
        # put into 
        dd_torch = pytorch_util.dict_to_torch(dd)
        
        # must normalize ourselves, here, because of the way co-training works with separate normalizers per dataset
        ndd_torch = self.batch_loader.transfer_and_norm(dd_torch)
        
        nobs_torch = ndd_torch['obs']
        
        return nobs_torch
    
    def unnorm_cpu_action(self, naction_gpu):
        
        # split into joint_action and pose_action. See batch_loader.MageHandBatchLoader.__next__ for ordering information
        nb_joints = 22
        joint_action = naction_gpu[..., :nb_joints]
        rel_pose_action = naction_gpu[..., nb_joints:]
        
        naction_gpu_dd = {
            "joint_action": joint_action, "rel_pose_action": rel_pose_action
        }
        
        action_cpu_dd = self.batch_loader.unnorm_and_transfer(naction_gpu_dd)
                
        return action_cpu_dd
    
    def norm_gpu_action(self, action_cpu):
        action_cpu_dd = {"action": action_cpu}
        
        naction_dd = self.batch_loader.transfer_and_norm(action_cpu_dd)
        
        naction_gpu = naction_dd['action']
        
        return naction_gpu
        

    def RunInference(self, obs_dict_np):
                
        # run inference
        with torch.no_grad():
            s = time.time()
            
            nobs_torch = self.norm_gpu_obs(obs_dict_np)
            
            # inside predict_action -> conditional_sample is where the iteration occurs. `for t in scheduler.timesteps`
            naction_gpu, naction_rel_gpu, all_nactions_gpu = self.policy.infer(nobs_torch, task_id=self.task_id)
                        
            future_actions = self.unnorm_cpu_action(naction_gpu)
            all_actions = self.unnorm_cpu_action(all_nactions_gpu)
        
            return future_actions, all_actions

    def GetOb(self, h, is_image=False):
        # convert deque to list of np arrays. 
        ls = self.DequeToList(h)
        
        # convert to np
        np1 = np.stack(ls)
        
            
        np2 = np1
            
        return np2
        
    """ Convert deque to list """
    def DequeToList(self, dq):
        # Reversed since we `deque.appendleft` the MOST RECENT time step but we want our input obs to go from left-to-right from past-to-present
        # example: deque.appendleft(1); deque.appendleft(2); deque[0] == 2; deque[1] == 1 so we iterate from max idx value to min idx value
        out = []
        for idx in reversed(range(len(dq))):
            out.append(dq[idx])
            
        return out
    
