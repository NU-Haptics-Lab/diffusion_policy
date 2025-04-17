import time
import math
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
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
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

# ROS2 stuff
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.duration import Duration
import builtin_interfaces.msg

import rosbag2_py
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ROS2 messages
from sensor_msgs.msg import Image
from ros2_to_rlds_msgs.msg import Float64array


OmegaConf.register_new_resolver("eval", eval, replace=True)

## globals
# values taken from `gestures_diffuison_py/datasets/convert_dataset.py`
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_NB_CHANNELS = 3
OBSERVATION_TOPIC = '/gestures_diffusion/leftcam/image_resized'
STATE_TOPIC = '/GesturesDiffusionTelemetry/joint_states'
CHECKPOINT_PATH = "/home/omnid/dexnex/libraries/diffusion_policy/outputs/2025-03-12/11-56-55/checkpoints/epoch=0150-train_loss=0.006.ckpt"
NODE_NAME = 'GesturesDiffusionEval'

class EvalDexNex(Node):
    def __init__(self):
        # init ROS node
        super().__init__(NODE_NAME)
        
        # declare ROS2 params
        # self.declare_parameter('input', , "Path to checkpoint") 
        # self.declare_parameter('frequency', 10.0, "Control frequency in Hz.")
        # self.declare_parameter('debug', False, "Whether to debug.")
        
        # # get ROS2 params
        # input = self.get_parameter('input')
        # frequency = self.get_parameter('frequency')
        # self.DEBUG = self.get_parameter('debug')
        input = CHECKPOINT_PATH
        frequency = 1.0
        self.DEBUG = False
        
        
        if self.DEBUG:
            frequency = 0.2
        
        # load checkpoint
        ckpt_path = input
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        #
        self.cfg = cfg
        
        # cfg params
        use_ema = cfg.training.use_ema
        n_obs_steps = cfg.n_obs_steps
        img_shape = cfg.shape_meta.obs.image.shape
        state_length = cfg.task.dataset.state_length
        

        # diffusion model
        policy: BaseImagePolicy
        self.policy = workspace.model
        if use_ema:
            self.policy = workspace.ema_model

        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)

        # set inference params
        self.policy.num_inference_steps = 16 # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        # setup experiment
        dt = 1/frequency

        print("n_obs_steps: ", n_obs_steps)
        
        # default values, one time step
        self.m_joint_states = np.zeros(state_length)
        self.m_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NB_CHANNELS))
        
        # observation history
        self.image_history = deque(maxlen=n_obs_steps) # use deque instead of queue because it has maxlen
        self.state_history = deque(maxlen=n_obs_steps) # use deque instead of queue because it has maxlen
        
        # save ros data for n_obs_steps times so that our deques aren't empty and have the correct data format
        for _ in range(n_obs_steps):
            self.SaveRosData()
        
        # ensure policy is reset
        with torch.no_grad():
            self.policy.reset()
        
        ## ROS2 setup
        # QoS, required for image compat
        qos_profile = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        # subs
        self.sub1_ = self.create_subscription(Float64array, STATE_TOPIC, self.SubJointStates, 1)
        
        self.sub2_ = self.create_subscription(Image, OBSERVATION_TOPIC, self.SubImage, qos_profile)
        
        # pubs
        self.pub_trajectory = self.create_publisher(JointTrajectory, "~/joint_trajectory", 10)
        
        # timer
        self.timer_ = self.create_timer(dt, self.Run)
    
    """ Use a separate sub in case the publish rate differs from our frequency """
    def SubJointStates(self, msg):
        self.m_joint_states = msg.data
        
    def SubImage(self, msg):
        # put msg into an np.array
        img_np = np.array(msg.data, dtype="float")
        
        # reshape the image
        img_np_reshaped = np.reshape(img_np, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NB_CHANNELS))
        
        self.m_image = img_np_reshaped
        
    """ Take the asynch raw ROS2 messages and save them for use in our policy """
    def SaveRosData(self):
        # push ROS2 data to our queues
        self.state_history.appendleft(self.m_joint_states)
        
        self.image_history.appendleft(self.m_image)
        
    """ Convert deque to list """
    def DequeToList(self, dq):
        # Reversed since we `deque.appendleft` the MOST RECENT time step but we want our input obs to go from left-to-right from past-to-present
        # example: deque.appendleft(1); deque.appendleft(2); deque[0] == 2; deque[1] == 1 so we iterate from max idx value to min idx value
        out = []
        for idx in reversed(range(len(dq))):
            out.append(dq[idx])
            
        return out

    """
    From predict_action::217
    obs_dict: must include "obs" key THIS IS A LIE!!!! lol
    
    From pusht_image_dataset.py::78
    'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, state_length
            },
    """
    def GetObs(self):
        # convert deque to list of np arrays. 
        imgs_ls = self.DequeToList(self.image_history)
        states_ls = self.DequeToList(self.state_history)
            
        # convert to np
        imgs_np = np.stack(imgs_ls)
        states_np = np.stack(states_ls)
        
        # same preprocessing as dexnex_pusht_image_dataset.py
        image = np.moveaxis(imgs_np, -1, 1) / 255
        
        # construct output dictionary
        obs_dict_np = {
                'image': image, # T, 3, 96, 96
                'agent_pos': states_np, # T, state_length
            }
        
        # done
        return obs_dict_np

    def NewEpisode(self):
        with torch.no_grad():
            self.policy.reset()

    def RunInference(self, obs_dict_np):
        # run inference
        with torch.no_grad():
            s = time.time()

            # convert from numpy to torch, add a batch dimension, and transfer to the GPU
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            
            # inside predict_action -> conditional_sample is where the iteration occurs. `for t in scheduler.timesteps`
            result = self.policy.predict_action(obs_dict)
            
            # # this action starts from the first obs step
            # action = result['action'][0].detach().to('cpu').numpy()
            print('Inference latency:', time.time() - s)
        
            return result

    """  """
    def PublishTrajectory(self, results):
        # extract full action from policy output
        # action_pred = results['action_pred'] # horizon long (16), on GPU
        action = results['action'] # n_action_steps long (8), on GPU. First action is at the current time
        
        # publish joint state trajectory to ROS2
        msg = JointTrajectory()
        msg.points = []
        
        # transfer to CPU and numpy. Take the first index to remove the batch axis
        action_cpu_np = action[0].detach().to('cpu').numpy()
        
        # iterate over the trajectory step dimension
        for idx in range(action_cpu_np.shape[0]):
            point = action_cpu_np[idx]
            
            msg2 = JointTrajectoryPoint()
            msg2.positions = point[:] # copy
            
            msg.points.append(msg2)
            
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_trajectory.publish(msg)
        

    """ Ran on a timer """
    def Run(self):
        # sync ROS2 data
        self.SaveRosData()
        
        # get observation
        obs_dict_np = self.GetObs()
        
        # run inference
        results = self.RunInference(obs_dict_np)
        
        # publish results
        self.PublishTrajectory(results)
        
        
def main(args=None):
    rclpy.init(args=args)

    node = EvalDexNex()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()