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
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

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
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from haptx_interfaces.msg import BiotacNormalized
from ros2_to_rlds_msgs.msg import Float64array

# moveit for FK
from moveit.core.robot_model import RobotModel
from moveit.core.robot_state import RobotState


OmegaConf.register_new_resolver("eval", eval, replace=True)

## globals
# values taken from `gestures_diffuison_py/datasets/convert_dataset.py`
# whether to enable debug mode
DEBUG = False

# whether to use the custom num of inference steps during denoising (generally recommended)
USE_CUSTOM_INFERENCE_STEPS = True

# paper uses 16 for real-time control, higher nb = more accurate waypoints w.r.t. the inputs. 
# WARNING: if this value is less than the training value, then you MUST use a DDIM scheduler instead of a DDPM scheduler (you have to replace the policy's scheduler in this script)
NUM_CUSTOM_INFERENCE_STEPS = 75


USE_MAX_ACTION_STEPS = True

# not a good idea, I don't recommend it. Was used for troubleshooting.
USE_DEFAULT_STATE = False


ANALYTICS = True


TEST_REPEATED_HISTORY = False


USE_FINGERTIP_POS = True

# rosbag convert uses height, width, channels
IMAGE_SHAPE = (192, 192, 3)

# rosbag convert uses height, width, channels
IMAGE2_SHAPE = (180, 320, 3)

# CURRENTLY UNUSED
# CROP_WIDTH = 1
# CROP_HEIGHT = 35 # to remove Davin from the frame

# TODO: just set this from the policy
OUTPUT_IMAGE_W = 192
OUTPUT_IMAGE_H = 192


OBSERVATION_TOPIC = '/dexnex_projects_common/leftcam/image_resized'
OBSERVATION_TOPIC_2 = '/diffusion/wrist_cam/image_raw'
STATE_TOPIC = '/avatar/joint_states'
HAPTICS_TOPIC = '/biotac/lh/normalized'

# seconds
INFERENCE_DT = 2.5

# Hz
INFERENCE_FREQUENCY = 1. / INFERENCE_DT

# Hz, from convert_dataset_rosbag.py. Very important that this value is the same as the dataset
DATA_FREQUENCY = 3

# out of 15 total action steps. Consider, inference takes about 0.1 - 0.3 seconds depending on num_inference_steps. Take into account the data frequency and you can calculate the nb waypoints to skip. Example: DATA_FREQUENCY = 10.0, NUM_CUSTOM_INFERENCE_STEPS = 50. About 0.3s inference. nb to skip ~= 0.3 / 0.1 = 3ish. Add on 1 to account for mechanical latency. I think at least 1 needs to be skipped from the command latency in the dataset. Works better with skipping 2
NB_WAYPOINTS_TO_SKIP = 2
NB_WAYPOINTS_TO_SKIP_TEST = 0
TEST_WAYPOINT_DT = 0.25

# 999 takes all
NB_WAYPOINTS_TO_KEEP = 99

# whether to average waypoints
AVERAGE_WAYPOINTS = False

# how many average waypoints to calculate
NB_AVERAGING_WAYPOINTS = 10

CHECKPOINT_DIR = "/home/omnid/dexnex/libraries/diffusion_policy/outputs/2025-04-18/13-51-01/checkpoints"

# checkpoint name
CHECKPOINT_NAME = "epoch=0075-train_loss=0.020.ckpt"

# concat
CHECKPOINT_PATH = CHECKPOINT_DIR + "/" + CHECKPOINT_NAME

NODE_NAME = 'DiffusionPolicy'
NAMESPACE = 'diffusion'


JOINT_STATES_LENGTH = 66
INPUT_STATE_LENGTH = 35 # backwards compat
OUTPUT_ACTION_LENGTH = 21 # 6 gofa, 2 wrist, 5 th, 4 ff, 4 mf
JOINT_COMMAND_NAMES = [
"gofa1_joint_1",
"gofa1_joint_2",
"gofa1_joint_3",
"gofa1_joint_4",
"gofa1_joint_5",
"gofa1_joint_6",
"lh_WRJ2",
"lh_WRJ1",
"lh_FFJ4",
"lh_FFJ3",
"lh_FFJ2",
"lh_FFJ1",
"lh_MFJ4",
"lh_MFJ3",
"lh_MFJ2",
"lh_MFJ1",
"lh_THJ5",
"lh_THJ4",
"lh_THJ3",
"lh_THJ2",
"lh_THJ1",
] # correct order
SRDF_XML_PATH = '/home/omnid/dexnex/ws_avatar/src/avatar_master/avatar_moveit_config/config/avatar.srdf'
URDF_XML_PATH = '/home/omnid/dexnex/ws_avatar/src/avatar_master/avatar_moveit_config/config/avatar.urdf'

# Task Mask
TASK_MASK = np.zeros(JOINT_STATES_LENGTH, dtype=bool) # default False
TASK_MASK[0:6] = True # left gofa
TASK_MASK[6:8] = True # left wrist
TASK_MASK[8:12] = True # left ff
TASK_MASK[12:16] = True # left mf
TASK_MASK[25:30] = True # left th

# IF DEBUGGING
if DEBUG:
    INFERENCE_FREQUENCY = 0.2

class EvalDexNex(Node):
    def __init__(self):
        # init ROS node
        super().__init__(NODE_NAME, namespace=NAMESPACE)
        
        # declare ROS2 params
        # self.declare_parameter('input', , "Path to checkpoint") 
        # self.declare_parameter('inference_frequency', 10.0, "Control inference_frequency in Hz.")
        # self.declare_parameter('debug', False, "Whether to debug.")
        
        # # get ROS2 params
        # input = self.get_parameter('input')
        # inference_frequency = self.get_parameter('inference_frequency')
        # self.DEBUG = self.get_parameter('debug')
        self.data_frequency = DATA_FREQUENCY # Hz, from convert_dataset_rosbag.py. Very important that this value is the same as the dataset
            
        ## Analytics
        if ANALYTICS:
            self.ANALYTICS_telemetry_dt_ls = list()
        
        # load checkpoint
        payload = torch.load(open(CHECKPOINT_PATH, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        ## overwrite cfg values to speed up inference
        # cfg.pred_action_steps_only = True
            
            
        
        ## load the workspace
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        #
        self.cfg = cfg
        
        # cfg params
        use_ema = cfg.training.use_ema
        n_obs_steps = cfg.n_obs_steps
        self.n_obs_steps = n_obs_steps

        # diffusion model
        policy: BaseImagePolicy
        self.policy = workspace.model
        if use_ema:
            self.policy = workspace.ema_model
            
        # setup inference scheduler
        if USE_CUSTOM_INFERENCE_STEPS:
            cfg_scheduler = cfg.policy.noise_scheduler
            
            # setup DDIM scheduler
            scheduler = DDIMScheduler(
                num_train_timesteps=cfg_scheduler.num_train_timesteps,
                beta_start=cfg_scheduler.beta_start,
                beta_end=cfg_scheduler.beta_end,
                beta_schedule=cfg_scheduler.beta_schedule,
                clip_sample=cfg_scheduler.clip_sample,
                set_alpha_to_one=True, # taken from train_diffusion_unet_real_hybrid_workspace.yaml
                steps_offset=0, # taken from train_diffusion_unet_real_hybrid_workspace.yaml
                prediction_type=cfg_scheduler.prediction_type
                )
            
            scheduler.set_timesteps(NUM_CUSTOM_INFERENCE_STEPS)
            
            # replace the policy's scheduler
            self.policy.noise_scheduler = scheduler
            
            # override policy's num inference steps
            self.policy.num_inference_steps = NUM_CUSTOM_INFERENCE_STEPS

        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)
            
        if USE_MAX_ACTION_STEPS:
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
            
        print("n_action_steps: {}".format(self.policy.n_action_steps))

        # setup experiment
        dt = 1/INFERENCE_FREQUENCY

        print("n_obs_steps: ", n_obs_steps)
        
        # default values, one time step
        if USE_DEFAULT_STATE:
            self.m_joint_states_msg = JointState()
            self.m_joint_states_msg.position = np.zeros(JOINT_STATES_LENGTH)
            self.m_haptics = np.zeros(5) # allow default haptics data for ease of testing
            # self.m_image = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_NB_CHANNELS))
            # self.m_image2 = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_NB_CHANNELS)) # update with image2 shape
        else:
            self.m_joint_states_msg = None
            self.m_haptics = np.zeros(5) # allow default haptics data for ease of testing
            self.m_image = None # np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NB_CHANNELS))
            self.m_image2 = None
            
        # TEST
        if True:
            self.m_image2 = np.zeros(IMAGE2_SHAPE[0] * IMAGE2_SHAPE[1] * IMAGE2_SHAPE[2]) # update with image2 shape
            
        self.m_image_msg = Image()
        self.m_image2_msg = Image()
        
        # observation history
        self.image_history = deque(maxlen=n_obs_steps) # use deque instead of queue because it has maxlen
        self.image2_history = deque(maxlen=n_obs_steps) # use deque instead of queue because it has maxlen
        self.state_history = deque(maxlen=n_obs_steps) # use deque instead of queue because it has maxlen
        
        # ensure policy is reset
        with torch.no_grad():
            self.policy.reset() # don't think this actually does anything for a Unet policy
        
        ## ROS2 setup
        # QoS, required for image compat
        qos_profile = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        # subs
        self.sub1_ = self.create_subscription(JointState, STATE_TOPIC, self.SubJointStates, 1)
        
        self.sub2_ = self.create_subscription(Image, OBSERVATION_TOPIC, self.SubImage, qos_profile)
        
        self.sub_image2_ = self.create_subscription(Image, OBSERVATION_TOPIC_2, self.SubImage2, qos_profile)
        
        self.sub3_ = self.create_subscription(BiotacNormalized, HAPTICS_TOPIC, self.SubHaptics, qos_profile)
        
        # pubs
        self.pub_trajectory = self.create_publisher(JointTrajectory, "~/out/joint_trajectory", 10)
        
        # timer
        self.timer_ = self.create_timer(dt, self.Run)
        
        if False:
            self.timer2_ = self.create_timer(1.0 / self.data_frequency, self.TimerRosData) # MUST be ran at the same rate as the difference between states/obs that the policy was trained on.
        
        # moveit, if FK is needed
        if USE_FINGERTIP_POS:
            # load the robot model for FK
            self.robot_model = RobotModel(URDF_XML_PATH, SRDF_XML_PATH)
            self.robot_state = RobotState(self.robot_model)
            self.robot_state.set_to_default_values()
        
        # save ros data for n_obs_steps times so that our deques aren't empty and have the correct data format
        if USE_DEFAULT_STATE:
            for _ in range(n_obs_steps):
                self.SaveRosData()
    
    """ Use a separate sub in case the publish rate differs from our inference_frequency """
    def SubJointStates(self, msg):
        self.m_joint_states_msg = msg
        
    def SubImage(self, msg):
        self.m_image_msg = msg
        
        # put msg into an np.array
        img_np = np.array(msg.data, dtype="float")
        
        self.m_image = img_np
        
    def SubImage2(self, msg):
        self.m_image2_msg = msg
        
        # put msg into an np.array
        img_np = np.array(msg.data, dtype="float")
        
        self.m_image2 = img_np
        
    def SubHaptics(self, msg):
        self.m_haptics = msg.values[:]
        
    def ImageProc(self, imp_np, shape):
        # reshape the image
        obs_image_data_np_reshaped = np.reshape(imp_np, shape)
        
        # rosbag-convert uses height, width, channels
        h, w, c = shape
        
        # # crop the image -- skip for now
        obs_image_data_np_cropped = obs_image_data_np_reshaped
        # obs_image_data_np_cropped = obs_image_data_np_reshaped[CROP_HEIGHT:-CROP_HEIGHT, CROP_WIDTH:-CROP_WIDTH, :]
        
        # resize the image. Ok technically rosbag-convert uses (w, h) but since the policy input is a sq image it doesn't matter. But I should update future rosbag-converts
        obs_image_data_np_resized = cv2.resize(obs_image_data_np_cropped, dsize=(OUTPUT_IMAGE_H, OUTPUT_IMAGE_W), interpolation=cv2.INTER_CUBIC)
        
        return obs_image_data_np_resized
        
    """ Preprocess the raw ros image data. Basically the same as what I have to do in `convert_dataset.py` """
    def PreProcessRosImgData(self, img_np, shape):
        # extract parameters
        policy_img_shape = self.cfg.shape_meta.obs.image.shape
        
        # policy_img_shape example: [3, 96, 96]
        policy_w = policy_img_shape[1]
        policy_h = policy_img_shape[2]
        
        assert(policy_w == OUTPUT_IMAGE_W)
        assert(policy_h == OUTPUT_IMAGE_H)
        
        # do the same image proc as in `convert_dataset_rosbag.py`
        img_np_procd = self.ImageProc(img_np, shape)
        
        return img_np_procd
    
    def TimerRosData(self):
        # if we want to repeat the most recent state in our history
        if TEST_REPEATED_HISTORY:
            for _ in range(self.cfg.n_obs_steps):
                self.SaveRosData()
                
        else:
            self.SaveRosData()
        
    """ Take the asynch raw ROS2 messages and save them for use in our policy. MUST BE RAN AT THE SAME RATE AS THE DATASET HISTORY DELAY """
    def SaveRosData(self):
        ## State
        # ensure telemetry is flowing
        if self.m_haptics is None or self.m_joint_states_msg is None:
            print("No haptics or joint states yet.")
        else:
            if USE_FINGERTIP_POS:
                # set the robot state from the most recent message
                self.robot_state.joint_positions = dict(zip(self.m_joint_states_msg.name, self.m_joint_states_msg.position))
                
                # force an update, else the transforms won't change
                self.robot_state.update()
                
                # get the fingertip FK. get_global_link_transform outputs a 4x4 affine transformation matrix, so the last column contains the position
                th_pos = self.robot_state.get_global_link_transform("lh_thtip")[:3, 3]
                ff_pos = self.robot_state.get_global_link_transform("lh_fftip")[:3, 3]
                mf_pos = self.robot_state.get_global_link_transform("lh_mftip")[:3, 3]
                
                # assemble the full state
                state = np.concatenate((
                    np.array(self.m_joint_states_msg.position)[TASK_MASK],
                    self.m_haptics,
                    th_pos,
                    ff_pos,
                    mf_pos,
                ))
                
                # backwards compat
                state = state[:INPUT_STATE_LENGTH]
            else:
                # assemble the full state
                state = np.concatenate((
                    np.array(self.m_joint_states_msg.position)[TASK_MASK],
                    self.m_haptics,
                ))
            #endif
            
            # ensure correct dtype (float)
            state = state.astype(np.float32, copy=False)
            
            # push ROS2 data to our queues
            self.state_history.appendleft(state)
        
        ## Image
        if self.m_image is None:
            print("No obs image yet.")
        else:
            # pre process the raw ROS img
            img_np_resized = self.PreProcessRosImgData(self.m_image, IMAGE_SHAPE)
            
            self.image_history.appendleft(img_np_resized)
                
            if ANALYTICS:
                # save image time
                t1 = rclpy.time.Time.from_msg(self.m_image_msg.header.stamp)
                self.ANALYTICS_telemetry_dt_ls.append(t1.nanoseconds / 1e9)
        
        ## Image2
        if self.m_image2 is None:
            print("No obs image2 yet.")
        else:
            # pre process the raw ROS img
            img_np_resized = self.PreProcessRosImgData(self.m_image2, IMAGE2_SHAPE)
            
            self.image2_history.appendleft(img_np_resized)
        
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
        img2s_ls = self.DequeToList(self.image2_history)
        states_ls = self.DequeToList(self.state_history)
            
        # convert to np
        imgs_np = np.stack(imgs_ls)
        img2s_np = np.stack(img2s_ls)
        states_np = np.stack(states_ls)
        
        # same preprocessing as dexnex_pusht_image_dataset.py.
        image = np.moveaxis(imgs_np, -1, 1) / 255.0
        image2 = np.moveaxis(img2s_np, -1, 1) / 255.0
        
        # construct output dictionary
        obs_dict_np = {
                'image': image,
                'image2': image2,
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
            
            # must normalize ourselves, here, because of the way co-training works with separate normalizers per dataset
            nobs_dict = self.policy.normalizer.normalize(obs_dict)
            
            # inside predict_action -> conditional_sample is where the iteration occurs. `for t in scheduler.timesteps`
            nresult = self.policy.predict_action(nobs_dict)
            naction_pred = nresult["naction_pred"]
            naction = nresult["naction"]
            
            # unnormalize
            action_pred = self.policy.normalizer['action'].unnormalize(naction_pred)
            action = self.policy.normalizer['action'].unnormalize(naction)
            
            result = {
                'action': action,
                'action_pred': action_pred
            }
            
            if DEBUG or ANALYTICS:
                print('Inference latency:', time.time() - s)
        
            return result

    """  """
    def PublishTrajectory(self, results):
        
        # extract full action from policy output
        # action_pred = results['action_pred'] # horizon long (16), on GPU
        action = results['action'] # n_action_steps long (8), on GPU. First action is at the current time
        
        # take the first index to remove the batch axis. Transfer to CPU and numpy. 
        action_cpu_np = action[0].detach().to('cpu').numpy()
        avg_np = np.zeros((NB_AVERAGING_WAYPOINTS, OUTPUT_ACTION_LENGTH))
        # avg_np = np.zeros((1, OUTPUT_ACTION_LENGTH))
        
        # # remove the first half of the traj because it's usually too far behind and cause a positive feedback loop of undesirable behavior
        # 
        actions_to_avg = action_cpu_np[NB_WAYPOINTS_TO_SKIP:NB_WAYPOINTS_TO_SKIP+NB_WAYPOINTS_TO_KEEP]
        
        
        if AVERAGE_WAYPOINTS:
            nb_pts_per = int(np.floor(actions_to_avg.shape[0] / NB_AVERAGING_WAYPOINTS))
            
            for i in range(NB_AVERAGING_WAYPOINTS):
                # average across waypoints
                idx = i * nb_pts_per
                avg_np[i] = actions_to_avg[idx:idx+nb_pts_per].mean(axis=0)
                
            actions_to_publish = avg_np
            
        else:
            actions_to_publish = actions_to_avg
        
        # publish joint state trajectory to ROS2
        ### All joints
        msg = JointTrajectory()
        msg.points = []
        msg.joint_names = JOINT_COMMAND_NAMES
        
        # iterate over the trajectory step dimension
        for idx in range(actions_to_publish.shape[0]):
            point = actions_to_publish[idx]
            
            msg2 = JointTrajectoryPoint()
            
            # first 6 are for the arm
            msg2.positions = point[:] # copy
            
            msg.points.append(msg2)
            
        # use the joint states msg stamp. Should be similar enough to the image headers. Don't have to worry about the msg updating since this is a blocking call
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.stamp = self.m_joint_states_msg.header.stamp
        self.pub_trajectory.publish(msg)
        

    """ Ran on a timer """
    def Run(self):
        # TEST. just save the most recent data here instead of using the timer
        if True:
            self.SaveRosData()
        
        # check that we have states and observations
        if len(self.state_history) != self.n_obs_steps or len(self.image_history) != self.n_obs_steps:
            print("No state or obs yet.")
            return
        
        # get observation
        obs_dict_np = self.GetObs()
        
        # run inference
        result = self.RunInference(obs_dict_np)
        
        # publish results
        if False:
            # TEST - send each waypoint separately. It helps with sync'ing the shadow hand and the arm when using JTC & my custom simple traj player.
            for i in range(NB_WAYPOINTS_TO_SKIP_TEST, NB_WAYPOINTS_TO_KEEP, 1):
                action = result['action']
                dd = {'action': action[:, i:i+1, ...]}
                self.PublishTrajectory(dd)
                time.sleep(TEST_WAYPOINT_DT)
        else:
            self.PublishTrajectory(result)
        
        # analytics
        if ANALYTICS:
            self.CalculateAnalytics()
            
    def CalculateAnalytics(self):
        if len(self.ANALYTICS_telemetry_dt_ls) > 2:
            arr = np.array(self.ANALYTICS_telemetry_dt_ls)
            dts = np.abs(arr[0:-1] - arr[1:])
            dt = np.mean(dts)
            print("Average image msg SaveRos dt: {}".format(dt))
            
            # reset list
            self.ANALYTICS_telemetry_dt_ls = []
                
    def Test(self):
        plt.figure()
        
        while rclpy.ok():
            # spin some
            rclpy.spin_once(self, timeout_sec=0.05)
            
            # plot
            if len(self.image_history) > 0:
                vis_img2 = np.clip(self.image_history[0] / 255.0, 0.0, 1.0) # othewise plt will print a bunch of annoying warnings
                plt.imshow(vis_img2)
                plt.show(block=False)
                plt.pause(0.05)
            
        
        
def main(args=None):
    rclpy.init(args=args)

    node = EvalDexNex()

    # node.Test()
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()