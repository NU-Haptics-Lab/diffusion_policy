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

class EvalDexNex(Node):
    def __init__(self,
                 debug,
                 test,
                 node_name,
                 namespace,
                 data_frequency,
                 analytics,
                 use_custom_inference_steps,
                 num_custom_inference_steps,
                 use_max_action_steps,
                 use_default_state,
                 test_repeated_history,
                 use_fingertip_pos,
                 ros_image_shape,
                 ros_image2_shape,
                 torch_image_w,
                 torch_image_h,
                 observation_topic,
                 observation_topic_2,
                 state_topic,
                 haptics_topic,
                 inference_dt,
                 nb_waypoints_to_skip,
                 nb_waypoints_to_skip_test,
                 test_waypoint_dt,
                 nb_waypoints_to_keep,
                 average_waypoints: bool,
                 nb_averaging_waypoints,
                 joint_states_length,
                 input_state_length,
                 output_action_length,
                 joint_command_names,
                 srdf_xml_path,
                 urdf_xml_path,
                 use_ema,
                 noise_scheduler: DDIMScheduler,
                 batch_loader: BatchLoader,
                 ):
        # save inputs
        self.debug = debug
        self.test = test
        self.node_name = node_name
        self.namespace = namespace
        self.data_frequency = data_frequency
        self.analytics = analytics
        self.use_custom_inference_steps = use_custom_inference_steps
        self.num_custom_inference_steps = num_custom_inference_steps
        self.use_max_action_steps = use_max_action_steps
        self.use_default_state = use_default_state
        self.test_repeated_history = test_repeated_history
        self.use_fingertip_pos = use_fingertip_pos
        self.ros_image_shape = ros_image_shape
        self.ros_image2_shape = ros_image2_shape
        self.torch_image_w = torch_image_w
        self.torch_image_h = torch_image_h
        self.observation_topic = observation_topic
        self.observation_topic_2 = observation_topic_2
        self.state_topic = state_topic
        self.haptics_topic = haptics_topic
        self.inference_dt = inference_dt
        self.nb_waypoints_to_skip = nb_waypoints_to_skip
        self.nb_waypoints_to_skip_test = nb_waypoints_to_skip_test
        self.test_waypoint_dt = test_waypoint_dt
        self.nb_waypoints_to_keep = nb_waypoints_to_keep
        self.average_waypoints = average_waypoints
        self.nb_averaging_waypoints = nb_averaging_waypoints
        self.joint_states_length = joint_states_length
        self.input_state_length = input_state_length
        self.output_action_length = output_action_length
        self.joint_command_names = joint_command_names
        self.srdf_xml_path = srdf_xml_path
        self.urdf_xml_path = urdf_xml_path
        self.use_ema = use_ema
        self.noise_scheduler = noise_scheduler
        self.batch_loader = batch_loader


        # calculated from input parameters
        self.inference_frequency = 1. / self.inference_dt 

        # Task Mask
        self.task_mask = np.zeros(self.joint_states_length, dtype=bool) # default False
        self.task_mask[0:6] = True # left gofa
        self.task_mask[6:8] = True # left wrist
        self.task_mask[8:12] = True # left ff
        self.task_mask[12:16] = True # left mf
        self.task_mask[25:30] = True # left th
        
        # save n obs steps
        self.n_obs_steps = globals.CONFIG.models.models.actor.model.model.n_obs_steps

        # IF DEBUGGING
        if self.debug:
            self.inference_frequency = 0.2

        # init ROS node
        super().__init__(self.node_name, namespace=self.namespace)
        
        # declare ROS2 params
        # self.declare_parameter('input', , "Path to checkpoint") 
        # self.declare_parameter('inference_frequency', 10.0, "Control inference_frequency in Hz.")
        # self.declare_parameter('debug', False, "Whether to debug.")
        
        # # get ROS2 params
        # input = self.get_parameter('input')
        # inference_frequency = self.get_parameter('inference_frequency')
        # self.DEBUG = self.get_parameter('debug')
            
        ## Analytics
        if self.analytics:
            self.ANALYTICS_telemetry_dt_ls = list()
        
        # extract diffusion model
        # ema or not
        if self.use_ema:
            self.policy = globals.MODELS["actor"].get_ema_model()

        else:
            self.policy = globals.MODELS["actor"].get_model()

            
        # setup inference scheduler
        if self.use_custom_inference_steps:
            self.noise_scheduler.set_timesteps(self.num_custom_inference_steps)
            
            # replace the policy's scheduler
            self.policy.noise_scheduler = self.noise_scheduler
            
            # override policy's num inference steps
            self.policy.num_inference_steps = self.num_custom_inference_steps
            
        if self.use_max_action_steps:
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
            
        print("n_action_steps: {}".format(self.policy.n_action_steps))
        print("n_obs_steps: ", self.n_obs_steps)
        
        # default values, one time step
        if self.test or self.use_default_state:
            self.m_joint_states_msg = JointState()
            self.m_joint_states_msg.position = np.zeros(self.joint_states_length)
            self.m_haptics = np.zeros(5) # allow default haptics data for ease of testing
            # self.m_image = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_NB_CHANNELS))
            # self.m_image2 = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_NB_CHANNELS)) # update with image2 shape
        else:
            self.m_joint_states_msg = None
            self.m_haptics = np.zeros(5) # allow default haptics data for ease of testing
            self.m_image = None # np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NB_CHANNELS))
            self.m_image2 = None
            
        if self.test:
            self.m_image = np.zeros(np.prod(self.ros_image_shape))
            self.m_image2 = np.zeros(np.prod(self.ros_image2_shape))
            
        # TEST
        if True:
            self.m_image2 = np.zeros(self.ros_image2_shape[0] * self.ros_image2_shape[1] * self.ros_image2_shape[2]) # update with image2 shape
            
        self.m_image_msg = Image()
        self.m_image2_msg = Image()
        self.m_stamp = None
        
        # observation history
        self.image_history = deque(maxlen=self.n_obs_steps) # use deque instead of queue because it has maxlen
        self.image2_history = deque(maxlen=self.n_obs_steps) # use deque instead of queue because it has maxlen
        self.state_history = deque(maxlen=self.n_obs_steps) # use deque instead of queue because it has maxlen
        
        # ensure policy is reset
        with torch.no_grad():
            self.policy.reset() # don't think this actually does anything for a Unet policy
        
        # setup ros
        self.setup_ros()
        
        if False:
            self.timer2_ = self.create_timer(1.0 / self.data_frequency, self.TimerRosData) # MUST be ran at the same rate as the difference between states/obs that the policy was trained on.
        
        # moveit
        self.setup_moveit()
        
        # save ros data for n_obs_steps times so that our deques aren't empty and have the correct data format
        if self.test or self.use_default_state:
            for _ in range(self.n_obs_steps):
                self.SaveRosData()
        
    def setup_ros(self):
        ## ROS2 setup
        # QoS, required for image compat
        qos_profile = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        # subs
        self.sub1_ = self.create_subscription(JointState, self.state_topic, self.SubJointStates, 1)
        
        self.sub2_ = self.create_subscription(Image, self.observation_topic, self.SubImage, qos_profile)
        
        self.sub_image2_ = self.create_subscription(Image, self.observation_topic_2, self.SubImage2, qos_profile)
        
        self.sub3_ = self.create_subscription(BiotacNormalized, self.haptics_topic, self.SubHaptics, qos_profile)
        
        # pubs
        self.pub_trajectory = self.create_publisher(JointTrajectory, "~/out/joint_trajectory", 10)
        
        # timer
        self.timer_ = self.create_timer(self.inference_dt, self.Run)
        
    def setup_moveit(self):
        # moveit, if FK is needed
        if self.use_fingertip_pos:
            # load the robot model for FK
            self.robot_model = RobotModel(self.urdf_xml_path, self.srdf_xml_path)
            self.robot_state = RobotState(self.robot_model)
            self.robot_state.set_to_default_values()
    
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
        obs_image_data_np_resized = cv2.resize(obs_image_data_np_cropped, dsize=(self.torch_image_h, self.torch_image_w), interpolation=cv2.INTER_CUBIC)
        
        return obs_image_data_np_resized
        
    """ Preprocess the raw ros image data. Basically the same as what I have to do in `convert_dataset.py` """
    def PreProcessRosImgData(self, img_np, shape):
        # extract parameters
        policy_img_shape = globals.CONFIG.shape_meta.img.shape
        
        # policy_img_shape example: [3, 96, 96]
        policy_w = policy_img_shape[1]
        policy_h = policy_img_shape[2]
        
        assert(policy_w == self.torch_image_w)
        assert(policy_h == self.torch_image_h)
        
        # do the same image proc as in `convert_dataset_rosbag.py`
        img_np_procd = self.ImageProc(img_np, shape)
        
        return img_np_procd
    
    def TimerRosData(self):
        # if we want to repeat the most recent state in our history
        if self.test_repeated_history:
            for _ in range(self.n_obs_steps):
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
            if self.use_fingertip_pos:
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
                    np.array(self.m_joint_states_msg.position)[self.task_mask],
                    self.m_haptics,
                    th_pos,
                    ff_pos,
                    mf_pos,
                ))
                
                # backwards compat
                state = state[:self.input_state_length]
            else:
                # assemble the full state
                state = np.concatenate((
                    np.array(self.m_joint_states_msg.position)[self.task_mask],
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
            img_np_resized = self.PreProcessRosImgData(self.m_image, self.ros_image_shape)
            
            self.image_history.appendleft(img_np_resized)
                
            if self.analytics:
                # save image time
                t1 = rclpy.time.Time.from_msg(self.m_image_msg.header.stamp)
                self.ANALYTICS_telemetry_dt_ls.append(t1.nanoseconds / 1e9)
        
        ## Image2
        if self.m_image2 is None:
            print("No obs image2 yet.")
        else:
            # pre process the raw ROS img
            img_np_resized = self.PreProcessRosImgData(self.m_image2, self.ros_image2_shape)
            
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
        obs_dict_np_all = {
                'img': image,
                'img2': image2,
                'state': states_np, # T, state_length
            }
        
        obs_dict_np = {}
        
        # reduce to only the used obs keys
        for key, val in obs_dict_np_all.items():
            if key in globals.CONFIG.obs_keys_to_use:
                obs_dict_np[key] = val
        
        # done
        return obs_dict_np

    def NewEpisode(self):
        with torch.no_grad():
            self.policy.reset()

    def RunInference(self, obs_dict_np):
                
        # run inference
        with torch.no_grad():
            s = time.time()
            
            # wrap in a data dict, which is what the batch loader expects
            dd = {"obs": obs_dict_np}
            
            # put into 
            dd_torch = pytorch_util.dict_to_torch(dd)
            
            # must normalize ourselves, here, because of the way co-training works with separate normalizers per dataset
            ndd_torch = self.batch_loader.transfer_and_norm(dd_torch)
            
            nobs_torch = ndd_torch['obs']
            
            # inside predict_action -> conditional_sample is where the iteration occurs. `for t in scheduler.timesteps`
            nresult_gpu = self.policy.predict_action(nobs_torch)

            # naction doesn't include past actions
            naction_gpu = nresult_gpu["naction"]

            # must wrap in a dict
            naction_gpu_dd = {"action": naction_gpu}
            
            # unnormalize
            action_dd = self.batch_loader.unnorm_and_transfer(naction_gpu_dd)
                        
            action = action_dd['action']
            
            if self.debug or self.analytics:
                print('Inference latency:', time.time() - s)
        
            return action

    """  """
    def PublishTrajectory(self, action): 
        # cast into numpy. 
        action = action.numpy()   
        
        # squeeze the first index to remove the batch axis. 
        action = np.squeeze(action)
        
        avg_np = np.zeros((self.nb_averaging_waypoints, self.output_action_length))
        # avg_np = np.zeros((1, OUTPUT_ACTION_LENGTH))
        
        # # remove the first half of the traj because it's usually too far behind and cause a positive feedback loop of undesirable behavior
        # 
        actions_to_avg = action[self.nb_waypoints_to_skip:self.nb_waypoints_to_skip + self.nb_waypoints_to_keep]
        
        
        if self.average_waypoints:
            nb_pts_per = int(np.floor(actions_to_avg.shape[0] / self.nb_averaging_waypoints))
            
            for i in range(self.nb_averaging_waypoints):
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
        msg.joint_names = self.joint_command_names
        
        # iterate over the trajectory step dimension
        for idx in range(actions_to_publish.shape[0]):
            point = actions_to_publish[idx]
            
            msg2 = JointTrajectoryPoint()
            
            # first 6 are for the arm
            msg2.positions = point[:] # copy
            
            msg.points.append(msg2)
            
        # use the joint states msg stamp. Should be similar enough to the image headers. Don't have to worry about the msg updating since this is a blocking call
        # msg.header.stamp = self.get_clock().now().to_msg()
        self.m_stamp = self.m_joint_states_msg.header.stamp # THIS IS THE SAME STAMP AS WHEN THE DIFFUSION POLICY BEGAN EVALUATION, see comment above
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
        action = self.RunInference(obs_dict_np)
        
        # publish results
        if False:
            # TEST - send each waypoint separately. It helps with sync'ing the shadow hand and the arm when using JTC & my custom simple traj player.
            for i in range(NB_WAYPOINTS_TO_SKIP_TEST, NB_WAYPOINTS_TO_KEEP, 1):
                action = result['action']
                dd = {'action': action[:, i:i+1, ...]}
                self.PublishTrajectory(dd)
                time.sleep(TEST_WAYPOINT_DT)
        else:
            self.PublishTrajectory(action)
        
        # analytics
        if self.analytics:
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
    

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config')),
)
def main(eval_cfg: OmegaConf, args=None):
    # make checkpoint path
    checkpoint_path = os.path.join(eval_cfg.checkpoint_dir, eval_cfg.checkpoint_name)
    
    # load the training checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    
    # extract training config
    training_cfg = payload['cfg']
    
    # save training config into the global config
    globals.CONFIG = training_cfg
    
    # Temporarily disable strict mode to add new keys
    OmegaConf.set_struct(globals.CONFIG, False)
    
    # modify some values for evaluation
    # globals.CONFIG.spin_replay_buffer = False # TODO: add to training config
    # globals.CONFIG.spin_dataloaders = False # TODO: add to training config
    # globals.CONFIG.spin_session_trainer = False
    globals.CONFIG.replay_buffer_loader.do_loading = False
    
    # whether we're debugging
    if eval_cfg.debug:
        pass
    
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(globals.CONFIG)
    
    # apply overrides, much faster than merge
    OmegaConf.unsafe_merge(globals.CONFIG, globals.CONFIG.override)
    print("Config merged.")
        
    # spin up the logger
    globals.LOGGER = hydra.utils.instantiate(globals.CONFIG.logging)
    print("Logger spun.")
    
    # spin up the replay buffer loader
    globals.REPLAY_BUFFER_LOADER = hydra.utils.instantiate(globals.CONFIG.replay_buffer_loader)
    print("Replay Buffer Loader spun.")
        
    # spin up the models
    globals.MODELS = hydra.utils.instantiate(globals.CONFIG.models)
    print("Models spun.")
    
    # spin up the checkpointer
    globals.CHECKPOINTER = hydra.utils.instantiate(globals.CONFIG.checkpoint)
    print("Checkpointer spun.")
    
    # if resuming, load
    globals.CHECKPOINTER.load()
    
    ### ROS2
    rclpy.init(args=args)

    # spin up the ros2 class
    node = hydra.utils.instantiate(eval_cfg.eval_dexnex)
    print("Node created.")

    # node.Test()
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    ### End ROS2


if __name__ == "__main__":
    # only need the following if using my meta dataset
    # torch.multiprocessing.set_start_method('spawn') # or 'forkserver'
    main()