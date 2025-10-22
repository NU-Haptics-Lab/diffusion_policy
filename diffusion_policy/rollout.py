from diffusion_policy import utils
import avatar_drake_sim.rl.sac
from avatar_drake_sim.utils.utils import load_yaml_config
from diffusion_policy.common.replay_buffer import ReplayBuffer
import diffusion_policy.globals as globals
from evaluation.eval import EvalDexNex
from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.common.sarsa_sampler import DatasetSampler
from diffusion_policy.dataset.train_and_val import TrainAndVal

import numpy as np
import gymnasium as gym
import torch

import yaml
from omegaconf import OmegaConf

class RolloutEnv:
    def __init__(self) -> None:
        pass
    
    def setup(self):
        """
        Set up the Dexnex Drake Sim env
        """
        pass

class Rollout:
    def __init__(self,
                 evaluator: EvalDexNex,
                 freq = 10,
                 num_rollouts_per_trigger = 10,
                 rb_id = "sim_online_rl",
                 ) -> None:
        self.evaluator = evaluator
        self.freq = freq
        self.num_rollouts_per_trigger = num_rollouts_per_trigger
        self.rb_id = rb_id
        
        self.setup()
        
    def setup(self):
        # Register environment with gym globally
        gym.envs.register(id="DexnexGym-v0", entry_point="avatar_drake_sim.rl.dexnex_gym:DexnexGymEnv") #type:ignore

        class Args:
            task = "box-and-blocks"
            debug = globals.CONFIG.debug #type:ignore
            test = globals.CONFIG.debug #type:ignore
            profile = False
        
        configs = load_yaml_config()
        configs['_rl_temp'] = {}        # Only way to pass additional variables from DexnexGym creation
        USERNAME = 'alienware'
        configs['filepaths'] = configs['filepaths'][USERNAME]
        
        if not configs['rl_params']['use_rgb'] and configs['rl_params']['use_depth']:
            print("Training with depth images and not RGB is currently not supported.") 
            return
        
        # If not using left hand, lock left hand joints and leave out of action space
        if not configs['rl_params']['use_contact_forces']:
            if 'lh_' not in configs['task_params']['lock_targets']: configs['task_params']['lock_targets'].append('lh_')
            if 'rh_' not in configs['task_params']['lock_targets']: configs['task_params']['lock_targets'].append('rh_')
            
        
        self.env = gym.make("DexnexGym-v0", configs=configs, args=Args)
        
        # setup the evaluator
        # get the model
        actor: ModelEmaOptim = globals.MODELS["actor"] #type:ignore
        policy = actor.get_model()
        self.evaluator.set_policy(policy)
    
    def run(self):
        """
        Run one portion of rollout
        """
        
        # if our number is called
        if utils.StepFreqTrigger(self.freq):
            
            # rollout n times per trigger
            for n in range(self.num_rollouts_per_trigger):
                # rollout_prep
                self.rollout_prep()
            
                # one rollout
                samples = self.one_rollout()
                
                # dump the samples to the replay buffer
                self.save_episode(samples)
                
            # must re-index the sampler
            sampler: TrainAndVal = globals.DATALOADERS[self.rb_id]
            sampler.init()
                
    def rollout_prep(self):
        # update the eval class
        
        # reset the sim class
        obs, info = self.env.reset()
                
        # convert the obs
        state, img, img2 = self.convert_obs(obs)
                
        self.save_obs(state, img, img2)
        
        # update the evaluator
        self.evaluator.save_data(
            state,
            img,
            img2
        )
        
        pass
    
    def save_obs(self, state, img, img2):
        self.state = state
        self.img = img
        self.img2 = img2
    
    def step_trajectory(self, actions):
        total_reward = 0.0
        samples = []
        
        for action in actions:
        # step the env
            new_obs, rewards, terminated, truncated, infos = self.env.step(action) #type:ignore
            
            done = terminated or truncated
            
            total_reward += rewards
            
            if done:
                break
            
        return new_obs, total_reward, done, infos
        
    def one_rollout(self):
        """
        Run one rollout
        """
        samples = []
        done = False
        while not done:
            # get the action trajectory
            actions = self.evaluator.infer()
            actions = torch.squeeze(actions)
            actions = actions.numpy()
            
            # none protection
            if actions is not None:
                # execute the full trajectory
                new_obs, rewards, dones, infos = self.step_trajectory(actions)
                
                self.save_samples(actions, rewards, samples)
                
                # convert the obs
                state, img, img2 = self.convert_obs(new_obs)
                
                self.save_obs(state, img, img2)
                
                # update the evaluator
                self.evaluator.save_data(
                    state,
                    img,
                    img2
                )
                
                if dones:
                    done = True
                
        return samples
    
    def save_samples(self, actions, rewards, samples):
        # save the sample using the old obs, current action, current reward
        data = {
            'state': self.state,
            'img': self.img,
            'img2': self.img2,
            'action': actions,
            'reward': np.float32(rewards),
        }
        
        to_save = data
        # for key in globals.CONFIG.obs_keys_to_load:
        #     to_save[key] = data[key]
        
        # append the data
        samples.append(to_save)
            
    def convert_obs(self, obs):
        dd = {}
        
        """
        sim::avatar_gofa1_joint_1_q
        sim::avatar_gofa1_joint_2_q
        sim::avatar_gofa1_joint_3_q
        sim::avatar_gofa1_joint_4_q
        sim::avatar_gofa1_joint_5_q
        sim::avatar_gofa1_joint_6_q
        sim::avatar_lh_WRJ2_q
        sim::avatar_lh_WRJ1_q
        sim::avatar_lh_FFJ4_q
        sim::avatar_lh_FFJ3_q
        sim::avatar_lh_FFJ2_q
        sim::avatar_lh_FFJ1_q
        sim::avatar_lh_MFJ4_q
        sim::avatar_lh_MFJ3_q
        sim::avatar_lh_MFJ2_q
        sim::avatar_lh_MFJ1_q
        sim::avatar_lh_THJ5_q
        sim::avatar_lh_THJ4_q
        sim::avatar_lh_THJ3_q
        sim::avatar_lh_THJ2_q
        sim::avatar_lh_THJ1_q
        sim::avatar_gofa1_joint_1_w
        sim::avatar_gofa1_joint_2_w
        sim::avatar_gofa1_joint_3_w
        sim::avatar_gofa1_joint_4_w
        sim::avatar_gofa1_joint_5_w
        sim::avatar_gofa1_joint_6_w
        sim::avatar_lh_WRJ2_w
        sim::avatar_lh_WRJ1_w
        sim::avatar_lh_FFJ4_w
        sim::avatar_lh_FFJ3_w
        sim::avatar_lh_FFJ2_w
        sim::avatar_lh_FFJ1_w
        sim::avatar_lh_MFJ4_w
        sim::avatar_lh_MFJ3_w
        sim::avatar_lh_MFJ2_w
        sim::avatar_lh_MFJ1_w
        sim::avatar_lh_THJ5_w
        sim::avatar_lh_THJ4_w
        sim::avatar_lh_THJ3_w
        sim::avatar_lh_THJ2_w
        sim::avatar_lh_THJ1_w
        
        lh_fftip 42:45
        lh_mftip 45:48
        lh_rftip 48:51
        lh_lftip 51:54
        lh_thtip 54:57
        """
        state = obs['state'][0]
        haptics = obs['lh_contact_forces'][0]
        ff_pos = state[42:45]
        mf_pos = state[45:48]
        th_pos = state[54:57]
        
        out_state = np.concatenate((
            state[0:21],
            haptics,
            th_pos,
            ff_pos,
            mf_pos,
            ))
        
        # make a write-able copy
        left_cam_rgb = obs['left_cam_rgb'].copy()
        
        # convert to float
        left_cam_rgb = np.array(left_cam_rgb, dtype='float')
        
        # must moveaxis because that bug is still in the gen dataset script
        img = np.moveaxis(left_cam_rgb, -1, 1)
        img /= 255.0
        
        # currently no wrist cam is active
        img2 = np.zeros_like(img)
        
        return out_state, img, img2
        
            
    def save_episode(self, episode):
        if len(episode) > 0:
            # basically unzip the list of dicts and put into an np array
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            
            # use the replay buffer to write to disk
            rb: ReplayBuffer = globals.REPLAY_BUFFER_LOADER[self.rb_id] # type:ignore
            rb.add_episode(data_dict, compressors='disk')
            
    
    def save_data(self,
                  image_,
                  image2_,
                  robot_state,
                  avatar_state_,
                  episode,
                  administer_reward,
                  ):
        # check for image data
        if len(image_.data) == 0 or len(image2_.data) == 0:
            return
        
        # assemble the observation state
        # using https://github.com/NU-Haptics-Lab/DexNexSimulationIntegration/issues/63
        d = np.array(avatar_state_.data)
        
        joint_state = np.concatenate((
            d[0:16], # gofa, wr, ff, mf
            d[25:30] # th
        )) # correct order
        
        ff_pos = d[158:161]
        mf_pos = d[165:168]
        th_pos = d[186:189]
        
        obs_state_data_np = np.concatenate((
            joint_state,
            biotac_.values[:],
            th_pos,
            ff_pos,
            mf_pos,
            ))
        
        # actions only include the gofa, wr, ff, th
        action_state_data_np = np.array(joint_cmd_.position)[TASK_MASK]
        
        ## Image Proc
        obs_image_data_np_reshaped = proc_img(image_, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NB_CHANNELS))
        img2 = proc_img(image2_, (CAM2_H, CAM2_W, IMAGE_NB_CHANNELS))
                
        # crop the image
        obs_image_data_np_cropped = obs_image_data_np_reshaped
        
        img_moveaxis = resize_img(obs_image_data_np_cropped)
        img2_moveaxis = resize_img(img2)
        
        ## construct output vars
        output_img = img_moveaxis
        output_img2 = img2_moveaxis
        output_state = obs_state_data_np
        output_action = action_state_data_np
        
        # reward logic -- +1.0 if a block was transferred, -0.1 otherwise
        reward = 0.0
        if administer_reward:
            reward += reward_value
            
            # reward has been captured, so we can now deactivate this flag
            administer_reward = False
        
        # construct the data dict
        data = {
            'state': np.float32(output_state),
            'img': output_img,
            'img2': output_img2,
            'action': np.float32(output_action),
            'reward': np.float32(reward),
        }
        
        # append the data to the episode
        episode.append(data)