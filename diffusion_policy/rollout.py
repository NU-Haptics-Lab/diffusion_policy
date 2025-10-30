from diffusion_policy import utils
import avatar_drake_sim.rl.sac
from avatar_drake_sim.utils.utils import load_yaml_config
from diffusion_policy.common.replay_buffer import ReplayBuffer
import diffusion_policy.globals as globals
from evaluation.eval import EvalDexNex
from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.common.sarsa_sampler import DatasetSampler
from diffusion_policy.dataset.train_and_val import TrainAndVal

from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss
from diffusion_policy.common import pytorch_util

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
                 use_online_rollout = True,
                 rollout_num_actions = 10,
                 use_critic_preferred_actions = False,
                 use_critic_preferred_actions_early_exit = True,
                 critic_preferred_actions_nb = 5,
                 ) -> None:
        self.evaluator = evaluator
        self.freq = freq
        self.num_rollouts_per_trigger = num_rollouts_per_trigger
        self.rb_id = rb_id
        self.use_online_rollout = use_online_rollout
        self.rollout_num_actions = rollout_num_actions
        self.use_critic_preferred_actions = use_critic_preferred_actions
        self.use_critic_preferred_actions_early_exit = use_critic_preferred_actions_early_exit
        self.critic_preferred_actions_nb = critic_preferred_actions_nb

        # refs
        self.critic = None

        self.setup()
        
    def setup(self):
        if self.use_online_rollout:
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
            
            # ema or regular model?
            policy = actor.get_ema_model()
            self.evaluator.set_policy(policy)

            # get the critic
            critic: CriticLoss = globals.MODELS["critic"] #type:ignore
            self.critic = critic.get_model()
            self.critic_ops = critic.critic.MakeOptions(self.evaluator.task_id)

    
    def run(self):
        """
        Run one portion of rollout
        """
        
        # if our number is called
        if self.use_online_rollout and utils.StepFreqTrigger(self.freq):
            
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
    
    def save_obs(self, state, img, img2):
        self.state = state
        self.img = img
        self.img2 = img2
    
    def step_trajectory(self, actions):
        total_reward = 0.0
        samples = []
        
        new_obs = None
        done = False
        infos = None
        
        m = min(len(actions), self.rollout_num_actions)
        for i in range(m):
            action = actions[i]
            # step the env
            new_obs, rewards, terminated, truncated, infos = self.env.step(action) #type:ignore
            
            # save sample using the old obs
            self.save_samples(action, rewards, samples)
            
            # convert the obs
            state, img, img2 = self.convert_obs(new_obs)
            
            # save the obs
            self.save_obs(state, img, img2)
            
            done = terminated or truncated
            
            total_reward += float(rewards)
            
            if done:
                break
            
        return samples, new_obs, total_reward, done, infos
    
    def eval_actions(self, actionss):
        # get the state
        obs_dict_np = self.evaluator.GetObs()
        
        # put on gpu and norm
        nobs = self.evaluator.norm_gpu_obs(obs_dict_np)
        
        # tile each entry
        nb = actionss.shape[0]
        def f(x):
            dims = [nb] + (len(x.shape)-1) * [1]
            return torch.tile(x, dims=dims)
        
        nobs = pytorch_util.dict_apply(nobs, f)
        
        # put actionss on gpu and norm
        nactionss = self.evaluator.norm_gpu_action(actionss)
        
        # squeeze out the history dim (not needed for the critic)
        na2 = torch.squeeze(nactionss, dim=1)

        with torch.no_grad():
            assert(self.critic is not None)
            
            # critic is expecting norm'd obs and actions of the form [batch, history, ...]
            self.critic.eval()
            qvals1, qvals2 = self.critic.forward(nobs, na2, self.critic_ops)
            self.critic.train()

        return qvals1
    
    def infer_action(self) -> tuple[torch.Tensor, float]:
        if self.use_critic_preferred_actions:
            # actionss = []
            qvals = []
            
            best_actions = None
            best_qval = -999.0

            # infer n times
            for i in range(self.critic_preferred_actions_nb):
                # this returns unnorm'd actions on cpu
                actions, all_actions = self.evaluator.infer()

                # actionss.append(all_actions)
                
                # batch eval
                # stack and add batch dim
                # actionss2 = torch.stack([all_actions])
                assert(all_actions is not None)
                actionss2 = torch.unsqueeze(all_actions, dim=0)
                qval = self.eval_actions(actionss2)
                
                qvalp = qval.squeeze()
                qvals.append(qvalp)
                
                assert(actionss2.shape[0] == qval.shape[0])
                
                # update best action
                if qvalp > best_qval:
                    best_qval = qvalp
                    best_actions = actions
                
                if self.use_critic_preferred_actions_early_exit:
                    # exit if qval is positive
                    if qval > 0.0:
                        break

            # # get the argmax of qvals
            # idx = torch.argmin(qvals)

            # # get the action
            # best_actions = actionss[idx]
            
            # logging
            globals.LOGGER.log_one("rollout/avg_qval", best_qval)


            assert(best_actions is not None)
            return best_actions, best_qval
        else:
            actions, all_actions = self.evaluator.infer()
            assert(actions is not None)
            return actions, 0.0
        
    def one_rollout(self):
        """
        Run one rollout
        """
        samples = []
        total_reward = 0.0
        best_qvals = []
        done = False
        while not done:
            # get the action trajectory
            actions, best_qval = self.infer_action() 
            actions = torch.squeeze(actions)
            actions = actions.numpy()
            best_qvals.append(best_qval)
            
            # none protection
            if actions is not None:
                # execute the full trajectory
                new_samples, new_obs, rewards, dones, infos = self.step_trajectory(actions)
                
                # append all new samples
                samples += new_samples
                
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
                    
                # logging
                total_reward += rewards
                
        # logging ... doesn't work with multiple rollouts because the STEP doesn't change ... see inside log_one
        globals.LOGGER.log_one("rollout/ep_reward", total_reward)
        globals.LOGGER.log_one("rollout/avg_best_qval", np.array(best_qvals).mean())
            
        return samples
    
    def save_samples(self, actions, rewards, samples):
        # save the sample using the old obs, current action, current reward
        assert(self.state is not None)
        assert(self.img is not None)
        assert(self.img2 is not None)
        
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