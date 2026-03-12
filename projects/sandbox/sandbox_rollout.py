



"""
for use with the DexNex Intelligence stack
"""

# print("""
#       Sandbox Rollout. Reminders: be sure to seed the rl replay buffer with the BC data-set
#       """)




import numpy as np
from diffusion_policy.rollout import (
    Rollout,
    RolloutEnv,
)

import gymnasium as gym

# import pytorch utils
from diffusion_policy.common import pytorch_util
from diffusion_policy import utils


import diffusion_policy.globals as globals

from diffusion_policy.evaluation.eval import SimpleInference

import avatar_drake_sim.sims.sandbox.sandbox_common as commons


from avatar_drake_sim.sims.sandbox.sandbox_gym import SandboxGymEnv

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecEnv

    
class SandboxRLRolloutEnv(RolloutEnv):
    def __init__(self,
                 sandbox_commons_options: dict | None = None,
                 drake_env_globals_config_path = None,
                 ):
        super().__init__()
        self.sandbox_commons_options = sandbox_commons_options
        self.drake_env_globals_config_path = drake_env_globals_config_path
        
        # load options
        if sandbox_commons_options is not None:
            commons.set_current_globals(sandbox_commons_options)
        
    def setup(self):
        
        import logging; logging.getLogger("drake").setLevel(logging.ERROR)
        
        # Register environment with gym globally
        # gym.envs.register(id="SandboxRLGymEnv-v0", entry_point="avatar_drake_sim.sims.sandbox.sandbox_gym:SandboxGymEnv") #type:ignore
            
        
        # self.env = gym.make("SandboxRLGymEnv-v0", globals_config_path=self.drake_env_globals_config_path) #type:ignore
        self.env: VecEnv = SandboxGymEnv(globals_config_path=self.drake_env_globals_config_path)
        
        return self.env
            
class SandboxRollout(Rollout):
    def convert_drake_obs(self, drake_obs):
        """
        convert from the drake obs dict to the obs dict expected by the policy
        """
        return drake_obs
    
    def save_data(self, obs):
        # convert
        self.obs = self.convert_drake_obs(obs)
        
        # save to evaluator
        assert(isinstance(self.evaluator, SimpleInference))
        self.evaluator.save_obs(obs)
    
    def convert_action(self, action_dict):
        """
        convert from ML to Drake
        """
        return action_dict
    
    def rollout_prep(self):
        # update the eval class
        
        # reset the sim class
        obs = self.env.reset()
                        
        # update the evaluator
        self.save_data(obs)
                
    def one_rollout(self):
        """
        Run one rollout
        """
        samples = []
        total_reward = np.zeros(commons.NB_PARALLEL_ENVS)
        best_qvals = []
        total_jerk = 0.0
        env_dones = np.zeros(commons.NB_PARALLEL_ENVS, dtype=bool)
        total_env_time = 0.0
        total_inference_time = 0.0
        
        done = False
        while not done:
            tic = utils.tic()
            # get the action trajectory
            actions, best_qval, failed = self.infer_action() 
            toc = utils.toc(tic)
            globals.log_one_if_exists("profiling/SandboxRollout.infer_action", toc)
            total_inference_time += toc
            
            if failed:
                print("No valid action. Episode failure.")
                done = True
                break
            
            # none protection
            if actions is not None:
                actions = self.prep_action_for_stepping(actions)
                
                best_qvals.append(best_qval)
                
                # execute the full trajectory
                new_samples, new_obs, rewards, dones, infos, jerk, env_time = self.step_trajectory(actions)
                
                # append all new samples
                samples += new_samples
                
                # update the evaluator
                self.save_data(
                    new_obs,
                )
                
                # logging
                not_done = ~env_dones
                total_reward[not_done] += rewards[not_done]
                total_jerk += jerk
                total_env_time += env_time
                
                # if dones:
                #     done = True
                env_dones = np.logical_or(env_dones, dones)
                done = np.all(env_dones)
                    
                
        # normalize jerk by episode length
        if len(samples) > 0:
            total_jerk /= len(samples)
            
        return samples, total_reward, best_qvals, total_jerk, total_env_time, total_inference_time

# immediate next steps:
    # collect larger mage hand demo dataset
        
    
    def step_trajectory(self, actions):
        total_reward = np.zeros(commons.NB_PARALLEL_ENVS)
        samples = []
        
        new_obs = None
        env_dones = np.zeros(commons.NB_PARALLEL_ENVS, dtype=bool)
        infos = None
        
        B = actions.shape[0]
        nb_envs = B
        H = actions.shape[1]
        
        
        total_jerk = 0.0
        total_compute_time = 0.0
        
        m = min(H, self.rollout_num_actions)
        for i in range(m):
            tic = utils.tic()
            action = actions[:, i, :]
            
            # convert action for Drake
            action = self.convert_action(action)
            
            # step the env
            new_obs, rewards, dones, infos = self.env.step(action) #type:ignore
            
            # save sample using the old obs. Required for RL
            self.save_samples(action, rewards, env_dones, samples)
            
            # must save the obs for the next save_samples
            self.save_data(new_obs)
                        
            # done = terminated or truncated
            env_dones = np.logical_or(env_dones, dones)
            done = np.all(env_dones)
            
            not_done = ~env_dones
            total_reward[not_done] += rewards[not_done]
            
            toc = utils.toc(tic)
            globals.log_one_if_exists("profiling/SandboxRollout.step_trajectory", toc)
            total_compute_time += toc
            if done:
                break
            
        return samples, new_obs, total_reward, env_dones, infos, total_jerk, total_compute_time
    
    def save_samples(self, actions, rewards, env_dones, samples):
        """
        each sample must contain all the obs keys, the action key, and 'reward'
        """
        # save the sample using the old obs, current action, current reward
        action_key = globals.CONFIG.action_key # type: ignore
        
        # the old obs
        obs = self.obs
        
        assert(obs is not None)
        
        data = {
            action_key: actions,
            'reward': np.float32(rewards),
            'dones': env_dones, # need dones for episode masking
        }
        
        data.update(obs)
        
        to_save = data
        # for key in globals.CONFIG.obs_keys_to_load:
        #     to_save[key] = data[key]
        
        # append the data
        samples.append(to_save)
        
    def save_envs_episode_to_rb(self, episode: list[dict], rb):
        # separate out the envs in the episode and save each one as its own episode in the rb
        
        # convert episodes to arrays
        env_episode_dicts = []
        
        for env_idx in range(commons.NB_PARALLEL_ENVS):
            env_episode_dict = {}
            
            # get the not done mask
            done_mask = np.array([step['dones'][env_idx] for step in episode])
            
            # if env never finished? is this a bug?
            if done_mask.sum() == 0:
                idx = len(done_mask) - 1
                
            # get the first index
            else:
                idx = np.where(done_mask)[0][0]
            
            # mask
            keep_mask = np.zeros_like(done_mask, dtype=bool)
            keep_mask[0:idx+1] = True
            
            
            for key in episode[0].keys():
                data_arr = np.array([step[key][env_idx] for step in episode])
                keep_data_arr = data_arr[keep_mask]
                env_episode_dict[key] = keep_data_arr
                
            env_episode_dicts.append(env_episode_dict)
            
        # confirm that reward has captured the episode termination correctly
        for episode in env_episode_dicts:
            rb.add_episode(episode, compressors='disk')