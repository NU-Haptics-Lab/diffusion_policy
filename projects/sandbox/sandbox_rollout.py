



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

    
class SandboxRLRolloutEnv(RolloutEnv):
    def setup(self):
        
        import logging; logging.getLogger("drake").setLevel(logging.ERROR)
        
        # Register environment with gym globally
        gym.envs.register(id="SandboxRLGymEnv-v0", entry_point="avatar_drake_sim.sims.sandbox.sandbox_gym:SandboxGymEnv") #type:ignore
            
        
        self.env = gym.make("SandboxRLGymEnv-v0")
        
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
        obs, info = self.env.reset()
                        
        # update the evaluator
        self.save_data(obs)
                
    def one_rollout(self):
        """
        Run one rollout
        """
        samples = []
        total_reward = 0.0
        best_qvals = []
        total_jerk = 0.0
        
        done = False
        while not done:
            # get the action trajectory
            actions, best_qval, failed = self.infer_action() 
            
            if failed:
                print("No valid action. Episode failure.")
                done = True
                break
            
            # none protection
            if actions is not None:
                actions = self.prep_action_for_stepping(actions)
                
                best_qvals.append(best_qval)
                
                # execute the full trajectory
                new_samples, new_obs, rewards, dones, infos, jerk = self.step_trajectory(actions)
                
                # append all new samples
                samples += new_samples
                
                # update the evaluator
                self.save_data(
                    new_obs,
                )
                
                if dones:
                    done = True
                    
                # logging
                total_reward += rewards
                total_jerk += jerk
                
        # normalize jerk by episode length
        if len(samples) > 0:
            total_jerk /= len(samples)
            
        return samples, total_reward, best_qvals, total_jerk

# immediate next steps:
    # collect larger mage hand demo dataset
        
    
    def step_trajectory(self, actions):
        total_reward = 0.0
        samples = []
        
        new_obs = None
        done = False
        infos = None
        
        
        total_jerk = 0.0
        
        m = min(len(actions), self.rollout_num_actions)
        for i in range(m):
            action = actions[i]
            
            # convert action for Drake
            action = self.convert_action(action)
            
            # step the env
            new_obs, rewards, terminated, truncated, infos = self.env.step(action) #type:ignore
            
            # save sample using the old obs. Required for RL
            self.save_samples(action, rewards, samples)
            
            # must save the obs for the next save_samples
            self.save_data(new_obs)
                        
            done = terminated or truncated
            
            total_reward += float(rewards)
            
            if done:
                break
            
        return samples, new_obs, total_reward, done, infos, total_jerk
    
    def save_samples(self, actions, rewards, samples):
        """
        each sample must contain all the obs keys, the action key, and 'reward'
        """
        # save the sample using the old obs, current action, current reward
        action_key = globals.CONFIG.action_key # type: ignore
        
        obs = self.obs
        
        assert(obs is not None)
        
        data = {
            action_key: actions,
            'reward': np.float32(rewards),
        }
        
        data.update(obs)
        
        to_save = data
        # for key in globals.CONFIG.obs_keys_to_load:
        #     to_save[key] = data[key]
        
        # append the data
        samples.append(to_save)