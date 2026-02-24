


"""
the BCExplorationAlgorithm is a wrapper around a policy algorithm, which provides additional modes of exploration.

This is to be used instead of an epsilon-greedy algorithm, for example.

In robotics, it's almost never a good idea to randomly explore (state-action space too large), so instead we can explore in other ways (B.C. policies, etc)


policy can be whatever, but it must have methods: inference

"""



import os

import torch as th
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np

import torch as th

from torch import nn


from avatar_drake_sim.sims.sandbox.sandbox_common import (
    MaxParams,
    LEARNING_RATE_DT,
    CURRENT_PARAMS,
    LOADED_ML_RUNNERS,
)


from avatar_drake_sim.sims.sandbox.classes.simple_scheduling import (
    SimpleExponentialScheduler,
    SimpleLogarithmicScheduler,
    MultipleSchedulersSampler,
    SimpleConstantScheduler,
    SimpleSigmoidHighToLowScheduler,
    SimpleSigmoidLowToHighScheduler,
)

from avatar_drake_sim.sims.sandbox.classes.run_diffusion_policy import (
    DiffusionPolicyRunner,
    SandboxRobotBCPolicyRunner,
    SandboxRobotImplicitBCPolicyRunner,
)


from diffusers.schedulers.scheduling_ddim import (
    DDIMScheduler
)

import diffusion_policy.globals as globals

# the max allowable nb joints
NUM_ACTIONS = MaxParams().max_nb_joint_pos

class BCExplorationAlgorithm(nn.Module):
    """
    No actor, just critic.
    
    Also no entropy loss, I want to do a custom exploration strategy
    """
    def __init__(self, 
                 policy,
                 exploration_rate_random = 0.4,
                 exploration_rate_policy = 0.4,
                 use_bc_explore = True,
                 ):
        super().__init__()
        self.policy = policy
        self.use_bc_explore = use_bc_explore
        
        # backwards compat:
        self.noise_scheduler = DDIMScheduler()
        
        # my B.C. policy
        # self.diffusion_bc_runner = DiffusionPolicyRunner()
        # self.diffusion_bc_runner = SandboxRobotBCPolicyRunner()
        if self.use_bc_explore:
            self.diffusion_bc_runner = SandboxRobotImplicitBCPolicyRunner()
        
            # save in our loaded ML runners
            LOADED_ML_RUNNERS['diffusion_policy_runner'] = self.diffusion_bc_runner
        
        # my exploration schedulers
        self.exploration_policy = SimpleConstantScheduler(value = 1.0)
        self.exploration_random = SimpleExponentialScheduler(initial_value = 0.05, decay_rate=0.99995)
        self.exploration_BC = SimpleExponentialScheduler(initial_value = 4.0, decay_rate = 0.99995)
        
        if self.use_bc_explore:
            explorers = [self.exploration_random, self.exploration_policy, self.exploration_BC]
        else:
            explorers = [self.exploration_random, self.exploration_policy]
        
        self.exploration_chooser = MultipleSchedulersSampler(explorers)
        
        # my I/O noise schedulers -- inputs: max noise mag, max nb training steps
        self.noise_action_sch = SimpleSigmoidLowToHighScheduler(0.1, 100000) # noise in torque
        
        
    def reset(self):
        self.policy.reset()
        
    def get_model(self):
        return self.policy.get_model()
    
    def step(self):
        self.policy.step()
        
        
    def loss(self, nbatch, rb_id=None):
        """
        used during training
        """
        return self.policy.loss(nbatch)
    
    def get_future_actions(self, all_actions):
        start = np.argmax(np.array(globals.CONFIG.action_rel_indices) >= 0)
        
        future_actions = all_actions[:, start:]
        return future_actions
        
    def optimal_policy_action(self, observation):
        """
        input and output are CPU
        """
        # move each observation to torch
        # obs_dict = {key: th.as_tensor(observation[key]).to(globals.CONFIG.device).float() for key in observation}
        obs_dict = observation
        
        # do the inference
        final_action_tensor = self.policy.inference(obs_dict)
        
        future_action_tensor = self.get_future_actions(final_action_tensor)
        
        return future_action_tensor, final_action_tensor
    
    def get_torque_from_position_action(self, position_action, observation):
        """
        convert a position action to a torque action using a PD controller
        """
        # current q from observation
        ep_joint_q = observation['robot_joint_pos'][0].squeeze()
        
        # simple P controller, no D term for now
        kp_heavy = 20.0
        kp_else = 0.1
        
        position_action = position_action[:ep_joint_q.shape[0]]
        
        # take the shorter of the two
        length = min(position_action.shape[0], ep_joint_q.shape[0])
        position_action = position_action[:length]
        ep_joint_q = ep_joint_q[:length]
        
        error = position_action - ep_joint_q
        
        torque_action = th.zeros(NUM_ACTIONS)
        torque_action[:5] = kp_heavy * error[:5] # heavy joints get higher gains
        torque_action[5:length] = kp_else * error[5:length] # other joints
        
        return torque_action
    
    def get_diffusion_bc_action(self, observation):
        """
        get the action from the diffusion BC policy
        """
        # state = observation['robot_joint_pos']
        
        # actions_joint_positions = self.diffusion_bc_runner.infer_from_robot_state(state)
        actions_joint_positions = self.diffusion_bc_runner.infer(observation)
        
        # take the last action
        action_joint_positions = actions_joint_positions[-1]
        
        #convert to torque
        action = self.get_torque_from_position_action(action_joint_positions, observation)
        
        # prep for output
        out_action = th.zeros(NUM_ACTIONS)
        
        length = min(action.shape[0], NUM_ACTIONS)
        
        out_action[:length] = action[:length]
        
        final_action_tensor = out_action
        
        future_action_tensor = self.get_future_actions(final_action_tensor)
        
        return future_action_tensor, final_action_tensor
            
    # alias
    def explore(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    # predict is called by SB3
    def predict(self,observation):
        """
        Overrides the base_class predict function to include more exploration modes.
        """
        assert(isinstance(observation, dict))
        
        batch_size = next(iter(observation.values())).shape[0]
        horizon = len(globals.CONFIG.action_rel_indices) # type: ignore
        
        # draw n, but only between 0 and random + policy likelihood (no dist reduction action)
        choice_idx = self.exploration_chooser.sample(globals.STEP)
        
        if choice_idx == 0:
            choice = "random"
        elif choice_idx == 1:
            choice = "policy"
        elif choice_idx == 2:
            choice = "BC"
        else:
            raise ValueError("Invalid choice index from exploration chooser")
            
        
        # random action
        if choice == "random":
            action = th.rand(size=(batch_size, horizon, NUM_ACTIONS), device=globals.CONFIG.device)
            
            future_actions = self.get_future_actions(action)
                
        # optimal action w.r.t. the policy
        elif choice == "policy":
            future_actions, action = self.optimal_policy_action(observation)
            
            # # add env dim, required for "is_vectorized_observation"
            # action = np.expand_dims(action, axis=0)
            
        elif choice == "BC":
            future_actions, action = self.get_diffusion_bc_action(observation)
            
            # # add env dim, required for "is_vectorized_observation"
            # action = np.expand_dims(action, axis=0)
            
        else:
            raise ValueError("Invalid exploration choice")
            
        assert(action.shape[0] == batch_size)
        assert(action.shape[1] == horizon)
        assert(action.shape[2] == NUM_ACTIONS) # action dim
        
        return future_actions, action
    
    def infer(self, nobs):
        """
        used during rollouts
        """
        return self.explore(nobs)
        