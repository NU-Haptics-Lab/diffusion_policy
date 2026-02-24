


"""
the BCExplorationAlgorithm is a wrapper around a policy algorithm, which provides additional modes of exploration.

This is to be used instead of an epsilon-greedy algorithm, for example.

In robotics, it's almost never a good idea to randomly explore (state-action space too large), so instead we can explore in other ways (B.C. policies, etc)


policy can be whatever, but it must have methods: inference

"""















import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy

from stable_baselines3.common.envs import SimpleMultiObsEnv # Example dict env

from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
#
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Type, Union

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, DictReplayBufferSamples
#
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

import pydrake.all

from avatar_drake_sim.sims.sandbox.sandbox_gym import SandboxGymEnv


from pydrake.multibody.plant import MultibodyPlant
from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff, ExtractGradient, ExtractValue
import torch as th

from torch import nn

import gymnasium as gym

from stable_baselines3.common.buffers import DictReplayBuffer

from avatar_drake_sim.sims.sandbox.sandbox_utils import DifferentiableFK, ReplayBufferWithNextAction

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    Simulator,
    Diagram,
    JacobianWrtVariable,
    StartMeshcat,
)
import pydrake.all
import pydrake.visualization

from drake_gym.drake_gym import DrakeGymEnv

from avatar_drake_sim.sims.sandbox.sandbox_common import (
    MaxParams,
    LEARNING_RATE_DT,
    CURRENT_PARAMS,
    LOADED_ML_RUNNERS,
)

from avatar_drake_sim.sims.sandbox.sandbox_robot import load_gofa_and_robot_forearm_and_palm

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
                 exploration_rate_metric = 0.0,
                 use_min_dist_explore = False, # the bc policy now does this
                 use_bc_explore = True,
                 ):
        super().__init__()
        self.policy = policy
        self.use_min_dist_explore = use_min_dist_explore
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
        self.exploration_random = SimpleExponentialScheduler(initial_value = 0.1, decay_rate=0.99995)
        self.exploration_BC = SimpleExponentialScheduler(initial_value = 4.0, decay_rate = 0.99995)
        
        if self.use_bc_explore:
            explorers = [self.exploration_random, self.exploration_policy, self.exploration_BC]
        else:
            explorers = [self.exploration_random, self.exploration_policy]
        
        self.exploration_chooser = MultipleSchedulersSampler(explorers)
        
        # my I/O noise schedulers -- inputs: max noise mag, max nb training steps
        self.noise_action_sch = SimpleSigmoidLowToHighScheduler(0.1, 100000) # noise in torque
        
        # params
        self.optimal_metric_link_name = "hand_palm_link"
        
        self.setup_gofa_drake_plant()
        
    def reset(self):
        self.policy.reset()
        
    def get_model(self):
        return self.policy.get_model()
    
    def step(self):
        self.policy.step()
        
        
    def loss(self, nbatch, rb_id=None):
        return self.policy.loss(nbatch)
        
    def setup_gofa_drake_plant(self):
        """
        make a dummy gofa plant with just the gofa arm. Use that plant for FK and jacobian calculations.
        gofa plant will always be valid no matter the hand
        """
        # make a plant with just the gofa arm
        self.plant = MultibodyPlant(time_step=0.01)
        parser = Parser(self.plant)
        
        gofa_model_instance, hand_model_instance, forearm_and_palm_str = load_gofa_and_robot_forearm_and_palm(self.plant, parser)
        
        self.plant.Finalize()
        
    def reset_callback(self, env: DrakeGymEnv):
        simulator = env.simulator
        
        assert(simulator is not None)
        
        context = simulator.get_mutable_context()
        self.episode_plant: MultibodyPlant = simulator.get_system().GetSubsystemByName("plant") #type:ignore
        
        # plant_context = plant.GetMyMutableContextFromRoot(context)
        
        # print(f"Post-Step: {plant.GetPositions(plant_context)}")
        pass
        # 
        
        # # get the builder?
        
        # self.meshcat.Delete()
        
        # pydrake.visualization.ApplyVisualizationConfig(vis_config, builder, scene_graph = scene_graph, meshcat = self.meshcat)   
        
    # def setup_differentiable_fk(self, env: DrakeGymEnv):
    #     """
    #     this must be done after the env reset because otherwise "simulator" won't exist. And we must use the same simulator as the episode so that the FK is correct?
        
    #     """
    #     simulator = env.simulator
    #     assert(simulator is not None)
        
    #     # TODO: different differentiable_fk for each vectorized env
    #     diagram: Diagram = simulator.get_system() # type:ignore
        
    #     plant: MultibodyPlant = diagram.GetSubsystemByName("plant") #type:ignore
    #     assert(plant is not None)
        
    #     # save the plant
    #     self.plant = plant
        
    #     frame_name = "hand_root_link"
        
    #     self.differentiable_fk = DifferentiableFK(
    #         frame_name = frame_name,
    #         plant = plant
    #     )
        
    #     pass
        
    def optimal_policy_action(self, observation):
        """
        input and output are CPU
        """
        # move each observation to torch
        obs_dict = {key: th.as_tensor(observation[key]).to(self.device).float() for key in observation}
        
        # do the inference
        final_action_tensor = self.policy.inference(obs_dict)
        
        assert(final_action_tensor.shape[0] == 1)
        assert(final_action_tensor.shape[1] == NUM_ACTIONS) # action dim
        
        # to cpu, to numpy, squeeze
        final_action_np = final_action_tensor.cpu().numpy().flatten()
        
        # cpu
        return final_action_np
    
    def get_dist_loss(self, position, target_position):
        return F.mse_loss(position, target_position)
    
    # def get_differentiable_dist_mse(self, observation):
    #     # 1. Prepare inputs
    #     q = observation['robot_joint_pos'] # Ensure this is a tensor with requires_grad=True
    #     v = observation['robot_joint_vel']
    #     q_tensor = th.tensor(q, device=self.device, dtype=th.float32, requires_grad=True)
        
    #     # convert to torch
    #     target_pos = th.as_tensor(observation['object_pos'], device=self.device)

    #     # 2. Call the Drake-backed FK
    #     current_velocities = self.differentiable_fd.forward(tau_th, q, v)
        
    #     # 3. Compute MSE
    #     metric = th.nn.functional.mse_loss(current_pos, target_pos)
        
    #     return q_tensor, metric
    
    def get_linear_jacobian(self, q, frame_name):
        # make a temp context
        context = self.plant.CreateDefaultContext()
        
        # fill it with our q's
        self.plant.SetPositions(context, q)
        
        # 1. Get the frame and world frame
        frame = self.plant.GetFrameByName(frame_name)
        world_frame = self.plant.world_frame()
        
        # 2. Compute the Jacobian
        # This calculates J such that V_WF = J * v
        # where v is the vector of generalized velocities
        J = self.plant.CalcJacobianTranslationalVelocity(
            context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=frame,
            p_BoBi_B=np.array([0, 0, 0]),       # Point on frame B (usually origin)
            frame_A=world_frame,  # Expressed in World
            frame_E=world_frame   # Measured relative to World
        )
        return J
    
    def forward_kinematics(self, q, frame_name):
        # make a temp context
        context = self.plant.CreateDefaultContext()
        
        # fill it with our q's
        self.plant.SetPositions(context, q)
        
        # 1. Get the frame and world frame
        frame = self.plant.GetFrameByName(frame_name)
        
        # 2. Compute the FK
        X_WF = self.plant.CalcRelativeTransform(context, self.plant.world_frame(), frame)
        
        # extract just the position
        position = X_WF.translation()
        
        return position
    
    def get_this_plants_qs(self, observation):
        """
        TODO: confirm this is correct. IDK the observation order but I think this will be correct
        """
        q = observation['robot_joint_pos'][0].squeeze()
        
        nb_q_plant = self.plant.num_positions()
        
        q = q[:nb_q_plant]
        return q
    
    def minimize_distance_action(self, observation: dict):
        """
        decreases distance between hand and object
        
        TODO:
        make a gofa plant with just the gofa arm. Use that plant for FK and jacobian calculations.
        gofa plant will always be valid no matter the hand
        """
        # current pos from F.K.
        plant_q = self.get_this_plants_qs(observation)
        
        current_pos = self.forward_kinematics(plant_q, frame_name=self.optimal_metric_link_name)
        
        # desired pos is the object pos
        target_pos = observation['object_pos'][0, 0] # batch 0, object 0
        
        target_vector = target_pos - current_pos
        
        # normalize
        target_vector = target_vector / (np.linalg.norm(target_vector) + 1e-8)
        
        # "convert" to vel
        desired_vel = target_vector / LEARNING_RATE_DT
        
        # "convert" to accel
        desired_accel = desired_vel / LEARNING_RATE_DT
        
        # "convert" to force
        desired_force = desired_accel # mass is "1"
        
        # assemble the wrench ... should be [M, N]
        wrench = np.zeros(6)
        wrench[3:] = desired_force
        
        # get the jacobian at the current configuration
        J = self.get_linear_jacobian(plant_q, frame_name=self.optimal_metric_link_name)
        
        # modern robotics eq 5.26
        tau = J.T @ desired_force
        
        # normalize so the max value is one
        tau = tau / tau.max()
        
        # scale taus so we move faster
        tau *= 5.0
        
        action = np.zeros(NUM_ACTIONS)
        action[:tau.shape[0]] = tau

        assert(action.shape[0] == NUM_ACTIONS)
        return action
    
    def gravity_comp_action(self, observation):
        """
        compute the gravity compensation torques and return those as an action, do it for the episode plant so it takes into account the hand
        """
        # current q from observation
        ep_joint_q = observation['robot_joint_pos'][0].squeeze()
        
        actuator_qs = ep_joint_q[:self.episode_plant.num_actuated_dofs()]
        
        qs = np.zeros(self.episode_plant.num_positions())
        qs[:actuator_qs.shape[0]] = actuator_qs
        
        idx = actuator_qs.shape[0]
        
        # must fill in block q so that a quat isn't all zeros
        qs[idx+3:idx+7] = np.array([1, 0, 0, 0]) # default quat, doesn't matter for gravity comp but can't be all zeros
        
        # make a temp context
        context = self.episode_plant.CreateDefaultContext()
        
        # fill it with our q's
        self.episode_plant.SetPositions(context, qs)
        
        # compute gravity compensation torques
        tau_gravity_comp = self.episode_plant.CalcGravityGeneralizedForces(context)
        
        action = np.zeros(NUM_ACTIONS)
        action[:tau_gravity_comp.shape[0]] = tau_gravity_comp
        
        assert(action.shape[0] == NUM_ACTIONS)
        
        return action
    
    def optimal_metric_action(self, observation):
        # minimize distance between hand and object
        action = self.minimize_distance_action(observation)
        
        # add on gravity comp
        # action += self.gravity_comp_action(observation)
        
        return action
    
    def get_ee_pos(self, observation):
        ee_pos = self.forward_kinematics(self.get_this_plants_qs(observation), frame_name=self.optimal_metric_link_name)
        return ee_pos
    
    def get_manipulability(self, observation):
        """
        get the manipulability value
        modern robotics page 199
        """
        plant_q = self.get_this_plants_qs(observation)
        J = self.get_linear_jacobian(plant_q, frame_name=self.optimal_metric_link_name)
        
        # manipulability is sqrt(det(J * J^T))
        JJt = J @ J.T
        manipulability = np.sqrt(np.linalg.det(JJt))
        
        return manipulability
    
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
        
        torque_action = np.zeros(NUM_ACTIONS)
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
        out_action = np.zeros(NUM_ACTIONS)
        
        length = min(action.shape[0], NUM_ACTIONS)
        
        out_action[:length] = action[:length]
        
        return out_action
            
    # alias
    def explore(self, *args, **kwargs):
        self.predict(*args, **kwargs)
    # predict is called by SB3
    def predict( 
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        assert(isinstance(observation, dict))
        
        if deterministic:
            action, state = self.policy.inference(observation, state, episode_start, deterministic)
            return action, state
        
        # do FK to get the current position of the gofa ee
        gofa_ee = self.get_ee_pos(observation)
        
        # compute the distance to the object
        dist_to_object = np.linalg.norm(gofa_ee - observation['object_pos'][0, 0])
        
        # whether we're in a low info region (aka no objects nearby)
        in_low_info_region = dist_to_object > 0.75
        
        ####
        # PROTECT AGAINST SINGULARITIES. it will crash the sim.
        if self.use_min_dist_explore:
            manipulability = self.get_manipulability(observation)
            
            min_manipulability_threshold = 0.015
            
            high_manipulability = manipulability > min_manipulability_threshold
            
            take_optimal_metric_action = high_manipulability and in_low_info_region
        else:
            take_optimal_metric_action = False
        ####
        
        if take_optimal_metric_action:
            # force a dist reduction action aka optimal_metric_action
            choice = "metric"
        else:
            # draw n, but only between 0 and random + policy likelihood (no dist reduction action)
            choice_idx = self.exploration_chooser.sample(self.num_timesteps)
            
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
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
                
                
        # optimal action w.r.t. the policy
        elif choice == "policy":
            action, state = self.optimal_policy_action(observation), None
            
            # add env dim, required for "is_vectorized_observation"
            action = np.expand_dims(action, axis=0)
            
        # optimal action w.r.t. some metric
        elif choice == "metric":
            # get the action
            action, state = self.optimal_metric_action(observation), None
            
            # add env dim, required for "is_vectorized_observation"
            action = np.expand_dims(action, axis=0)
            
        elif choice == "BC":
            action, state = self.get_diffusion_bc_action(observation), None
            
            # add env dim, required for "is_vectorized_observation"
            action = np.expand_dims(action, axis=0)
            
        else:
            raise ValueError("Invalid exploration choice")
            
        assert(action.shape[0] == 1)
        assert(action.shape[1] == NUM_ACTIONS) # action dim
        
        return action, state