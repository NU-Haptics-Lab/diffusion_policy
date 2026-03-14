

"""
implicit model means that the model takes (s, a) as its input and produces a single value. During inference, we can produce actions by doing gradient ascent on the input action to maximize the output. 

These classes, however, are not responsible for defining the loss, nor what the output means. That should be done by a parent class.
"""

import numpy as np

import torch
import torch as th
import torch.nn.functional as F

# reference:
from diffusion_policy.model.diffusion_model import (
    DiffusionModel
)

from diffusion_policy.model.obs_encoder import (
    ObsEncoderMaker
)

from diffusion_policy.model.conditional_slashnet1d import (
    ConditionalSlashnet1D,
)

from diffusion_policy.utils import (
    print_nb_params,
)

from diffusion_policy.common.pytorch_util import (
    dict_apply,
)

from torch import (
    nn
)

from diffusion_policy.policy.base_image_policy import (
    BaseImagePolicy
)

from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler
)

from diffusers.schedulers.scheduling_ddim import (
    DDIMScheduler,
)

import diffusion_policy.globals as globals

import torch._functorch.apis as functorch_apis

from tqdm import (
    tqdm
)

from line_profiler import profile

import avatar_drake_sim.sims.sandbox.sandbox_common as commons

from avatar_drake_sim.sims.sandbox.classes.simple_scheduling import (
    SimpleLinearScheduler
)

from dexnex_intelligence.lbfgs import (
    LBFGSProblem,
)

class LBFGSOptions:
    def __init__(self,
                 use_l2_reg = True,
                 use_jerk_penalty = True,
                 l2_reg_weight = 1.0,
                 jerk_weight = 1.0,
                 divide_by_nb_of_penalties = True,
                 use_add_noise = False,
                 noise_mag = 0.01, # remember, it's normalized
                 use_policy = True, # set False for debugging
                 policy_weight = 1.0,
                 ):
        self.use_l2_reg = use_l2_reg
        self.use_jerk_penalty = use_jerk_penalty
        self.l2_reg_weight = l2_reg_weight
        self.jerk_weight = jerk_weight
        self.divide_by_nb_of_penalties = divide_by_nb_of_penalties
        self.use_add_noise = use_add_noise
        self.noise_mag = noise_mag
        self.use_policy = use_policy
        self.policy_weight = policy_weight
        
        # set up schedules
        self.noise_mag_scheduler = SimpleLinearScheduler(0.0, self.noise_mag, 1.0, 0.0)
        
    def get_nb_penalties(self):
        return self.use_jerk_penalty + self.use_l2_reg
    
    def get_noise_mag(self):
        val = self.noise_mag_scheduler.get_value(commons.DIFFICULTY_SCALE)
        return val * self.use_add_noise
        

class ImplicitPolicy(nn.Module):
    def __init__(self,
            action_shape: list,
            obs_encoder_maker: ObsEncoderMaker,
            n_obs_steps, # input time-length
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            num_mid_module_repeats = 4,
            num_cond_encoder_layers = 2,
            num_cond_encoder_hidden_features = 128,
            ):
        super().__init__()
        self.action_shape = action_shape
        self.obs_encoder_maker = obs_encoder_maker
        self.n_obs_steps = n_obs_steps
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        self.num_mid_module_repeats = num_mid_module_repeats
        self.num_cond_encoder_layers = num_cond_encoder_layers
        self.num_cond_encoder_hidden_features = num_cond_encoder_hidden_features

    def setup(self):
        # setup all my children
        self.obs_encoder_maker.setup()
        
        # my members
        self.obs_encoder = self.obs_encoder_maker.get()
        assert(self.obs_encoder is not None)

        # parse shape_meta
        assert len(self.action_shape) == 1
        self.action_dim = self.action_shape[0]
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * self.n_obs_steps

        self.model = ConditionalSlashnet1D(
                        input_dim=self.action_dim,
                        global_cond_dim=global_cond_dim,
                        down_dims=self.down_dims,
                        kernel_size=self.kernel_size,
                        n_groups=self.n_groups,
                        cond_predict_scale=self.cond_predict_scale,
                        num_mid_module_repeats=self.num_mid_module_repeats,
                        num_cond_encoder_layers=self.num_cond_encoder_layers,
                        num_cond_encoder_hidden_features=self.num_cond_encoder_hidden_features,
                    )
        
        

        print_nb_params(self.model, "Dense params")
        print_nb_params(self.obs_encoder, "Vision params")

    def forward_obs_encoder(self, *args, **kwargs):
        return self.obs_encoder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def copy(self):
        other = ImplicitPolicy(
            self.action_shape,
            self.obs_encoder_maker.copy(),
            n_obs_steps = self.n_obs_steps,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_predict_scale=self.cond_predict_scale,
            num_mid_module_repeats=self.num_mid_module_repeats,
        )
        other.setup()
        
        return other

class ImplicitAlgorithm(BaseImagePolicy):
    def __init__(self,
            policy: ImplicitPolicy,
            use_compiled_policy = False,
            use_only_for_inference = False,
            use_lbfgs_for_inference = True,
            use_batch_gd_for_inference = False,
            use_add_inference_noise = False,
            inference_step_size = 1e-2,
            inference_iterations = 20,
            inference_noise = 0.01,
            lbfgs_options = LBFGSOptions(),
    ):
        super().__init__()
        
        self.use_compiled_policy = use_compiled_policy
        self.use_only_for_inference = use_only_for_inference
        self.use_lbfgs_for_inference = use_lbfgs_for_inference
        self.use_batch_gd_for_inference = use_batch_gd_for_inference
        self.use_add_inference_noise = use_add_inference_noise
        self.inference_step_size = inference_step_size
        self.inference_iterations = inference_iterations
        self.inference_noise = inference_noise
        self.lbfgs_options = lbfgs_options

        self.policy = policy
        
        # my members
        self.compiled_policy = None
        self.compiled_vmap_grad_batch_traj_fcn = None
        
        
    def setup(self):
        self.policy.setup()
        
        self.noise_scheduler = DDPMScheduler()

        self.vmap_grad_batch_traj_fcn = self.make_vmap_grad_batch_traj()

        # save from global config
        self.action_rel_indices = globals.CONFIG.action_rel_indices #type:ignore
        self.horizon = len(self.action_rel_indices)
        
        T = self.horizon
        Da = self.policy.action_dim
        device = globals.CONFIG.device # type: ignore
        B = globals.CONFIG.batch_size # type: ignore
        
        # Initialize these once in your class __init__
        # if we're still training this policy, we MUST compile with gradient required. Use default mode for interleaving training and updated policy rollouts
        if self.use_compiled_policy:
            print("Compiling Implicit BC policy...")
            if self.use_only_for_inference:
                # if only for inference, we can compile with grad disabled for faster inference speed
                self.policy.requires_grad_(False)
                
                self.compiled_policy = torch.compile(self.policy, mode="reduce-overhead")
                
                if self.use_batch_gd_for_inference:
                    self.compiled_vmap_grad_batch_traj_fcn = torch.compile(self.vmap_grad_batch_traj_fcn, mode="reduce-overhead")
            else:
                self.compiled_policy = torch.compile(self.policy, mode="default")
                
                if self.use_batch_gd_for_inference:
                    self.compiled_vmap_grad_batch_traj_fcn = torch.compile(self.vmap_grad_batch_traj_fcn, mode="default")

    # alias
    def predict_action(self, nobs): #type:ignore
        return self.inference(nobs)
    
    # alias
    def infer(self, nobs_dict: dict, action = None, task_id=None, noise_scheduler=None):
        return self.inference(nobs_dict)
    
    def inference_lbfgs(self, nobs: dict, warm_start_trajectory = None):
        problem = LBFGSProblem(self.policy, self.compiled_policy, self.horizon, self.device, self.dtype, self.action_rel_indices, self.lbfgs_options)

        best_trajs = problem.solve(nobs, warm_start_trajectory)

        return best_trajs
    
    # @torch.no_grad() need grads for lbfgs
    def inference_lbfgs_old(self, nobs: dict, warm_start_trajectory = None):
        """
        use torch L-BFGS
        """
        self.policy.eval()
        To = 1
        value = next(iter(nobs.values()))
        B = value.shape[0] # batch
        nb_envs = B
        T = self.horizon # trajectory length
        Da = self.policy.action_dim # action dimension
        device = self.device
        dtype = self.dtype
        timestep = 0.0
        total_nb_proposed_trajectories = warm_start_trajectory.shape[0] if warm_start_trajectory is not None else 0
        nb_proposed_trajectories_per_env = int(total_nb_proposed_trajectories / commons.NB_PARALLEL_ENVS) if total_nb_proposed_trajectories > 0 else 0
        
        # previous action
        naction_prev = nobs['robot_joint_prev_action']

        # reshape obs: B, T, ... to B*T ...
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        # get encoded obs, no grad to save time
        with torch.no_grad():
            nobs_features = self.policy.forward_obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
        # dummy trajectory
        if warm_start_trajectory is None:
            raise NotImplementedError("unsupported with vecenv's")
            dummy_trajectory = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

            # randomly initialize a traj
            trajectory = self.make_noise(dummy_trajectory)
        else:
            trajectory = warm_start_trajectory.clone().detach().to(device).to(dtype)

        # for debugging
        # original_trajectory = trajectory.clone().detach()
        
            
        ###################
        # do necessary tiling in case nb_envs > 1. Tile the global_cond instead of nobs because it'll be much smaller (and an array instead of a dict)
        if nb_envs > 1:
            # tile the trajectories
            # trajectory = th.tile(trajectory, (nb_envs, 1, 1))
            
            # repeat each global_cond nb_proposed_trajectories times, so that each proposed trajectory gets the same global_cond
            global_cond = th.repeat_interleave(global_cond, repeats=nb_proposed_trajectories_per_env, dim=0)
            
            # must also repeat_interleave the prev action
            naction_prev = th.repeat_interleave(naction_prev, repeats=nb_proposed_trajectories_per_env, dim=0)
            
            # assert the shapes are correct
            assert(trajectory.shape[0] == global_cond.shape[0] == naction_prev.shape[0])
        ###################
        
        # trajectory = torch.zeros((B, T, Da), device=device, requires_grad=True)
        trajectory.requires_grad_(True)
        optimizer = torch.optim.LBFGS([trajectory], 
                                        lr=1.0, # designed to work with unit step size
                                        max_iter=7, 
                                        history_size=4,
                                        # tolerance_grad=1e-6, 
                                        # tolerance_change=1e-7,
                                        line_search_fn="strong_wolfe",
                                        # line_search_fn=None, # faster than strong wolfe, but uses the fixed step size from lr
                    )
        
        if self.compiled_policy is not None:
            policy = self.compiled_policy
        else:
            policy = self.policy
        
        # initial scores for debugging
        # initial_scores = policy(trajectory, timestep, global_cond=global_cond).squeeze()
        
        # get past action indices for boundary condition enforcement
        past_actions = np.where(np.array(self.action_rel_indices) < 0)[0]
        
        # enforce initial actions
        if past_actions.shape[0] > 0:
            with torch.no_grad():
                # set the previous action
                trajectory[:, past_actions, :] = naction_prev
                
        original_trajectory = trajectory.clone().detach()
                
        # set stuff up for fast closure
        jerk_weight = self.lbfgs_options.jerk_weight * self.lbfgs_options.use_jerk_penalty
        policy_weight = self.lbfgs_options.policy_weight * self.lbfgs_options.use_policy
        l2_weight = self.lbfgs_options.l2_reg_weight * self.lbfgs_options.use_l2_reg
        noise_weight = self.lbfgs_options.get_noise_mag()
        divisor = self.lbfgs_options.get_nb_penalties() if self.lbfgs_options.divide_by_nb_of_penalties else 1.0
        
        # make the past action mask
        mask = torch.ones_like(trajectory)
        mask[:, past_actions, :] = 0.0
        # Pre-aligned fixed actions
        fixed_contribution = naction_prev * (1.0 - mask)
        
        # move to device
        jerk_weight = torch.tensor(jerk_weight, device=device, dtype=dtype)
        policy_weight = torch.tensor(policy_weight, device=device, dtype=dtype)
        l2_weight = torch.tensor(l2_weight, device=device, dtype=dtype)
        noise_weight = torch.tensor(noise_weight, device=device, dtype=dtype)
        divisor = torch.tensor(divisor, device=device, dtype=dtype)
        
        # losses = []
        def get_per_sample_loss():
            # nonlocals
            nonlocal trajectory
            
            optimizer.zero_grad()
            
            # setup
            with torch.no_grad():
                # noise? starting to look more similar to the core DDIM function
                if False:
                    trajectory += noise_weight * self.make_noise(trajectory)
                else:
                    trajectory += noise_weight * self.make_per_joint_noise(trajectory)
                    
                # enforce past actions
                trajectory.data.copy_(trajectory.data * mask + fixed_contribution)
            
            # run the policy
            qval = policy(trajectory, timestep, global_cond=global_cond)
            qval = qval.squeeze()
            
            qval = policy_weight * qval
            
            # add on a L2 regularization term to prevent massive actions
            # mean it over the trajectory dimension and the action dimension BUT NOT THE BATCH DIMENSION
            traj_norm_sq = trajectory.square().mean(dim=(1, 2))
                
            # jerk penalty
            # ... issue: nactions are normalized? Maybe doesn't matter...
            a_diff = th.diff(trajectory, dim=1) # shape (B, T-1, Da)
            jerk = a_diff / commons.LEARNING_RATE_DT
            
            # mean it over the trajectory dimension and the action dimension BUT NOT THE BATCH DIMENSION
            jerk_penalty = jerk.square().mean(dim=(1,2))
            
            # sum the penalties
            penalties = l2_weight * traj_norm_sq + jerk_weight * jerk_penalty
            
            # divide them
            penalties = penalties / divisor
                
            loss = -qval + penalties
            return loss
            
        def closure():
            per_sample_loss = get_per_sample_loss()
            
            loss = per_sample_loss.mean()
            loss.backward()
            
            # logging
            # losses.append(loss.item())
            return loss
        
        # initial losses
        with torch.no_grad():
            initial_losses = get_per_sample_loss()
                 
        # speed up .backward() by not calculating gradients for the policy parameters
        self.policy.requires_grad_(False)
        # just to make sure
        with torch.enable_grad():
            # only need to call step once    
            optimizer.step(closure) #type:ignore
        # restore grad for policy parameters
        self.policy.requires_grad_(True)

        self.policy.train()
        

        # 5. Find the highest scoring trajectory
        # if trajectory.shape[0] > 0:
        best_trajs = []
        with torch.no_grad():
            # final action enforcement (also should be done in get_per_sample_loss)
            trajectory[:, past_actions, :] = naction_prev
            final_losses = get_per_sample_loss()
            
            # reshape for easy indexing
            trajectory_envs = trajectory.view(nb_envs, nb_proposed_trajectories_per_env, T, Da)
            final_losses_envs = final_losses.view(nb_envs, nb_proposed_trajectories_per_env)
            
            # split by env
            for i in range(nb_envs):
                    if False:
                        final_losses_env = final_losses_envs[i]
                        trajectory_env = trajectory_envs[i]
                    
                        # min the loss
                        best_idx = torch.argmin(final_losses_env)
                        
                        # index, output is 2d: (T, Da)
                        best_traj = trajectory_env[best_idx].detach().clone()
                        
                        best_trajs.append(best_traj)
                        
                    # random traj
                    elif False:
                        trajectory_env = trajectory_envs[i]
                        
                        # random index
                        random_idx = torch.randint(0, nb_proposed_trajectories_per_env, (1,), device=trajectory.device)
                        
                        # index, squeeze batch dimension, output is 2d: (T, Da)
                        random_traj = trajectory_env[random_idx].detach().clone().squeeze(dim=0)
                        
                        # add it
                        best_trajs.append(random_traj)
                        
                    # random of top 5 trajs
                    else:
                        final_losses_env = final_losses_envs[i]
                        trajectory_env = trajectory_envs[i]
                    
                        # get the indices of the top 5 lowest loss trajectories
                        topk = min(5, nb_proposed_trajectories_per_env) # in case there are less than 5 proposed trajectories
                        best_indices = torch.topk(final_losses_env, k=topk, largest=False).indices
                        
                        # randomly choose one of the topk indices
                        random_idx = best_indices[torch.randint(0, topk, (1,), device=trajectory.device)]
                        
                        # index, squeeze batch dimension, output is 2d: (T, Da)
                        best_traj = trajectory_env[random_idx].detach().clone().squeeze(dim=0)
                        
                        # add it
                        best_trajs.append(best_traj)
                
        # else:
        #     best_traj = trajectory.detach().clone()
        
        # stack
        best_trajs = torch.stack(best_trajs)
            
        # final optional noise
        with torch.no_grad():
            # noise? starting to look more similar to the core DDIM function
            if False:
                trajectory += noise_weight * self.make_noise(trajectory)
            else:
                trajectory += noise_weight * self.make_per_joint_noise(trajectory)
            
            # enforce past actions (not actually needed, but nice to have for debugging)
            trajectory.data.copy_(trajectory.data * mask + fixed_contribution)
        
        return best_trajs
    
    def make_vmap_grad_batch_traj(self):
        def single_trajectory_loss(traj, global_cond):
            loss = -self.policy(traj.unsqueeze(0), 0.0, global_cond=global_cond).squeeze()
            return loss

        # 3. Vectorize the loss function over the batch dimension
        vectorized_grad_fcn = torch.vmap(functorch_apis.grad(single_trajectory_loss), in_dims=(0, None))
        
        return vectorized_grad_fcn
    
    # @profile
    def inference_batch_gd(self, nobs: dict, warm_start_trajectory = None):
        """
        use a warm start batch of trajectories and use vmap to optimize each. return the highest scoring traj from the batch
        """
        if warm_start_trajectory is None:
            raise NotImplementedError("Must supply a warm start trajectory")
        
        self.policy.eval()
        self.policy.requires_grad_(False)
        To = 1
        value = next(iter(nobs.values()))
        B = value.shape[0] # batch
        T = self.horizon # trajectory length
        Da = self.policy.action_dim # action dimension
        device = self.device
        dtype = self.dtype
        timestep = 0.0
        
        # get the grad fcn
        if self.compiled_vmap_grad_batch_traj_fcn is not None:
            get_grad_fcn = self.compiled_vmap_grad_batch_traj_fcn
        else:
            get_grad_fcn = self.vmap_grad_batch_traj_fcn

        # get the policy fcn
        if self.compiled_policy is not None:
            policy = self.compiled_policy
        else:
            policy = self.policy

        # reshape obs: B, T, ... to B*T ...
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        # get encoded obs, no grad to save time. Same global cond for all trajectories in the batch
        with torch.no_grad():
            nobs_features = policy.forward_obs_encoder(this_nobs) # type:ignore
            global_cond = nobs_features.reshape(B, -1)
            
            initial_scores = policy(warm_start_trajectory, timestep, global_cond=global_cond).squeeze()
            
        # just one sample
        assert(global_cond.shape[0] == 1)
        
        
        # Shape: (batch_size, traj_length, state_dim)
        trajectories = warm_start_trajectory.clone().detach().requires_grad_(True)
        batch_size = trajectories.shape[0]
        
        # 1. Define the optimization parameters
        step_size = self.inference_step_size
        iterations = self.inference_iterations
        
        # get past action indices for boundary condition enforcement
        past_actions = np.where(np.array(self.action_rel_indices) < 0)[0]
        original_trajectories = trajectories.clone().detach()
        
        
        #################
        # add noise if desired. initial noise. still added even if iterations == 0
        if self.use_add_inference_noise:
            trajectories += self.inference_noise * self.make_noise(trajectories)
        #################
    
        # 4. Optimization Loop (Stateless approach compatible with vmap)
        with torch.enable_grad():
                        
            for _ in range(iterations):
                # Compute scores for all trajectories in parallel
                grads = get_grad_fcn(trajectories, global_cond)
                
                # Manually update trajectories (mimicking a stateless optimizer)
                with torch.no_grad():
                    # clip the grad, max length 1.0?
                    if True:
                        bnorm = torch.norm(grads, dim=(1,2))
                        must_clip = bnorm > 1.0
                        grads[must_clip] /= bnorm[must_clip].unsqueeze(-1).unsqueeze(-1)
                    
                    # Apply update
                    trajectories -= step_size * grads
                    
                    # add noise if desired
                    if self.use_add_inference_noise:
                        trajectories += self.inference_noise * self.make_noise(trajectories)
                        
                    
                    # enforce past actions
                    trajectories[:, past_actions, :] = original_trajectories[:, past_actions, :]
                    
                    
                    
        # remove outliers?
        if True:
            threshold = 10.0 # normalized action value
            outlier = torch.any(torch.abs(trajectories) > threshold, dim=(1,2))
            trajectories = trajectories[~outlier]
            
            nb_outliers = outlier.sum()
            
        #############
        # smallest torque mags?
        if True:
            percent = 0.25
            traj_mags = torch.norm(trajectories, dim=2).mean(dim=1) # mean mag across time, for each traj in the batch
            sorted_indices = torch.argsort(traj_mags)
            
            keep_indices = sorted_indices[:int(percent * len(sorted_indices))]
            
            trajectories = trajectories[keep_indices]
        ############

        # 5. Find the highest scoring trajectory
        with torch.no_grad():
            final_scores = policy(trajectories, 0.0, global_cond=global_cond).squeeze()
            best_idx = torch.argmax(final_scores)
            
            # slice so we keep the batch dimension
            best_traj = trajectories[best_idx:best_idx+1].clone()

        self.policy.requires_grad_(True)
        self.policy.train()
        
        
        return best_traj.detach().clone()
    
    def inference(self, nobs: dict):
        if self.use_lbfgs_for_inference:
            traj = self.inference_lbfgs(nobs)
        else:
            traj = self.inference_batch_gd(nobs)
        
        
        future_traj = self.get_future_actions(traj)
        
        return future_traj, traj


    def get_future_actions(self, all_actions):
        start = np.argmax(np.array(self.action_rel_indices) >= 0)
        
        future_actions = all_actions[:, start:]
        return future_actions
        
    def make_noise(self, x, generator=None):
        """
        straight from DiffusionModel
        """
        n = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        clamp_val = globals.CONFIG.clamp_value #type:ignore
        
        if must_clamp:
            n = torch.clamp(n, -clamp_val, clamp_val)
            
        return n

    def make_per_joint_noise(self, x, generator=None):
        """
        this version creates one noise mag per joint and applies it across the trajectory. The idea is that we don't want trajectory noise to cancel itself out / oscillate like crazy, instead we want to explore in a consistent direction for a trajectory.
        """
        B = x.shape[0]
        H = x.shape[1]
        D = x.shape[2]
        
        # make noise
        n = torch.randn((B, D), dtype=x.dtype, device=x.device, generator=generator).unsqueeze(1) # shape (B, 1, D)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        clamp_val = globals.CONFIG.clamp_value #type:ignore
        
        if must_clamp:
            n = torch.clamp(n, -clamp_val, clamp_val)
            
        return n
    
    def forward(self, state_dict, action, options = None):
        this_nobs = state_dict
        
        batch_size = next(iter(this_nobs.values())).shape[0]
        
        # get encoded obs
        nobs_features = self.policy.forward_obs_encoder(this_nobs)
        
        global_cond = nobs_features.reshape(batch_size, -1)

        # predict. Might as well use the compiled policy since I already need it for inference
        if self.compiled_policy is not None:
            x = self.compiled_policy(action, global_cond=global_cond)
        else:
            x = self.policy(action, global_cond=global_cond)

        return x


    def get_val_action_mse_error(self, nbatch, task_id=None):
        # ground truth action
        nobs = nbatch['obs']
        gt_action = nbatch['action']
        
        # denoise
        future_traj, ntrajectory = self.predict_action(nobs)
        
        # extract the predicted action
        pred_action = ntrajectory
        
        # calc mse
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        
        # move to cpu
        action_mse_error = mse.item()
        
        return action_mse_error

    
    def copy(self):
        """
        Make a copy with the same parameters, but different weights
        """
        # other = copy.deepcopy(self.q1_model)
        
        # # re-init for new weight values
        # with torch.no_grad():
        #     nn.Module.__init__(other)
        
        
        other = ImplicitAlgorithm(
                        self.policy.copy(),
        )
        other.setup()
        
        return other
    
class ImplicitAlgorithmInferenceFromBCDataset(ImplicitAlgorithm):
    def __init__(self,
                 path,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path = path
    """
    very similar, just load a warm start batch of trajectories from the BC dataset
    """
    def setup(self):
        super().setup()
    
        ### using GPU actions
        if True:
            self.load_gpu_actions()

        # pre-tile actions?
        actions = self.get_gpu_actions_batch()
        
        # actions_tiled = th.tile(actions, (commons.NB_PARALLEL_ENVS, 1, 1))
            
        self.warm_start_trajectories = actions

            
    def load_gpu_actions(self):
        """
        load a config which lets use batch-load onto GPU
        """
        path = self.path # "/home/omnid/dexnex/libraries/diffusion_policy/diffusion_policy/config/sampler/sandbox_sampler_gpu_actions_rl_4.yaml"
        
        globals.load_global_config_from_path_and_save_to_globals_dict(path, "gpu_action_loader")
        
        with globals.use_config("gpu_action_loader"):
            self.my_action_rel_indices = globals.CONFIG.action_rel_indices # type: ignore
            
        self.train_action_rel_indices = self.action_rel_indices
        
        if self.my_action_rel_indices != self.train_action_rel_indices:
            print(f"Warning: my_action_rel_indices {self.my_action_rel_indices} is different from train_action_rel_indices {self.train_action_rel_indices}.")
        
    def get_gpu_actions_batch(self):
        with globals.use_config("gpu_action_loader"):
            batch_loader = globals.DEFAULT_BATCH_LOADER #type:ignore
            batch_loader.reset()
            batch = batch_loader.get_batch()
            actions = batch['action']
            return actions
        
    def inference_batch_gd(self, nobs: dict, warm_start_trajectory=None):
        """
        need to use a batch loader to deal with norming correctly
        """
        if True:
            actions = self.get_gpu_actions_batch()


        return super().inference_batch_gd(nobs, warm_start_trajectory=actions)
    
    def inference_lbfgs2(self, nobs: dict, warm_start_trajectory=None):
        """
        need to use a batch loader to deal with norming correctly
        """
        if False:
            actions = self.get_gpu_actions_batch()
            
        return super().inference_lbfgs(nobs, warm_start_trajectory=self.warm_start_trajectories)
    
    def inference_lbfgs(self, nobs: dict, warm_start_trajectory=None):
        """
        this version samples 10 random initial actions per env
        """
        actions = self.warm_start_trajectories
        nb_envs = commons.NB_PARALLEL_ENVS

        nb_per_env = 20

        # get random actions from actions
        random_actions = []
        for i in range(nb_envs):
            random_indices = torch.randint(0, actions.shape[0], (nb_per_env, ), device=actions.device)
            random_action = actions[random_indices]
            random_actions.append(random_action)
            
            # now draw nb_per_env actions from noise
            random_action = self.make_noise(actions[:nb_per_env], generator=torch.Generator(device=actions.device).manual_seed(i))
            random_actions.append(random_action)

        # stack them all, shape: (nb_envs * nb_per_env, T, Da)
        random_actions = torch.cat(random_actions, dim=0)

        return super().inference_lbfgs(nobs, warm_start_trajectory=random_actions)
