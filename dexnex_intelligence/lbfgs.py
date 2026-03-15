




import numpy as np
import torch as th
th.set_float32_matmul_precision('high') # gets rid of a warning


from diffusion_policy.common.pytorch_util import (
    dict_apply,
)

import avatar_drake_sim.sims.sandbox.sandbox_common as commons

import diffusion_policy.globals as globals

class LBFGSProblem:
    def __init__(self, 
                 policy,
                 compiled_policy,
                 horizon,
                 device,
                 dtype,
                 action_rel_indices,
                 lbfgs_options,
                 eps_greedy_eps_value = 0.0,
                 ) -> None:
        self.policy = policy
        self.compiled_policy = compiled_policy
        self.horizon = horizon
        self.device = device
        self.dtype = dtype
        self.action_rel_indices = action_rel_indices
        self.lbfgs_options = lbfgs_options
        self.eps_greedy_eps_value = eps_greedy_eps_value
        
        globals.log_one_if_exists("lbfgs/eps_greedy_eps_value", self.eps_greedy_eps_value)
        
        
    def solve(self, nobs: dict, warm_start_trajectory = None):
        """
        solve the optimization problem using L-BFGS
        """
        # setup
        self.policy.eval()
        To = 1
        value = next(iter(nobs.values()))
        nb_envs = value.shape[0] # batch
        T = self.horizon # trajectory length
        Da = self.policy.action_dim # action dimension
        device = self.device
        dtype = self.dtype
        timestep = 0.0
        total_nb_proposed_trajectories = warm_start_trajectory.shape[0] if warm_start_trajectory is not None else 0
        nb_proposed_trajectories_per_env = int(total_nb_proposed_trajectories / commons.NB_PARALLEL_ENVS) if total_nb_proposed_trajectories > 0 else 0
        
        # get the policy
        policy = self.compiled_policy if self.compiled_policy is not None else self.policy
        
        # previous action
        naction_prev = nobs['robot_joint_prev_action']

        # reshape obs: B, T, ... to B*T ...
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))

        # get encoded obs, no grad to save time
        with th.no_grad():
            nobs_features = policy.forward_obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(nb_envs, -1)
        # dummy trajectory
        if warm_start_trajectory is None:
            raise NotImplementedError("unsupported with vecenv's")
            dummy_trajectory = th.zeros(size=(B, T, Da), device=device, dtype=dtype)

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
        
        # trajectory = th.zeros((B, T, Da), device=device, requires_grad=True)
        trajectory.requires_grad_(True)
        optimizer = th.optim.LBFGS([trajectory], 
                                        lr=1.0, # designed to work with unit step size
                                        max_iter=7, 
                                        history_size=4,
                                        # tolerance_grad=1e-6, 
                                        # tolerance_change=1e-7,
                                        line_search_fn="strong_wolfe",
                                        # line_search_fn=None, # faster than strong wolfe, but uses the fixed step size from lr
                    )
        
        # initial scores for debugging
        # initial_scores = policy(trajectory, timestep, global_cond=global_cond).squeeze()
        
        # get past action indices for boundary condition enforcement
        past_actions = np.where(np.array(self.action_rel_indices) < 0)[0]
        
        # enforce initial actions
        if past_actions.shape[0] > 0:
            with th.no_grad():
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
        mask = th.ones_like(trajectory)
        mask[:, past_actions, :] = 0.0
        # Pre-aligned fixed actions
        fixed_contribution = naction_prev * (1.0 - mask)
        
        # move to device
        jerk_weight = th.tensor(jerk_weight, device=device, dtype=dtype)
        policy_weight = th.tensor(policy_weight, device=device, dtype=dtype)
        l2_weight = th.tensor(l2_weight, device=device, dtype=dtype)
        noise_weight = th.tensor(noise_weight, device=device, dtype=dtype)
        divisor = th.tensor(divisor, device=device, dtype=dtype)
        


        # initial forward/backward pass to determine the active joints based off how impactful the critic thinks each joint is. Similar to coordinate search
        # reminder: each env may have different active joints, so we do this per env
        with th.enable_grad():
            qval = policy(trajectory, timestep, global_cond=global_cond)
            loss = -qval.mean()
            loss.backward()

            # extract the grads
            grads = trajectory.grad.clone().detach()

            # new view for each env
            grads_envs = grads.view(nb_envs, nb_proposed_trajectories_per_env, T, Da)

            # abs the grads
            grads_envs_abs = grads_envs.abs()

            # mean the abs over the batch dim
            grads_envs_abs_mean = grads_envs_abs.mean(dim=1) # shape (nb_envs, T, Da)
            
            # saying the importance is proportional to the grad abs mean
            active_joints_importance = grads_envs_abs_mean

            # topk joints by mean abs grad for each env
            topk = globals.CONFIG.lbfgs_topk # TODO: schedule w.r.t. difficulty scale
            
            # know this: this may contain repeat joints in different waypoints
            topk_values, topk_indices = th.topk(active_joints_importance, k=topk, dim=-1) # shape (nb_envs, T, topk)
                        
            # epsilon mask for topk_indices
            eps_mask = th.rand(topk_indices.shape, device=device) < self.eps_greedy_eps_value
            
            # weighted random indices
            weights = th.ones((nb_envs*T, Da), device=device)
            
            if True:
                weights[:, :6] *= globals.CONFIG.lbfgs_active_joints_importance_gofa_multiplier # type: ignore
                
            # set replacement to False to get more variety
            random_indices = th.multinomial(weights, num_samples=topk_indices.shape[-1], replacement=False).view(topk_indices.shape)
                        
            topk_indices = th.where(eps_mask, random_indices, topk_indices)
            
            # # log em
            # for i in range(T):
            #     globals.log_one_if_exists(f"lbfgs/{i}_active_joints", int(topk_indices.squeeze()[i].cpu().numpy()))

            # make a mask of the active joints, shape (nb_envs, T, Da)
            active_joints_mask_one_per_env = th.zeros_like(active_joints_importance).scatter_(-1, topk_indices, 1.0)

            # reshape and tile the mask. shape: (B, T, Da), recall that B = nb_envs * nb_proposed_trajectories_per_env and each nb_proposed_trajectories_per_env pertains to a specific env
            active_joints_mask = active_joints_mask_one_per_env.view(nb_envs, 1, T, Da).expand(-1, nb_proposed_trajectories_per_env, -1, -1).reshape(total_nb_proposed_trajectories, T, Da)
            
            active_joints_mask_one_per_env = active_joints_mask_one_per_env.view(nb_envs, T, Da)

        def closure_setup(trajectory):
            # no grad
            with th.no_grad():
                # add noise
                trajectory += noise_weight * self.make_per_joint_noise(trajectory)

                # mask inactive joints
                trajectory.data.copy_(trajectory.data * active_joints_mask)
                    
                # enforce past actions
                trajectory.data.copy_(trajectory.data * mask + fixed_contribution)
                
            return trajectory

        
        def closure_get_per_sample_loss():
            nonlocal trajectory
            
            # zero out the grads, necessary for lbfgs? idk.
            optimizer.zero_grad()
            
            # setup
            trajectory = closure_setup(trajectory)
            
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
            per_sample_loss = closure_get_per_sample_loss()
            
            loss = per_sample_loss.mean()
            loss.backward()
            
            # logging
            # losses.append(loss.item())
            return loss
        
        # initial losses
        with th.no_grad():
            initial_losses = closure_get_per_sample_loss()
                 
        # speed up .backward() by not calculating gradients for the policy parameters
        # self.policy.requires_grad_(False) # might actually be slower
        # just to make sure
        with th.enable_grad():
            # only need to call step once    
            optimizer.step(closure) #type:ignore
        # restore grad for policy parameters
        # self.policy.requires_grad_(True)

        self.policy.train()
        
        # final noise
        trajectory = closure_setup(trajectory)

        # 5. Find the highest scoring trajectory
        # if trajectory.shape[0] > 0:
        best_trajs = []
        with th.no_grad():
            # final action enforcement (also should be done in get_per_sample_loss)
            trajectory[:, past_actions, :] = naction_prev
            final_losses = closure_get_per_sample_loss()
            
            # reshape for easy indexing
            trajectory_envs = trajectory.view(nb_envs, nb_proposed_trajectories_per_env, T, Da)
            final_losses_envs = final_losses.view(nb_envs, nb_proposed_trajectories_per_env)
            
            # split by env
            for i in range(nb_envs):
                    if False:
                        final_losses_env = final_losses_envs[i]
                        trajectory_env = trajectory_envs[i]
                    
                        # min the loss
                        best_idx = th.argmin(final_losses_env)
                        
                        # index, output is 2d: (T, Da)
                        best_traj = trajectory_env[best_idx].detach().clone()
                        
                        best_trajs.append(best_traj)
                        
                    # random traj
                    elif False:
                        trajectory_env = trajectory_envs[i]
                        
                        # random index
                        random_idx = th.randint(0, nb_proposed_trajectories_per_env, (1,), device=trajectory.device)
                        
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
                        best_indices = th.topk(final_losses_env, k=topk, largest=False).indices
                        
                        # randomly choose one of the topk indices
                        random_idx = best_indices[th.randint(0, topk, (1,), device=trajectory.device)]
                        
                        # index, squeeze batch dimension, output is 2d: (T, Da)
                        best_traj = trajectory_env[random_idx].detach().clone().squeeze(dim=0)
                        
                        # add it
                        best_trajs.append(best_traj)

        # stack
        best_trajs = th.stack(best_trajs)
        
        # eps greedy mask on the action magnitude as well?
        if True:
            eps_mask = th.rand(best_trajs.shape, device=best_trajs.device) < globals.CONFIG.lbfgs_random_action_eps_value # type: ignore
            
            random_trajs = self.make_per_joint_noise(best_trajs)
            
            # mult by active joints mask
            assert(active_joints_mask_one_per_env.shape == best_trajs.shape)
            random_trajs = random_trajs * active_joints_mask_one_per_env
            
            best_trajs = th.where(eps_mask, random_trajs, best_trajs)

        # we're done
        return best_trajs
    
    
    def make_per_joint_noise(self, x, generator=None):
        """
        this version creates one noise mag per joint and applies it across the trajectory. The idea is that we don't want trajectory noise to cancel itself out / oscillate like crazy, instead we want to explore in a consistent direction for a trajectory.
        """
        B = x.shape[0]
        H = x.shape[1]
        D = x.shape[2]
        
        # make noise
        n = th.randn((B, D), dtype=x.dtype, device=x.device, generator=generator).unsqueeze(1) # shape (B, 1, D)
        
        must_clamp = globals.CONFIG.clamp #type:ignore
        clamp_val = globals.CONFIG.clamp_value #type:ignore
        
        if must_clamp:
            n = th.clamp(n, -clamp_val, clamp_val)
            
        return n