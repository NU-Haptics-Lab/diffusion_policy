import numpy as np
from collections import defaultdict

import torch
from torch import nn

import diffusion_policy.globals as globals

from diffusion_policy.dataset.batch_loader import BatchLoader
from diffusion_policy.losses.actor_loss import ActorLoss
from diffusion_policy.model.diffusion_ql.diffusion_ql_loss import CriticLoss

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sarsa_sampler import DatasetSampler, Indices
from diffusion_policy.dataset.train_and_val import TrainAndVal

from diffusion_policy.model.model import ModelEmaOptim
from diffusion_policy.model.diffusion_ql.attractor import Attractor
from diffusion_policy.model.diffusion_ql.attractor import EnergyPenalty

from diffusion_policy.common.pytorch_util import dict_tensor_to
from diffusion_policy import utils

class BatchLoss:
    """
    Responsible for computing loss for one batch.
    Bundles the batch loader for a dataset and the policy heads associated with that dataset. Useful in co-training
    """

    def __init__(self,
            batch_loader: BatchLoader,
            eta: float = 0.0, # weight of the critic loss
            use_bc_loss = True,
            freqs = {}, # training frequencies, units: steps
        ):
        self.batch_loader = batch_loader
        self.eta = eta
        self.use_bc_loss = use_bc_loss
        self.freqs = freqs
        
        # save handles to nodes
        self.actor: ModelEmaOptim = globals.MODELS["actor"] #type:ignore
        self.actor_model = self.actor.get_model()
        self.critic: CriticLoss = globals.MODELS["critic"] #type:ignore
        
        # get the rb_id
        self.rb_id = self.batch_loader.rb_id

        # my members
        self.current_batch: dict

    def compute_loss(self):
        """
        Train for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch # save

        # get the BC loss
        if self.use_bc_loss:
            actor_loss, a0, timesteps = self.actor.loss(nbatch, self.rb_id)
            actor_loss = actor_loss.mean()
        else:
            actor_loss = 0.0

        # we're done
        losses = {
            'actor': 
                {
                    'bc': actor_loss,
                },
        }
        return losses
    
    def eval(self):
        # get the actor loss
        t = self.compute_loss()
        l = t['actor']['bc']
        loss = l.cpu()

        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=self.rb_id)

        return loss, action_mse_error
    
class BCBatchLoss(BatchLoss):
    pass
    
class TTREfficiencyWeightedBatchLoss(BatchLoss):
    """
    TTR - time-to-reward.
    Weight each sample in the batch based on the time-to-reward effiency
    """
    is_setup = defaultdict(bool)
    efficiencies = {} # so hacky...
    
    def __init__(self, batch_loader, 
                 eta=0.0,
                 on_gpu = True,
                 alpha = 0.75,
                 ):
        super().__init__(batch_loader, eta)
        self.on_gpu = on_gpu
        self.alpha = alpha

        self.setup()

    def setup(self):
        """
        compute the efficiencies. Each datapoint in a reward-yielding trajectory should have the same efficiency because it's the single-value efficiency of the trajectory.

        Might have to correct if we pad the dataset episodes, not sure
        """
        tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        if self.rb_id not in tasks_to_use:
            return
        
        if TTREfficiencyWeightedBatchLoss.is_setup[self.rb_id]:
            return
        
        dataloader: TrainAndVal = globals.DATALOADERS[self.rb_id]
        dss: DatasetSampler = dataloader.sampler #type:ignore ide error from sars vs sarsa. can ignore
        ttrs = -1 * np.ones((len(dss),)) # TODO: use len of the replay buffer dataset instead of the sampler (which might be padded)

        # traverse the dataset one episode at a time
        c = 0
        for ep in dss.episodes:
            c += 1
            if c%10==0:
                print("{} of {}".format(c, len(dss.episodes)))
            if globals.CONFIG.debug and c%20==0:
                break
            
            # loop vars
            reward_timestep = None
            old_reward_timestep = ep.get_id(0)

            # traverse through each datapt in the ep
            for i in range(len(ep)):
                r = ep.get_reward(i)

                # update reward
                if r > 0.0:
                    # check for false positive
                    # assume any time-to-reward less than 2.0 seconds (20 steps) was a false positive from the end of the previous episode so don't update the reward
                    if i > 20:
                        reward_timestep = ep.get_id(i)
                        
                        ttr = reward_timestep - old_reward_timestep
                        
                        ttrs[old_reward_timestep:reward_timestep] = ttr
                        
                        old_reward_timestep = reward_timestep + 1
                        
        # set all negative ttr's to the max
        ttrs[ttrs < 0.0] = ttrs.max()
        
        # efficiency between [0, 1] where 0 == max ttr, and 1 == min ttr
        mm = np.max(ttrs)
        mn = np.min(ttrs)
        
        # if all TTR's are equal, then set to all ones
        if mm == mn:
            efficiency = np.ones_like(ttrs)
        else:
            efficiency = (np.max(ttrs) - ttrs) / (np.max(ttrs) - np.min(ttrs))
        
        if self.on_gpu:
            efficiency = torch.tensor(efficiency, device=globals.CONFIG.device) #type:ignore

        TTREfficiencyWeightedBatchLoss.efficiencies[self.rb_id] = efficiency
        
        TTREfficiencyWeightedBatchLoss.is_setup[self.rb_id] = True
        
        pass
        

    def get_weights(self, indices: torch.Tensor):
        """
        Each sample starts with weight 1.0, and is given a higher weight if its trajectory more efficiently gets to a reward state
        """
        if not self.on_gpu:
            indices = indices.cpu()
        indices = indices.to(dtype=torch.long)
        
        b = indices.shape[0]
        base_weight = torch.ones([b, 1], device=globals.CONFIG.device) #type:ignore

        # time-to-reward efficiency
        ttr_eff = TTREfficiencyWeightedBatchLoss.efficiencies[self.rb_id][indices]

        # want to give a sample more weight if it's more efficient
        weights = (1.0 - self.alpha) * base_weight + self.alpha * ttr_eff

        return weights

    def compute_loss(self):
        """
        Train for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch # save

        # # get the BC loss
        loss, a0, timesteps = self.actor.loss(nbatch, self.rb_id)

        indices = nbatch["rb_index"]

        # get the TTR weight
        batch_weights = self.get_weights(indices)
        
        if not self.on_gpu:
            batch_weights = torch.tensor(batch_weights, device=globals.CONFIG.device) #type:ignore

        # element-wise multiplication
        weighted_loss = torch.mul(loss, batch_weights)
        
        mean_weighted_loss = weighted_loss.mean()
        losses = {
            'actor': mean_weighted_loss
        }
        
        # logging
        globals.LOGGER.log_one("BC/" + self.rb_id + ": bc_actor_loss", mean_weighted_loss)
        return losses
    
    def eval(self):
        loss = self.compute_loss()['actor']['bc'].cpu()
        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=self.rb_id)
        
        # right now eval is hard-coded to expect two tensors on cpu
        return loss, action_mse_error
    
class QvalWBatchLoss(TTREfficiencyWeightedBatchLoss):
    """
    Explicit Q-Val weighted batch loss
    """

    def setup(self):
        """
        compute the efficiencies. Each datapoint in a reward-yielding trajectory should have the same efficiency because it's the single-value efficiency of the trajectory.

        Might have to correct if we pad the dataset episodes, not sure
        """
        tasks_to_use = globals.CONFIG.tasks_to_use #type:ignore
        if self.rb_id not in tasks_to_use:
            return
        
        if TTREfficiencyWeightedBatchLoss.is_setup[self.rb_id]:
            return
        
        dataloader: TrainAndVal = globals.DATALOADERS[self.rb_id]
        dss: DatasetSampler = dataloader.sampler #type:ignore ide error from sars vs sarsa. can ignore
        ttrs = np.zeros((len(dss),)) # TODO: use len of the replay buffer dataset instead of the sampler (which might be padded)
        
        gamma = 0.995

        # traverse the dataset one episode at a time
        c = 0
        for ep in dss.episodes:
            c += 1
            if c%10==0:
                print("{} of {}".format(c, len(dss.episodes)))
            if globals.CONFIG.debug and c%20==0: #type:ignore
                break
            
            # loop vars
            q = 0.0

            # traverse through each datapt in the ep, in reverse order
            for i in reversed(range(len(ep))):
                r = ep.get_reward(i)
                id = ep.get_id(i)

                # update reward
                if r > 0.0:
                    # check for false positive
                    # assume any time-to-reward less than 2.0 seconds (20 steps) was a false positive from the end of the previous episode so don't update the reward
                    if i > 20:
                        # degrade q
                        q *= gamma
                        
                        # add on new rewards
                        q += r
                        
                else:
                    q *= gamma
                    
                ttrs[id] = q
                
            pass
                        
                        
        # efficiency between [0, 1] where 0 == max ttr, and 1 == min ttr
        mm = np.max(ttrs)
        mn = np.min(ttrs)
        
        # if all TTR's are equal, then set to all ones
        if mm == mn:
            efficiency = np.ones_like(ttrs)
        else:
            # ttrs is already [0, r]
            efficiency = ttrs / mm
        
        if self.on_gpu:
            efficiency = torch.tensor(efficiency, device=globals.CONFIG.device) #type:ignore

        TTREfficiencyWeightedBatchLoss.efficiencies[self.rb_id] = efficiency
        
        TTREfficiencyWeightedBatchLoss.is_setup[self.rb_id] = True
        
        pass
        
class DQLBatchLoss(BatchLoss):
    """
    Compute actor and critic loss and return them in a dictionary
    """
    def get_bc_losses(self):
        loss_arr, a0, timesteps = self.actor.loss(self.current_batch, self.rb_id)
            
        return loss_arr, a0, timesteps
    
    def get_bc_loss(self, is_eval=False):
        # get the BC loss
        bc_loss = utils.InitZeroTensorOnDevice()
        a0 = None
        timesteps = None
        
        if utils.StepFreqTrigger(self.freqs['actor']) or is_eval:
            loss_arr, a0, timesteps = self.get_bc_losses()
            bc_loss = loss_arr.mean()
            
        return bc_loss, a0, timesteps
    
    def compute_loss(self, 
                     is_eval=False, 
                     ):
        """
        compute loss for one batch.
        """
        losses = {}
        
        # flags
        models_to_train: list = globals.CONFIG.models_to_train # type: ignore
        use_dql: bool = globals.CONFIG.use_dql # type: ignore
        
        # for key in models_to_train:
        #     losses[key] = utils.InitZeroTensorOnDevice()
        losses = {
            'critic': utils.InitZeroTensorOnDevice(),
            'actor': {
                'bc': utils.InitZeroTensorOnDevice(),
                'dql': utils.InitZeroTensorOnDevice(),
                'attractor': utils.InitZeroTensorOnDevice(),
            }
        }

        train_actor = "actor" in models_to_train
        train_critic = "critic" in models_to_train
        need_dql_actor_loss = train_actor and use_dql
        
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch
        
        a0 = None
        
        # if we need the BC output
        if True:
            bc_loss, a0, timesteps = self.get_bc_loss(is_eval)
            
        # save
        self.a0 = a0
        
        # if we're training using BC loss
        if train_actor and self.use_bc_loss:
            # hack
            losses['actor']['bc'] = bc_loss
            
            # hack
            if is_eval:
                return losses
        
        # need critic loss if we're training critic, need actor loss if we're using dql
        if train_critic or need_dql_actor_loss:
            # get the DQL losses
            dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id, a0, timesteps)
            
            if train_critic:
                losses['critic'] = dql_critic_loss
            
            # dql_actor_loss shouldn't be None as long as need_dql_actor_loss is True, but I had to add it for pylance
            if need_dql_actor_loss and dql_actor_loss is not None and utils.StepFreqTrigger(self.freqs['actor']):
                # add on the DQL actor loss
                losses['actor']['dql'] = self.eta * dql_actor_loss
        
        # we're done
        return losses
    
    # TODO: rename all eval to validate
    def eval(self):
        loss = 0.0
        action_mse_error = 0.0
        
        if "actor" in globals.CONFIG.models_to_train: #type:ignore
            t = self.compute_loss(is_eval=True)['actor']['bc']
            loss = t.cpu()
        
            # get the action mse error
            action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=self.rb_id)
        
        # right now eval is hard-coded to expect two tensors on cpu
        return loss, action_mse_error
    
class CriticWeightedBC(DQLBatchLoss):
    """
    Train the actor via weighted BC and train the critic like normal.
    BC weighting comes from how good the critic thinks a sample is.
    Normally in SAC we back-propagate the critic directly into the actor, but I don't like that in high-dimensions because I think it leads to adversarial actors too easily.
    So this is a way for the critic to inform the policy without being directly connected and thereby risking critic exploitation.
    """
    def __init__(self, 
                 sqval_scale,
                 sqval_offset,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.sqval_scale = sqval_scale
        self.sqval_offset = sqval_offset
        
    def init_losses(self):
        losses = {}
        
        # for key in models_to_train:
        #     losses[key] = utils.InitZeroTensorOnDevice()
        losses = {
            'critic': utils.InitZeroTensorOnDevice(),
            'actor': {
                'bc': utils.InitZeroTensorOnDevice(),
                'dql': utils.InitZeroTensorOnDevice(),
                'attractor': utils.InitZeroTensorOnDevice(),
            }
        }
        return losses
    
    def get_next_batch(self):
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch
        return self.current_batch

    def compute_loss(self, 
                     is_eval=False, 
                     ):
        """
        compute loss for one batch.
        """
        losses = self.init_losses()

        # flags
        models_to_train: list = globals.CONFIG.models_to_train # type: ignore
        use_dql: bool = globals.CONFIG.use_dql # type: ignore

        train_actor = "actor" in models_to_train
        train_critic = "critic" in models_to_train
        
        # get the batch from the batch loader
        nbatch = self.get_next_batch()
        
        self.a0 = None
        
        # always need bc-loss and a0 now
        loss_arr, a0, timesteps = self.get_bc_losses()
        # save
        self.a0 = a0
        
        if utils.StepFreqTrigger(self.freqs['critic']):
            # get the DQL losses
            _, dql_critic_loss = self.critic.loss(nbatch, self.rb_id, a0, timesteps)
            
            if train_critic:
                losses['critic'] = dql_critic_loss
        
        if utils.StepFreqTrigger(self.freqs['actor']) or is_eval:
            # hack
            if is_eval:
                bc_loss = loss_arr.mean()
                losses['actor']['bc'] = bc_loss
                return losses
            
            
            ## Critic weighted BC
            if self.use_bc_loss:
                # eval each BC sample (no backprop) using a0
                qvals = self.critic.infer(nbatch, a0, self.rb_id)
                
                globals.LOGGER.log_one("cbc/avg_qval/" + self.rb_id, qvals.mean())

                # use sigmoid to convert the range to [0, 1]
                s = nn.Sigmoid()
                
                # scale qvals 
                qvals2 = qvals * self.sqval_scale
                
                # offset qvals
                qvals3 = qvals2 + self.sqval_offset
                
                # sigmoid the scaled qvals
                sqvals = s(qvals3)
                sqvals2 = torch.squeeze(sqvals)
                
                globals.LOGGER.log_one("cbc/sqvals/" + self.rb_id, sqvals2.mean())
                
                # could weight sqvals by timestep, but try this first

                # weight BC samples w.r.t the sigmoid qvals
                loss_arr2 = loss_arr.mean(axis=1)
                assert(sqvals2.shape == loss_arr2.shape)
                w_bc_loss = loss_arr2 * sqvals2
                
                assert(w_bc_loss.shape == loss_arr2.shape)

                losses['actor']['bc'] = w_bc_loss.mean()
        
        # we're done
        return losses
    
class AttractorLoss(DQLBatchLoss):
    """
    With an attractor.
    
    """
    def __init__(self,
                 attractor: Attractor,
                 energy_penalty: EnergyPenalty,
                 *args,
                 **kwargs,
                ):
        super().__init__(*args, **kwargs)
        
        self.attractor = attractor
        self.energy_penalty = energy_penalty
        
    def compute_loss(self, 
                     is_eval=False, 
                     ):
        losses = super().compute_loss(is_eval)
        losses['actor']['attractor'] = utils.InitZeroTensorOnDevice()
        
        if utils.GlobalStepFreqTrigger('actor'):
            # somehow get a0
            a0 = self.a0
            state = self.current_batch['obs']['state']
            
            # none protection
            if (a0 is None) or (state is None):
                return losses
                
            # call the attractor
            l = self.attractor.forward(state, a0)
            
            lmean = l.mean()
            
            losses['actor']['attractor'] += lmean
                
            # call the energy penalty
            l = self.energy_penalty.forward(state, a0)
            
            # sum across all waypoints
            lsum = l.sum()
            
            losses['actor']['attractor'] += lsum
            
            globals.LOGGER.log_one("AttractorLoss/lsum", lsum)
        
        return losses

class CriticBatchLoss(BatchLoss):
    """
    Compute actor and critic loss and return them in a dictionary
    
    OUT OF DATE
    """
    
    def compute_loss(self):
        """
        compute loss for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch

        # get the DQL losses
        dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id)
        
        # summed critic loss
        critic_loss = dql_critic_loss
        
        losses = {
            'critic': critic_loss
        }
        
        # we're done
        return losses
    
    # TODO: rename all eval to validate
    def eval(self):
        loss = self.compute_loss()['actor']['bc'].cpu()
        
        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=self.rb_id)
        
        # right now eval is hard-coded to expect two tensors on cpu
        return loss, action_mse_error
        

    
class WeightedBatchLoss:
    """
    Bundles a BatchLoss and a weight parameter. Useful in co-training
    """
    def __init__(self,
        batch_loss: BatchLoss,
        weight: float
        ):
        self.batch_loss = batch_loss
        self.weight = weight

    def compute_weighted_loss(self):
        losses = self.batch_loss.compute_loss()
        
        # losses can now be a nested dict
        # hack
        wloss = {}
        wloss['actor'] = dict_apply(losses['actor'], lambda x: self.weight * x)
        
        if 'critic' in losses:
            wloss['critic'] = losses['critic'] * self.weight
        
        # apply the weighting to each loss
        # wloss = dict_apply(losses, lambda x: self.weight * x)
        return wloss
    
    # TODO: rename eval to validate
    def compute_weighted_eval(self):
        # this batch_loss should already be in val mode
        wevals = self.weight * np.array(self.batch_loss.eval())
        return wevals
