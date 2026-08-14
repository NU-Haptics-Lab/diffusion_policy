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


def match_and_compute_keypoint_loss(pred_kpts, gt_px, valid, image_size):
    """
    Aux loss for supervising a spatial-softmax obs key's keypoints against
    known landmark pixel locations (e.g. pellets visible in a wrist camera
    image) -- see BatchLoss.keypoint_loss_weight.

    The keypoints aren't semantically ordered (symmetric at init; nothing
    ties "keypoint 3" to "pellet slot 1"), and the ground-truth slots
    (padded to a fixed count) have no consistent identity across samples
    either -- slot 0 isn't "the leftmost pellet", just whichever pellet the
    data-generation script happened to find first. So which keypoint should
    track which pellet isn't fixed; it's solved per-sample by nearest-
    neighbor matching in the model's own coordinate space, then only the
    matched pairs get gradient.

    Matching is greedy (process ground-truth slots in order, assign each to
    its nearest not-yet-taken keypoint for that sample), not the globally
    optimal Hungarian assignment -- simpler, fully vectorized across the
    batch (no CPU sync), and fine for this few points (<=4) against this
    many keypoints (16 in the current config): collisions forcing a
    suboptimal match are rare. Matching itself runs under no_grad (it's a
    discrete assignment); the returned loss still backpropagates through
    pred_kpts normally via the final gather + MSE.

    pred_kpts: (B, K, 2) predicted keypoints, [-1,1]^2, gradient-attached
    gt_px:     (B, N, 2) ground-truth landmark pixel coords, 0-padded past
               however many are valid for that sample
    valid:     (B, N) bool/float mask, 1 = real landmark, 0 = padding
    image_size: the image's pixel width/height (assumed square), for
               converting gt_px into the same [-1,1]^2 convention as
               pred_kpts (matches DexNexTransformerForDiffusion's ss_grid:
               torch.linspace(-1, 1, image_size) per axis)

    Returns a scalar: mean squared L2 distance over every valid (sample,
    landmark) pair in the batch, or exactly 0.0 if the whole batch has no
    valid landmarks at all.
    """
    B, K, _ = pred_kpts.shape
    N = gt_px.shape[1]
    valid = valid.bool()

    gt_norm = 2.0 * gt_px / (image_size - 1) - 1.0  # (B, N, 2)

    with torch.no_grad():
        dist = torch.cdist(gt_norm, pred_kpts.detach())  # (B, N, K)
        assigned = torch.zeros(B, N, dtype=torch.long, device=pred_kpts.device)
        taken = torch.zeros(B, K, dtype=torch.bool, device=pred_kpts.device)
        for n in range(N):
            d = dist[:, n, :].masked_fill(taken, float('inf'))
            idx = d.argmin(dim=-1)  # (B,)
            assigned[:, n] = idx
            # only consume a keypoint slot for samples where THIS landmark
            # is actually valid -- otherwise a padding slot could steal the
            # nearest keypoint away from a later real landmark in the same
            # sample (if valid landmarks aren't all sorted before padding)
            take_onehot = nn.functional.one_hot(idx, K).bool() & valid[:, n:n+1]
            taken = taken | take_onehot

    matched_pred = torch.gather(pred_kpts, 1, assigned.unsqueeze(-1).expand(-1, -1, 2))  # (B, N, 2)

    sq_err = (matched_pred - gt_norm).pow(2).sum(dim=-1)  # (B, N)
    valid_f = valid.float()
    # clamp(min=1.0) instead of a zero-check + branch: avoids a per-step
    # GPU->CPU sync (`.item()`/`if tensor:` both block on the device), and
    # is equivalent -- when the batch has zero valid landmarks, valid_f is
    # all zero so the numerator is already 0, and 0 / 1 == 0
    denom = valid_f.sum().clamp(min=1.0)
    return (sq_err * valid_f).sum() / denom


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
            loss_clip_value = 1.0,
            remove_outlier_losses = False, # use carefully
            subtask_weights = None, # optional: {subtask_id: weight}, applied per-sample to the bc loss and logged separately. No-op if nbatch has no 'subtask_id'.
            keypoint_loss_weight = 0.0, # aux loss weight for match_and_compute_keypoint_loss; 0.0 (default) disables it entirely
            keypoint_obs_key = None, # which spatial-softmax obs key to supervise -- only meaningful with keypoint_source="transformer" (e.g. "wrist_camera_image")
            keypoint_image_size = 64, # pixel width/height keypoint_gt_px_key's raw coords are in (NOT the patch grid's H/W -- e.g. wrist_pellet_keypoints_px is in 64x64 pixel space even when the model reads a 14x14 DINOv3 patch grid)
            keypoint_gt_px_key = "wrist_pellet_keypoints_px", # nbatch key: (B,1,N,2) ground-truth pixel coords
            keypoint_valid_key = "wrist_pellet_keypoints_valid", # nbatch key: (B,1,N) validity mask
            keypoint_source = "transformer", # "transformer" (DexNexTransformerForDiffusion.get_last_spatial_softmax_keypoints) or "pellet_localizer" (DiffusionModel.get_last_pellet_prediction)
            pellet_xyz_loss_weight = 0.0, # aux loss weight for the pellet localizer's xyz regression; 0.0 (default) disables it. Only meaningful with keypoint_source="pellet_localizer"
            pellet_xyz_gt_key = "target_pellet_location", # nbatch['obs'] key: (B,1,3) ground-truth pellet xyz
            pellet_classifier_loss_weight = 0.0, # aux loss weight for the pellet localizer's has-pellet classifier; 0.0 (default) disables it
            pellet_valid_key = "target_pellet_valid", # nbatch top-level key: (B,1) ground-truth has-pellet label; falls back to keypoint_valid_key.any(-1) if absent
        ):
        self.batch_loader = batch_loader
        self.eta = eta
        self.use_bc_loss = use_bc_loss
        self.freqs = freqs
        self.loss_clip_value = loss_clip_value
        self.remove_outlier_losses = remove_outlier_losses
        self.subtask_weights = subtask_weights
        self.keypoint_loss_weight = keypoint_loss_weight
        self.keypoint_obs_key = keypoint_obs_key
        self.keypoint_image_size = keypoint_image_size
        self.keypoint_gt_px_key = keypoint_gt_px_key
        self.keypoint_valid_key = keypoint_valid_key
        assert keypoint_source in ("transformer", "pellet_localizer")
        self.keypoint_source = keypoint_source
        self.pellet_xyz_loss_weight = pellet_xyz_loss_weight
        self.pellet_xyz_gt_key = pellet_xyz_gt_key
        self.pellet_classifier_loss_weight = pellet_classifier_loss_weight
        self.pellet_valid_key = pellet_valid_key
        
    def __len__(self):
        return len(self.batch_loader)
        
    def setup(self):
        # save handles to nodes
        self.actor: ModelEmaOptim = globals.MODELS["actor"] #type:ignore
        self.actor_model = self.actor.get_model()
        # self.critic: CriticLoss = globals.MODELS["critic"] #type:ignore
        
        # get the rb_id
        self.rb_id = self.batch_loader.rb_id

        # my members
        self.current_batch: dict = None #type:ignore
        self.a0 = None
        
        self.batch_loader.setup()
        
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
                'keypoint': utils.InitZeroTensorOnDevice(),
                'pellet_xyz': utils.InitZeroTensorOnDevice(),
            }
        }
        return losses

    def _get_last_keypoints(self):
        """
        Returns the (B, K, 2) predicted keypoints (gradient-attached) to
        supervise with match_and_compute_keypoint_loss, from whichever
        source keypoint_source names, or None if unavailable.
        """
        if self.keypoint_source == "pellet_localizer":
            _, _, keypoints = self.actor_model.get_last_pellet_prediction()
            return keypoints
        else:
            transformer = getattr(self.actor_model.model, 'transformer', None)
            if transformer is None or self.keypoint_obs_key is None:
                return None
            return transformer.get_last_spatial_softmax_keypoints(self.keypoint_obs_key)

    def compute_keypoint_loss(self, nbatch):
        """
        See match_and_compute_keypoint_loss. Reuses whichever module
        (keypoint_source) already computed the keypoints as a side effect of
        compute_sample_loss() (via self.actor.loss(...)) on this exact
        nbatch -- no extra forward pass. Only meaningful right after
        compute_sample_loss() has run this step; the caller enforces that
        ordering (see compute_loss below).
        """
        if self.keypoint_loss_weight <= 0.0:
            return utils.InitZeroTensorOnDevice()

        pred_kpts = self._get_last_keypoints()
        if pred_kpts is None or self.keypoint_gt_px_key not in nbatch:
            return utils.InitZeroTensorOnDevice()

        # (B, 1, N, 2)/(B, 1, N) -- squeeze the obs-history dim (obs_rel_indices
        # is a single "current" index for this key, same as any other obs)
        gt_px = nbatch[self.keypoint_gt_px_key].squeeze(1)
        valid = nbatch[self.keypoint_valid_key].squeeze(1)

        # keypoint_gt_px_key's coords are in the ORIGINAL pixel image's
        # space regardless of what grid resolution the model's spatial
        # softmax runs over (a 14x14 DINOv3 patch grid tiles the same image
        # edge to edge as the raw 64x64 pixels would, so both land in the
        # same [-1,1]^2 after normalization) -- keypoint_image_size is that
        # original pixel size, set explicitly rather than derived from
        # shape_meta, since there may be no obs key holding the raw image
        # at all once vision is routed through pre-extracted patch features.
        loss = match_and_compute_keypoint_loss(pred_kpts, gt_px, valid, self.keypoint_image_size)
        globals.LOGGER.log_one(f"BC/{self.rb_id}: keypoint_loss", loss)
        return self.keypoint_loss_weight * loss

    def compute_pellet_xyz_loss(self, nbatch):
        """
        Two losses from the pellet localizer's other two outputs (see
        PelletLocalizer): an xyz regression MSE against the ground-truth
        pellet location, and a has-pellet classifier BCE.

        The xyz loss is masked to samples where a pellet is actually visible
        (pellet_valid_key, e.g. target_pellet_valid -- the authoritative
        ground-truth label for whether target_pellet_location means
        anything for this sample; falls back to
        keypoint_valid_key.any(-1) if pellet_valid_key isn't in the batch)
        -- xyz is meaningless supervision on a frame with nothing to
        localize. Masked by the GROUND TRUTH label, not the classifier's
        own (possibly wrong, especially early in training) prediction, so a
        bad classifier doesn't also corrupt the xyz loss's signal. The
        classifier itself is supervised on every sample, valid or not --
        that's the whole point of it.
        """
        if self.pellet_xyz_loss_weight <= 0.0 and self.pellet_classifier_loss_weight <= 0.0:
            return utils.InitZeroTensorOnDevice()
        if self.keypoint_source != "pellet_localizer":
            return utils.InitZeroTensorOnDevice()

        pred_xyz, has_pellet_logit, _ = self.actor_model.get_last_pellet_prediction()
        if pred_xyz is None:
            return utils.InitZeroTensorOnDevice()

        has_pellet_gt = None
        if self.pellet_valid_key in nbatch:
            has_pellet_gt = nbatch[self.pellet_valid_key].squeeze(1).float().reshape(-1)  # (B,)
        elif self.keypoint_valid_key in nbatch:
            has_pellet_gt = nbatch[self.keypoint_valid_key].squeeze(1).bool().any(dim=-1).float()  # (B,)

        total = utils.InitZeroTensorOnDevice()

        if self.pellet_xyz_loss_weight > 0.0 and self.pellet_xyz_gt_key in nbatch['obs'] and has_pellet_gt is not None:
            gt_xyz = nbatch['obs'][self.pellet_xyz_gt_key].squeeze(1)  # (B, 3)
            sq_err = (pred_xyz - gt_xyz).pow(2).sum(dim=-1)  # (B,)
            denom = has_pellet_gt.sum().clamp(min=1.0)  # avoid a GPU->CPU sync (see match_and_compute_keypoint_loss)
            xyz_loss = (sq_err * has_pellet_gt).sum() / denom
            globals.LOGGER.log_one(f"BC/{self.rb_id}: pellet_xyz_loss", xyz_loss)
            total = total + self.pellet_xyz_loss_weight * xyz_loss

        if self.pellet_classifier_loss_weight > 0.0 and has_pellet_gt is not None:
            classifier_loss = nn.functional.binary_cross_entropy_with_logits(has_pellet_logit, has_pellet_gt)
            globals.LOGGER.log_one(f"BC/{self.rb_id}: pellet_classifier_loss", classifier_loss)
            total = total + self.pellet_classifier_loss_weight * classifier_loss

        return total


    def get_next_batch(self):
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch
        return self.current_batch
    
    def save_bc_loss(self, losses, actor_loss):
        """
        Save and Log
        """
        assert(not actor_loss.isnan().any())
        losses['actor']['bc'] = actor_loss
        
        globals.LOGGER.log_one("BC/" + self.rb_id + ": bc_actor_loss", actor_loss)
    
    def log_and_weight_by_subtask(self, nbatch, sample_loss):
        """
        Log each subtask_id's (raw, unweighted) loss separately, then apply
        self.subtask_weights per-sample before returning. No-op if the batch
        has no 'subtask_id' key.
        """
        if 'subtask_id' not in nbatch:
            return sample_loss

        subtask_id = torch.reshape(nbatch['subtask_id'], [-1])

        with torch.no_grad():
            unique_subtask_ids = torch.unique(subtask_id)
            # skip the per-subtask breakdown (and its .tolist() CPU sync) when
            # the whole batch is one subtask -- it'd just duplicate the total
            if unique_subtask_ids.numel() > 1:
                for sid in unique_subtask_ids.tolist():
                    mask = subtask_id == sid
                    if mask.any():
                        globals.LOGGER.log_one(f"BC/{self.rb_id}: subtask_{int(sid)}_loss", sample_loss[mask].mean())

        if self.subtask_weights is None:
            return sample_loss

        weights = torch.ones_like(sample_loss)
        for sid, w in self.subtask_weights.items():
            weights[subtask_id == int(sid)] = w

        return sample_loss * weights

    def compute_sample_loss(self):
        # get the batch from the batch loader
        nbatch = self.current_batch

        # compute loss
        actor_loss, a0, timesteps = self.actor.loss(nbatch, self.rb_id)

        sample_loss = torch.mean(actor_loss, dim=(1, 2))
        sample_loss = sample_loss.squeeze()

        # stashed pre-subtask-weighting, for eval()'s explicit val_loss/task_*
        # and val_loss/subtask_* breakdown -- separate namespace from the
        # BC/{rb_id}: subtask_{sid}_loss key logged just below, which is
        # shared with the training step and gets overwritten by whichever
        # call (train or val) ran most recently
        self._last_raw_sample_loss = sample_loss.detach()

        sample_loss = self.log_and_weight_by_subtask(nbatch, sample_loss)

        return sample_loss, a0, timesteps

    def compute_loss(self):
        """
        Train for one batch.
        """
        # get the batch from the batch loader
        self.get_next_batch()
        
        actor_loss = utils.InitZeroTensorOnDevice()
        losses = self.init_losses()
        
        if self.use_bc_loss:
            # get the BC loss
            sample_loss, a0, timesteps = self.compute_sample_loss()

            self.a0 = a0

            # mean it
            actor_loss = sample_loss.mean()

            # save and log
            self.save_bc_loss(losses, actor_loss)

            # must run right after compute_sample_loss(): reads the spatial
            # softmax keypoints that call's forward() pass just cached for
            # THIS nbatch (see compute_keypoint_loss's docstring)
            losses['actor']['keypoint'] = self.compute_keypoint_loss(self.current_batch)
            losses['actor']['pellet_xyz'] = self.compute_pellet_xyz_loss(self.current_batch)

        return losses

    def eval(self):
        loss = 0.0
        action_mse_error = 0.0

        # get the actor loss
        t = self.compute_loss()
        l = t['actor']['bc']
        loss = l.cpu()

        # explicit val_loss/task_* and val_loss/subtask_* breakdown -- own
        # namespace (unlike BC/{rb_id}: subtask_{sid}_loss, which is shared
        # with the training step and gets clobbered by whichever call ran
        # most recently), using the per-sample loss compute_sample_loss just
        # stashed for this exact (val) batch
        nbatch = self.current_batch
        raw_sample_loss = getattr(self, '_last_raw_sample_loss', None)
        if nbatch is not None and raw_sample_loss is not None:
            with torch.no_grad():
                for id_key, label in (('task_id', 'task'), ('subtask_id', 'subtask')):
                    if id_key not in nbatch:
                        continue
                    ids = torch.reshape(nbatch[id_key], [-1])
                    unique_ids = torch.unique(ids)
                    if unique_ids.numel() > 1:
                        for uid in unique_ids.tolist():
                            mask = ids == uid
                            if mask.any():
                                globals.LOGGER.log_one(f"val_loss/{label}_{int(uid)}", raw_sample_loss[mask].mean())

        # get the action mse error
        if self.current_batch is not None:
            task_id = self.current_batch['task_id'] if 'task_id' in self.current_batch else None
            action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=task_id)

        return loss, action_mse_error
    
    def reset(self):
        self.batch_loader.reset()
        
    def clip_outliers(self, loss):
        """
        if a loss is insanely large, clip it so it doesn't dominate the mean loss
        """
        outlier_threshold = 100.0
        outliers = torch.abs(loss) > outlier_threshold
        
        loss[outliers] = outlier_threshold * torch.sign(loss[outliers])
        
        return loss
    
    def remove_outliers(self, loss):
        """
        if a loss is insanely large, remove it from the batch.
        
        USE CAREFULLY
        """
        if self.remove_outlier_losses:
            outlier_threshold = 100.0
            inliers = torch.abs(loss) <= outlier_threshold
            
            loss_inliers = loss[inliers]
            
            # check for final length
            if loss_inliers.shape[0] == 0:
                print("CAUTION: all losses were deemed outliers. Returning the original loss.")
                
                return loss
            else:
                return loss_inliers
            
        return loss
            
    
""" alias """
class BC(BatchLoss):
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
    
# alias - Q Val Weighted. To be used with Rollout & Distill
class QVW(BatchLoss):
    
    @torch.no_grad()
    def compute_weights(self, metric: torch.Tensor):
        """
        weighting relative to the batch.
        metric should be higher when it's better and lower when it's worse
        """
        if metric.ndim > 0:
            # https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
            A = metric.clone()
            A -= A.min()
            A /= (A.max() + 1e-5) # protection against divide by zero
            
            # A: [0, 1]. Scale to [0.5, 1]
            w = A / 2.0 + 0.5
        else:
            w = torch.ones_like(metric)
            
        # there's an unlikely case that each qval is identical ... in which case all weights will be zero. Shouldn't crash
        
        return w
        
        
    def compute_loss(self):
        """
        weight each sample loss by qval
        """
        actor_loss = utils.InitZeroTensorOnDevice()
        losses = self.init_losses()

        # get the BC loss
        if self.use_bc_loss:
            # get the sample losses
            sample_loss, _, _ = self.compute_sample_loss()
            
            # qvals, range: [-1, +1]
            # metric = self.current_batch['qval']
            
            # higher must be better, so negate the ep len
            metric = -1.0 * self.current_batch['ep_len']
            metric = metric.squeeze()
            assert(metric.shape == sample_loss.shape)
            
            # compute a weighting
            weights = self.compute_weights(metric)
            assert(weights.shape == sample_loss.shape)
            
            w_loss = sample_loss * weights
            
            loss = w_loss.mean()
            
            self.save_bc_loss(losses, loss)
            
        return losses
        
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
        losses = self.init_losses()
        
        # flags
        models_to_train: list = globals.CONFIG.models_to_train # type: ignore
        use_dql: bool = globals.CONFIG.use_dql # type: ignore
        
        train_actor = "actor" in models_to_train
        train_critic = "critic" in models_to_train
        need_dql_actor_loss = train_actor and use_dql
        
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch
        
        # protection against batches with only 1 sample
        if nbatch['action'].shape[0]  <= 1:
            return losses
        
        a0 = None
        bc_loss = utils.InitZeroTensorOnDevice()
        timesteps = None
        
        # we need the a0 and timesteps
        if self.use_bc_loss:
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
            dql_actor_loss, dql_critic_loss = self.critic.loss(nbatch, self.rb_id, a0, timesteps, use_actor_loss=self.use_bc_loss)
            
            if train_critic:
                losses['critic'] = dql_critic_loss
            
            # dql_actor_loss shouldn't be None as long as need_dql_actor_loss is True, but I had to add it for pylance
            if need_dql_actor_loss and dql_actor_loss is not None and utils.GlobalStepFreqTrigger('dql'):
                # scale by the BC loss, don't add to graph
                scale = (self.eta * bc_loss).detach()
                
                # add on the DQL actor loss
                losses['actor']['dql'] = scale * dql_actor_loss
        
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
    
# alias
class DQL(DQLBatchLoss):
    pass
    
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
                 use_gradient_weighting,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.sqval_scale = sqval_scale
        self.sqval_offset = sqval_offset
        self.use_gradient_weighting = use_gradient_weighting

    def compute_gradient_weighting(self, nbatch):
        # forward pass WITH gradients enabled
        a0 = torch.tensor(self.a0, requires_grad=True)
        qvals = self.critic.infer(nbatch, a0, self.rb_id, use_grads=True)

        # loss is the negated qvals (because we want to maximize the qvals). Actually sign doesn't matter because we're simply finding which grads make the biggest impact on loss (torch.abs below)
        L = (-1.0 * qvals).mean()

        # backprop to get dL/d_{inputs} a.k.a. the derivative of the inputs w.r.t. the loss
        L.backward()

        # extract the action gradients
        a0_grads = a0.grad

        # could normalize grads w.r.t. the input statistics (analogous to batch norm)
        if True:
            # dim is the dim over which all inputs refer to the same input ... which is gonna be the batch dimension
            dim = 0
            eps = 1e-4
            x: torch.Tensor = a0_grads.clone()

            # normalize, https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576 
            m = x.mean(dim=dim, keepdim=True)
            s = x.std(dim=dim, unbiased=False, keepdim=True)
            x -= m
            x /= (s + eps)

            a0_grads2 = x

        # else, just use the a0_grads directly (this assumes are normalization of actions has relatively similar statistics across all inputs ... which is probably not true)
        else:
            a0_grads2 = a0_grads

        # abs so then large negative gradients are highly weighted
        a0_grads3 = torch.abs(a0_grads2)

        # maps to [0, 1]. Scale by 2x for greater separation for grad values between [0, 1]
        sqvals = nn.Tanh()(a0_grads3 * 2.0)
        sqvals2 = torch.squeeze(sqvals)
        
        # scale by q-val for each sample, too. This way we don't poison our BC from rollouts by learning to imitate shit actions.
        if True:
            # no grad
            with torch.no_grad():
                # qvals are in [-inf, 1]
                sample_weights = qvals.clone()
                
                # clamp to [0, 1], basically meaning that if a qval is < 0.0 (aka will pretty much never get a reward), give 0 weighting to its sample
                sample_weights2 = torch.clamp(sample_weights, 0, 1)
                
                # scale to [0.5, 1] I guess. Seems right to do.
                sample_weights3 = sample_weights2 / 2.0 + 0.5
                
                # reshape for broadcasting ... probably wasn't necessary to do explicitly
                sample_weights4 = torch.reshape(sample_weights3, [-1] + [1] * (len(sqvals2.shape)-1))
                
                # weight the sqvals
                sqvals3 = sqvals2 * sample_weights4
            
            
        
        globals.LOGGER.log_one("cibc/sqvals/" + self.rb_id, sqvals3.mean())

        # we're done
        return sqvals3

    def compute_simple_weighting(self, nbatch):
        # eval each BC sample (no backprop) using the g.t. a0
        qvals = self.critic.infer(nbatch, self.a0, self.rb_id)

        # use sigmoid to convert the range to [0, 1]
        s = nn.Sigmoid()
        
        # scale qvals 
        qvals2 = qvals * self.sqval_scale
        
        # offset qvals
        qvals3 = qvals2 + self.sqval_offset
        
        # sigmoid the scaled qvals
        sqvals = s(qvals3)
        sqvals2 = torch.squeeze(sqvals)
        
        globals.LOGGER.log_one("cibc/sqvals/" + self.rb_id, sqvals2.mean())

        return sqvals2


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
        
        self.a0 = nbatch['action']
        
        # always need bc-loss
        loss_arr, _, timesteps = self.get_bc_losses()
        
        if utils.StepFreqTrigger(self.freqs['critic']):
            # get the DQL losses
            _, dql_critic_loss = self.critic.loss(nbatch, self.rb_id, None, timesteps)
            
            if train_critic:
                losses['critic'] = dql_critic_loss
        
        if utils.StepFreqTrigger(self.freqs['cibc']) or is_eval:
            # hack
            if is_eval:
                bc_loss = loss_arr.mean()
                losses['actor']['bc'] = bc_loss
                
                return losses
            
            
            ## Critic weighted BC
            if self.use_bc_loss:
                if self.use_gradient_weighting:
                    sqvals2 = self.compute_gradient_weighting(nbatch)
                else:
                    sqvals2 = self.compute_simple_weighting(nbatch)
                
                # could weight sqvals by timestep, but try this first

                # weight BC samples w.r.t the sigmoid qvals
                # loss_arr2 = loss_arr.mean(axis=1)
                assert(sqvals2.shape == loss_arr.shape)
                w_bc_loss = loss_arr * sqvals2
                
                assert(w_bc_loss.shape == loss_arr.shape)

                losses['actor']['bc'] = w_bc_loss.mean()
                
                globals.LOGGER.log_one("BC/" + self.rb_id + ": bc_actor_loss", w_bc_loss.mean())
        
        # we're done
        return losses
    
# alias name
class CIBC(CriticWeightedBC):
    pass
    
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
        losses = self.compute_loss()
        
        assert(losses is not None)
        assert(losses['actor'] is not None)

        loss = losses['actor']['bc'].cpu()
        
        # get the action mse error
        action_mse_error = self.actor_model.get_val_action_mse_error(self.current_batch, task_id=self.rb_id)
        
        # right now eval is hard-coded to expect two tensors on cpu
        return loss, action_mse_error
        
class ResSAC(BatchLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # save handles to nodes
        self.res_actor: ModelEmaOptim = globals.MODELS["res_actor"] #type:ignore
        
        self.critic: CriticLoss = globals.MODELS["critic"] #type:ignore
        
    def init_my_losses(self, losses):
        """
        init this class's specific losses
        """
        losses['res_actor'] = utils.InitZeroTensorOnDevice()
        
    def compute_loss(self):
        # BC loss
        losses = super().compute_loss()
        
        # init my losses
        self.init_my_losses(losses)
        
        # get the current obs
        obs = self.current_batch['obs']
        nbatch = self.current_batch
        task_id = "00" # TESTING
        
        # get the current state
        s = obs['state'] # TODO: minor optimization: copy s and set requires_grad to False
        
        # get the current action 0, ensure it's detached from the compute graph
        # assert(isinstance(self.a0, torch.Tensor))
        # a0 = self.a0.clone().detach()
        
        # try using the dataset actions instead of the predicted denoised actions
        a0 = nbatch['action']
        
        # residual action (summation is done in res-actor)
        res_actor_model = self.res_actor.get_model()
        a02 = res_actor_model(a0, global_cond=s)
        
        # get the res actor loss
        res_actor_loss, critic_loss = self.critic.loss(nbatch, task_id, a0=a02, use_actor_loss=True)
        
        # save it
        losses['res_actor'] = res_actor_loss
        losses['critic'] = critic_loss
        
        return losses
    
    def eval(self):
        # BC eval
        loss, action_mse_error = super().eval()
        
        return loss, action_mse_error

class CriticOnlyBatchLoss(BatchLoss):
    def setup(self):
        # get the rb_id
        self.rb_id = self.batch_loader.rb_id

        # my members
        self.current_batch: dict = None #type:ignore
        
        # get a handle to the critic
        self.critic = globals.MODELS["critic"] #type:ignore
        
        self.batch_loader.setup()

    def compute_loss(self):
        """
        compute loss for one batch.
        """
        # get the batch from the batch loader
        nbatch = next(self.batch_loader)
        self.current_batch = nbatch

        # get the DQL losses
        critic_loss = self.critic.loss(nbatch, self.rb_id)
        
        # remove nan's
        if torch.isnan(critic_loss).any() or torch.isinf(critic_loss).any():
            print("Warning: critic loss contains NaN's. Removing NaN samples")
            # replacing nan's with zeros won't solve the problem since the forward pass likely contains nan's. So we just have to remove those samples from the batch
            nb_nan = torch.isnan(critic_loss).sum().item()
            print("Number of NaN samples in critic loss: {}".format(nb_nan))
            
            nb_inf = torch.isinf(critic_loss).sum().item()
            print("Number of Inf samples in critic loss: {}".format(nb_inf))
            
            mask_keep = ~(torch.isnan(critic_loss) | torch.isinf(critic_loss))
            
            critic_loss = critic_loss[mask_keep]
            
        
        # remove outlier losses... use carefully
        # critic_loss2 = self.remove_outliers(critic_loss)
        critic_loss2 = critic_loss
        
        critic_loss3 = critic_loss2.mean()
        
        losses = {
            'critic': critic_loss3
        }
        
        globals.log_one_if_exists("losses/" + self.rb_id + ": critic_loss", critic_loss3)
        
        # we're done
        return losses
    
    def eval(self):
        losses = self.compute_loss()
        
        assert(losses is not None)

        loss = np.array(0.0)
        
        # get the action mse error
        action_mse_error = np.array(0.0)
        
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
        
    def setup(self):
        self.batch_loss.setup()
        
    def __len__(self):
        return len(self.batch_loss)
        
    def weight_loss(self, loss):
        # type protection
        assert(isinstance(loss, torch.Tensor))
        
        l2 = loss * self.weight
        
        return l2

    def compute_weighted_loss(self):
        losses = self.batch_loss.compute_loss()
        
        # losses can now be a nested dict
        # hack
        wloss = {}
        
        if 'actor' in losses:
            wloss['actor'] = dict_apply(losses['actor'], lambda x: self.weight * x)
        
        if 'critic' in losses:
            wloss['critic'] = self.weight_loss(losses['critic'])
        
        if 'res_actor' in losses:
            wloss['res_actor'] = self.weight_loss(losses['res_actor'])
        
        # apply the weighting to each loss
        # wloss = dict_apply(losses, lambda x: self.weight * x)
        return wloss
    
    # TODO: rename eval to validate
    def compute_weighted_eval(self):
        # this batch_loss should already be in val mode
        wevals = self.weight * np.array(self.batch_loss.eval())
        return wevals
            
    def reset(self):
        self.batch_loss.reset()