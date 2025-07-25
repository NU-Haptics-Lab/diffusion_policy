if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.dexnex_meta_dataset import MetaDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion_ql import QL_Training

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    # def __init__(self, cfg: OmegaConf, output_dir=None):
    def __init__(self, 
                 cfg: OmegaConf, 
                 output_dir = None
                #  output_dir = "/home/omnid/dexnex/libraries/diffusion_policy/outputs/2025-04-22/16-58-25"
                 ):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        print("Total nb params: %e" % sum(p.numel() for p in self.model.parameters()))

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        config = cfg # I just prefer it spelled out

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        cls = hydra.utils.get_class(cfg.task.dataset._target_)
        meta_dataset = cls(cfg.task.dataset.num_train_batches, cfg)
        
        # configure validation dataset
        meta_val_dataset = meta_dataset.get_validation_dataset(cfg.task.dataset.num_val_batches)
        
        # need to save as a class member so that the normalizers get pickled. Assume the first dataset listed is the task normalizer, used in evaluation.
        normalizer = meta_dataset.get_normalizers()[0]
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(meta_dataset) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        
        # configure diffusion-ql (if valid)
        if config.training.use_qloss:
            # make the critic network
            critic = <>
            
            # make the trainer
            self.ql_trainer = QL_Training(critic)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            
        # # TEST
        nbatch = next(meta_dataset)
        policy = self.ema_model
        policy.eval()
        raw_loss = self.model.compute_loss(nbatch)
        nresult = policy.predict_action(nbatch['nobs'])
        result = meta_dataset.unnormalize(nresult)
        policy.train()

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(meta_dataset, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        # device transfer -- done in meta_dataset
                        if train_sampling_batch is None:
                            train_sampling_batch = nbatch

                        # compute loss
                        raw_loss = self.model.compute_loss(nbatch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        # diffusion-ql, q-learning
                        if config.training.use_qloss:
                            qloss, qmetrics = self.ql_trainer(nbatch['nobs'])
                            loss += qloss
                            
                            # add metrics to the wandb step logger
                            step_log.update(qmetrics)
                        
                        # calculate gradients
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(meta_dataset)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_action_mse_errors = list()
                        with tqdm.tqdm(meta_val_dataset, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, nbatch in enumerate(tepoch):
                                # transfer to device done in meta_dataset
                                loss = self.model.compute_loss(nbatch)
                                val_losses.append(loss)
                                
                                # action mse
                                nobs_dict = nbatch['nobs']
                                gt_naction = nbatch['naction']
                                
                                result = policy.predict_action(nobs_dict)
                                pred_naction = result['naction_pred']
                                mse = torch.nn.functional.mse_loss(pred_naction, gt_naction)
                                action_mse_error = mse.item()
                                val_action_mse_errors.append(action_mse_error)
                                
                                del nobs_dict
                                del gt_naction
                                del result
                                del pred_naction
                                del mse
                                
                                
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                        if len(val_action_mse_errors) > 0:
                            val_action_mse_error = torch.mean(torch.tensor(val_action_mse_errors)).item()
                            # log epoch average validation loss
                            step_log['val_action_mse_error'] = val_action_mse_error

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        # device transfer now done in meta_dataset
                        nbatch = train_sampling_batch
                        nobs_dict = nbatch['nobs']
                        gt_naction = nbatch['naction']
                        
                        nresult = policy.predict_action(nobs_dict)
                        naction_pred = nresult['naction_pred']
                        
                        # compare normalized action_pred to normalized ground truth action
                        mse = torch.nn.functional.mse_loss(naction_pred, gt_naction)
                        step_log['train_action_mse_error'] = mse.item()
                        del nbatch
                        del nobs_dict
                        del gt_naction
                        del nresult
                        del naction_pred
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
