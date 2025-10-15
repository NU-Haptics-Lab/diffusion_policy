from typing import Optional, Dict
import os
import pathlib
import torch
import dill
import threading
from torch import nn

import diffusion_policy.globals as globals
from diffusion_policy.utils import EveryEpoch
from diffusion_policy.utils import _copy_to_cpu
from hydra.core.hydra_config import HydraConfig

# class Checkpointer:
#     def __init__(self,
#                  output_dir = "",
#                  rel_save_dir = "",
#                  resume = False
#                  ):
        
#         # old
#         if resume:
#             self.output_dir = output_dir
            
#         # new 
#         else:
#             self.output_dir = HydraConfig.get().runtime.output_dir

#         self.rel_save_dir = rel_save_dir
#         self.save_dir = os.path.join(self.output_dir, rel_save_dir)
        
#     def get_full_path(self,
#             path=None, 
#             tag='latest'):
        
#         # default path
#         if path is None:
#             path = pathlib.Path(self.save_dir).joinpath(f'{tag}.ckpt')
#         else:
#             path = pathlib.Path(path)
            
#         return path
        
#     def make_dir(self):
            
#         # ensure directory exists, make it if it doesn't
#         path.parent.mkdir(parents=False, exist_ok=True)
    

class TopKCheckpointManager:
    def __init__(self,
            monitor_key: str,
            rel_save_dir = "checkpoints",
            output_dir = "",
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt',
            checkpoint_every = 1,
            save_last_ckpt = False,
            save_last_snapshot = False,
            resume = False,
            resume_tag = "latest",
            use_current_directory = True
        ):
        assert mode in ['max', 'min']
        assert k >= 0
        
        # old
        if resume:
            self.output_dir = output_dir
            
        # new 
        elif use_current_directory:
            self.output_dir = HydraConfig.get().runtime.output_dir
            
        else:
            self.output_dir = output_dir

        self.rel_save_dir = rel_save_dir
        self.save_dir = os.path.join(self.output_dir, rel_save_dir)
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        self.checkpoint_every = checkpoint_every
        self.save_last_ckpt = save_last_ckpt
        self.save_last_snapshot = save_last_snapshot
        self.resume = resume
        self.resume_tag = resume_tag
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
        
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.save_dir).joinpath(f'{tag}.ckpt')
    
    def force_save(self):
        print("Saving...")
        
        # save the current state as `latest`
        # checkpointing
        if self.save_last_ckpt:
            self.save_checkpoint()

        # snapshotting
        if self.save_last_snapshot:
            # NOT IMPLEMENTED / TESTED
            # self.save_snapshot()
            pass

        # sanitize metric names
        metric_dict = dict()
        for key, value in globals.LOGGER.data.items():
            new_key = key.replace('/', '_')
            metric_dict[new_key] = value
        
        # Now, save a top-k checkpoint
        # We can't copy the last checkpoint here
        # since save_checkpoint uses threads.
        # therefore at this point the file might have been empty!
        topk_ckpt_path = self.get_ckpt_path(metric_dict)

        if topk_ckpt_path is not None:
            self.save_checkpoint(path=topk_ckpt_path)
            
        print("Saved.")
    
    def save(self):
        if EveryEpoch(self.checkpoint_every):
            self.force_save()
            
    def get_state_dicts(self, dd):
        """
        Get non-nn.Module state dicts, like optimizers and lr-schedulers
        """
        # TODO
        
        for key, value in self.__dict__.items():
            # modules are captured elsewhere
            if not isinstance(value, nn.Module):
                # make sure it has a state_dict and load_state_dict
                if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                    dd[key] = value.state_dict()
                
            # recurse
            self.get_state_dicts(value)
        

    def save_checkpoint(self, 
            path=None, 
            tag='latest', 
            ):
        """
        Simplify: no script uses exclude_keys, so I'm going to exclude it from the function
        only include_keys used are ['global_step', 'epoch'], so just add them explicitly
        """
        
        # default path
        if path is None:
            path = pathlib.Path(self.save_dir).joinpath(f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
            
        # ensure directory exists, make it if it doesn't (including parent dir's)
        path.parent.mkdir(parents=True, exist_ok=True)

        # saving payload
        payload = {
            'cfg': globals.CONFIG,
            'step': globals.STEP,
            'epoch': globals.EPOCH,
            'models_state_dict': globals.MODELS.state_dict(),
            'non_models_state_dicts': {}
        }
        
        # self.get_state_dicts(payload['non_models_state_dicts'])
        
        torch.save(payload, path.open('wb'), pickle_module=dill)

        # return the checkpoint path
        return str(path.absolute())
    
    def load_globals(self, payload):
        # hard-coded for now
        globals.STEP = payload['step']
        globals.EPOCH = payload['epoch']
    
    def load_payload(self, payload, **kwargs):
        # modules
        if True: # "critic" in globals.CONFIG.load:
            # hack for backwards compat.
            # TODO: fix
            globals.MODELS.load_state_dict(payload['models_state_dict'], strict=False)
        
        if "globals" in globals.CONFIG.load:
            self.load_globals(payload)


    def load_checkpoint(self, path=None, tag='latest',
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

        # load from disk
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)

        # load into classes
        self.load_payload(payload)
        return payload 
    
    def load(self):
        # hacky, works for now
        if self.resume: # "critic" in globals.CONFIG.load:
            lastest_ckpt_path = self.get_checkpoint_path(self.resume_tag)
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint: {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                
            else:
                print("Checkpointer::load: lastest_ckpt_path wasn't a file.")
                raise
            
# class GlobalCheckpointer(TopKCheckpointManager):
#     """
#     Saves global params
#     """

#     def save_checkpoint(self, 
#             path=None, 
#             tag='latest', 
#             ):
#         """
#         Simplify: no script uses exclude_keys, so I'm going to exclude it from the function
#         only include_keys used are ['global_step', 'epoch'], so just add them explicitly
#         """

#         # saving payload
#         payload = {
#             'cfg': globals.CONFIG,
#             'step': globals.STEP,
#             'epoch': globals.EPOCH,
#         }
        
#         # self.get_state_dicts(payload['non_models_state_dicts'])
        
#         torch.save(payload, path.open('wb'), pickle_module=dill)

#         # return the checkpoint path
#         return str(path.absolute())
    
#     def load_payload(self, payload, **kwargs):
#         # hard-coded for now
#         globals.STEP = payload['step']
#         globals.EPOCH = payload['epoch']