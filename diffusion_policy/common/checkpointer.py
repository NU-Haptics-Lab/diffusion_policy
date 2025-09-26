from typing import Optional, Dict
import os
import pathlib
import torch
import dill
import threading

import diffusion_policy.globals as globals
from diffusion_policy.utils import EveryEpoch
from diffusion_policy.utils import _copy_to_cpu
from hydra.core.hydra_config import HydraConfig

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
            resume = False
        ):
        assert mode in ['max', 'min']
        assert k >= 0
        
        # old
        if self.resume:
            self.output_dir = output_dir
            
        # new 
        else:
            self.output_dir = HydraConfig.get().runtime.output_dir

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

    def save(self):
        if EveryEpoch(self.checkpoint_every):
            # checkpointing
            if self.save_last_ckpt:
                self.save_checkpoint()
            if self.save_last_snapshot:
                # NOT IMPLEMENTED / TESTED
                # self.save_snapshot()
                pass

            # sanitize metric names
            metric_dict = dict()
            for key, value in step_log.items():
                new_key = key.replace('/', '_')
                metric_dict[new_key] = value
            
            # We can't copy the last checkpoint here
            # since save_checkpoint uses threads.
            # therefore at this point the file might have been empty!
            topk_ckpt_path = self.get_ckpt_path(metric_dict)

            if topk_ckpt_path is not None:
                self.save_checkpoint(path=topk_ckpt_path)

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.save_dir).joinpath(f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
            
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload 
    
    def load(self):
        if self.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint: {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
    
    # NOT IMPLEMENTED / TESTED
    # def save_snapshot(self, tag='latest'):
    #     """
    #     Quick loading and saving for reserach, saves full state of the workspace.

    #     However, loading a snapshot assumes the code stays exactly the same.
    #     Use save_checkpoint for long-term storage.
    #     """
    #     path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
    #     path.parent.mkdir(parents=False, exist_ok=True)
    #     torch.save(self, path.open('wb'), pickle_module=dill)
    #     return str(path.absolute())