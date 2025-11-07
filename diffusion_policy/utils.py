import torch
import copy
from hydra.core.hydra_config import HydraConfig

import diffusion_policy.globals as globals
from diffusion_policy.common import pytorch_util

def EveryEpoch(every: int):
    yes = (globals.EPOCH % every) == 0
    return yes

def StepFreqTrigger(every: int):
    yes = (globals.STEP % every) == 0
    return yes

def GlobalStepFreqTrigger(key):
    if key not in globals.CONFIG.step_freqs:
        return False
    else:
        yes = (globals.STEP % globals.CONFIG.step_freqs[key]) == 0
        return yes

def InitZeroTensorOnDevice():
    t = torch.tensor(0.0, device=globals.CONFIG.device, dtype=torch.float32) #type: ignore
    return t

def get_output_dir(_output_dir=None):
    output_dir = _output_dir
    if output_dir is None:
        output_dir = HydraConfig.get().runtime.output_dir
    return output_dir

def print_nb_params(model, descriptor):
    print(descriptor + ": %e" % sum(p.numel() for p in model.parameters()))


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
    
    
def norm_gpu_action(action_cpu):
    action_cpu_dd = {"action": action_cpu}
    
    naction_dd = globals.DEFAULT_BATCH_LOADER.transfer_and_norm(action_cpu_dd)
    
    naction_gpu = naction_dd['action']
    
    return naction_gpu
    
def norm_gpu_critic(batch, a):    
    # put on gpu and norm
    dd_torch = pytorch_util.dict_to_torch(batch)
    
    nbatch = globals.DEFAULT_BATCH_LOADER.transfer_and_norm(dd_torch)
    
    # put actionss on gpu and norm
    na = norm_gpu_action(torch.tensor(a))
    
    return nbatch, na

def add_batch_dim(batch: dict):
    def f(x):
        x = torch.unsqueeze(x, dim=0)
        return x
    return pytorch_util.dict_apply(batch, f)

def standardize_shape(x):
    """
    make at-least 2d
    """
    x = torch.tensor(x)
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, dim=0)
        
    return x

def compute_jerk(s, a, dt=0.1):
    """
    s - initial state, 1d
    a - next state, 1d
    """
    s = standardize_shape(s)
    a = standardize_shape(a)
    
    nb_joints = a.shape[-1]
    
    s0 = s[:, :nb_joints]
    
    s = torch.vstack([s0, a])
            
    # get the delta state
    r_shifted_s = s[1:, :]
    l_shifted_s = s[:-1, :]
    
    # units: normalized joint states
    ds = torch.abs(r_shifted_s - l_shifted_s)
    
    # units: normalized joint-state / s^3
    ds2 = ds / dt**3
    
    # sum along the traj dim
    sum1 = torch.sum(ds2, dim = 1)
    
    return sum1