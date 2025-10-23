import torch
import copy
from hydra.core.hydra_config import HydraConfig

import diffusion_policy.globals as globals

def EveryEpoch(every: int):
    yes = (globals.EPOCH % every) == 0
    return yes

def StepFreqTrigger(every: int):
    yes = (globals.STEP % every) == 0
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