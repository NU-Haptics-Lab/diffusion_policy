
from hydra.core.hydra_config import HydraConfig

import diffusion_policy.globals as globals

def EveryEpoch(every: int):
    yes = (globals.EPOCH % every) == 0
    return yes

def get_output_dir(_output_dir=None):
    output_dir = _output_dir
    if output_dir is None:
        output_dir = HydraConfig.get().runtime.output_dir
    return output_dir