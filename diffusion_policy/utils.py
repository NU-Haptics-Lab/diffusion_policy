import diffusion_policy.globals as globals

def EveryEpoch(every: int):
    yes = (globals.EPOCH % every) == 0
    return yes