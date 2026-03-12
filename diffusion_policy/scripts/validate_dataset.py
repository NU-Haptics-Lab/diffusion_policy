

"""
load replay buffer, plot rewards
"""



from diffusion_policy.common.replay_buffer import ReplayBuffer

import numpy as np
import scipy.stats as stats

path = "/media/dexnex_ssd/data/sandbox/online_rl_replay_buffer.zarr"

rb = ReplayBuffer.create_from_path(path, mode="r")

data = rb.data
rewards = data["reward"][:]
rewards = np.array(rewards)

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()

print("min, max, mean reward: ", rewards.min(), rewards.max(), rewards.mean())

# for all keys, print the min max mean, var
for key in data.keys():
    values = data[key][:]
    values = np.array(values)
    print(f"{key}: min {values.min()}, max {values.max()}, mean {values.mean()}, var {values.var()}")
    
    # plot
    plt.figure()
    plt.plot(values)
    plt.title(key)
    plt.show()