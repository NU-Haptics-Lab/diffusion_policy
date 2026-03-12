import torch
import copy
from hydra.core.hydra_config import HydraConfig

import diffusion_policy.globals as globals
from diffusion_policy.common import pytorch_util
import numpy as np

from omegaconf import (
    OmegaConf,
    DictConfig,
)

import time

def tic():
    return time.time()

def toc(tic):
    return time.time() - tic

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
    if x.ndim == 0:
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
    elif x.ndim == 1:
        x = torch.unsqueeze(x, dim=0)
        
    return x

def compute_jerk_waypoints(s, a, dt=0.1):
    """
    s - initial state, 1d
    a - next state, 1d
    """
    s = standardize_shape(s)
    a = standardize_shape(a)
    
    nb_joints = a.shape[-1]
    
    s0 = s[:, :nb_joints]
    
    # s = torch.vstack([s0, a])
            
    # # get the delta state
    # r_shifted_s = s[1:, :]
    # l_shifted_s = s[:-1, :]
    
    # units: normalized joint states
    # ds = torch.abs(r_shifted_s - l_shifted_s)
    ds = torch.abs(s0 - a)
    
    # units: normalized joint-state / s^3
    ds2 = ds / dt**3
    
    # sum along waypoint dim
    sum1 = torch.sum(ds2, dim = 1)
    
    return sum1

@torch.no_grad()
def compute_jerk(s, a, dt=0.1):
    """
    s - initial state, 1d
    a - next state, 1d
    """
    sum1 = compute_jerk_waypoints(s, a, dt)
    
    # mean across all waypoints
    mean1 = torch.mean(sum1)
    
    return mean1

@torch.no_grad()
def compute_energy(s):
    """
    s is [batch-dim, waypoints, state-dim]
    """
    s = torch.tensor(s)
    
    # get the delta state
    r_shifted_s = s[:, 1:, :]
    l_shifted_s = s[:, :-1, :]
        
    ds = torch.abs(r_shifted_s - l_shifted_s)
    
    # power 2 each element
    ds2 = torch.pow(ds, 2.0)
    
    # gofa hack to account for greater link mass
    ds2[:, :, 0:6] = ds2[:, :, 0:6] * 5.0
    
    # much lower energy for moving fingers
    ds2[:, :, 6:] = ds2[:, :, 6:] * 0.1
    
    # sum along the traj dim
    sum1 = torch.sum(ds2, dim = 1)
    
    # sum along state dim
    sum2 = torch.sum(sum1, dim = 1)
    
    # done, shape should be [batch size, 1]
    return sum2

def compute_stats(arr):
    arr2 = np.array(arr)
    
    mean = arr2.mean()
    std = arr2.std()
    min = arr2.min()
    max = arr2.max()
    
    return mean, std, min, max

def make_qvals(rewards, discount = 0.975):
    qvals = []
    qval = 0.0
        
    
    for reward in reversed(rewards):
        qval = reward + discount * qval
        
        qvals.append(qval)
        
    qvals2 = np.array(qvals)
    
    # reverse
    qvals3 = np.flip(qvals2)
    
    assert(not np.any(np.isnan(qvals3)))
    return qvals3

def add_qvals(dd):
    rewards = dd['reward']
    qvals = make_qvals(rewards)
    
    dd['qval'] = np.array(qvals, dtype=np.float32)
    pass

def make_data_dict(episode):
    data_dict = dict()
    for key in episode[0].keys():
        data_dict[key] = np.stack([x[key] for x in episode])
        
    return data_dict

# gemini generated code
import torch

def unflatten_state_dict(state_dict):
    """
    Converts a flat PyTorch state_dict into a nested dictionary.
    """
    nested_dict = {}
    for key, value in state_dict.items():
        # Split the key by the delimiter (typically '.')
        keys = key.split('.')
        current_dict = nested_dict
        # Traverse the keys, creating nested dictionaries as needed
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                # If it's the last key, assign the value (tensor)
                current_dict[k] = value
            else:
                # If not the last key, continue nesting or create a new dict
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
    return nested_dict

def flatten_nested_dict(nested_dict, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single-level dictionary with 
    dot-separated keys, reversing the unflatten_state_dict function.
    """
    items = []
    for k, v in nested_dict.items():
        # Construct the new key by combining parent key and current key
        new_key = parent_key + sep + k if parent_key else k
        
        # Check if the value is a dictionary (and not a tensor, to stop recursion at the parameters)
        if isinstance(v, dict):
            # Recurse if it's a dictionary
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            # Append the key-value pair if it's a leaf node (e.g., a tensor)
            items.append((new_key, v))
            
    return dict(items)

def drake_pos_to_pose(pos):
    """
    convert a position [3] to a pose [7] wxyz, xyz
    """
    pos2 = np.array(pos).flatten()
    assert(pos2.shape == (3,))
    
    # identity quat
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    pose = np.concatenate([quat, pos2], axis=0)
    
    return pose

def drake_compute_rel_pose(src_pose, dst_pose):
    """
    compute the relative pose from src to dst
    both poses are [7] wxyz, xyz
    
    use tf2
    """
    # only allowed to have a single src pose
    src_pose2 = np.array(src_pose).flatten()
    assert(src_pose2.shape == (7,))
    
    # can have multiple dst poses
    dst_pose2 = np.array(dst_pose).reshape(-1, 7)
    
    from pydrake.all import (
        RigidTransform,
        Quaternion
    )
    

    def numpy_to_pose(array: np.ndarray) -> RigidTransform:
        q = array[:4]
        p = array[4:7]
        
        quat = Quaternion(q[0], q[1], q[2], q[3])
        
        pose = RigidTransform(quat, p) #type:ignore
        
        return pose
    
    

    def pose_to_numpy(pose: RigidTransform) -> np.ndarray:
        q = pose.rotation().ToQuaternion()
        p = pose.translation()
        
        # Pack row
        row = np.concatenate([q.wxyz(), p])
        
        return row
    
    src_tf = numpy_to_pose(src_pose2)
    
    rel_poses = []
    for dst_pose in dst_pose2:
        dst_tf = numpy_to_pose(dst_pose)
    
        rel_tf = src_tf.inverse() @ dst_tf
    
        rel_pose = pose_to_numpy(rel_tf)
        rel_poses.append(rel_pose)
        
    # stack rel poses
    rel_pose = np.stack(rel_poses, axis=0)
    
    return rel_pose

def drake_compute_yaw_only_pose_from_pose(pose):
    """
    given a pose [7] wxyz, xyz
    compute a new pose that only has yaw rotation
    
    output: [7] wxyz, xyz
    """
    pose2 = np.array(pose).flatten()
    assert(pose2.shape == (7,))
    
    from pydrake.all import (
        RigidTransform,
        Quaternion,
        RollPitchYaw
    )
    
    q = pose2[0:4]
    p = pose2[4:7]
    
    quat = Quaternion(q[0], q[1], q[2], q[3])
        
    rpy = RollPitchYaw(quat)
    
    yaw = rpy.yaw_angle()
    
    yaw_rpy = RollPitchYaw(0.0, 0.0, yaw)
    
    yaw_quat = yaw_rpy.ToQuaternion()
        
    yaw_pose = np.concatenate([yaw_quat.wxyz(), p], axis=0)
    
    return yaw_pose

def drake_extract_pitch_roll_from_pose(pose):
    """
    given a pose [7] wxyz, xyz
    extract pitch and roll angles in radians
    
    output: [2] pitch, roll
    """
    pose2 = np.array(pose).flatten()
    assert(pose2.shape == (7,))
    
    from pydrake.all import (
        RigidTransform,
        Quaternion,
        RollPitchYaw
    )
    
    q = pose2[0:4]
    
    quat = Quaternion(q[0], q[1], q[2], q[3])
    
    rpy = RollPitchYaw(quat)
    
    pitch = rpy.pitch_angle()
    roll = rpy.roll_angle()
        
    pitch_roll = np.array([pitch, roll], dtype=np.float32)
    
    return pitch_roll

def drake_compute_rel_pos(src_pose, dst_pose):
    rel_pose = drake_compute_rel_pose(src_pose, dst_pose)
    
    rel_pos = rel_pose[:, 4:7]
    return rel_pos

def drake_compute_abs_pose_from_rel(src_pose, rel_pose):
    """
    compute the absolute pose from src and rel pose
    both poses are [7] wxyz, xyz
    
    use tf2
    """
    # only allowed to have a single src pose
    src_pose2 = np.array(src_pose).flatten()
    assert(src_pose2.shape == (7,))
    
    # can have multiple dst poses
    rel_pose2 = np.array(rel_pose).reshape(-1, 7)
    
    from pydrake.all import (
        RigidTransform,
        Quaternion
    )
    

    def numpy_to_pose(array: np.ndarray) -> RigidTransform:
        q = array[:4]
        p = array[4:7]
        
        quat = Quaternion(q[0], q[1], q[2], q[3])
        
        pose = RigidTransform(quat, p) #type:ignore
        
        return pose
    
    

    def pose_to_numpy(pose: RigidTransform) -> np.ndarray:
        q = pose.rotation().ToQuaternion()
        p = pose.translation()
        
        # Pack row
        row = np.concatenate([q.wxyz(), p])
        
        return row
    
    src_tf = numpy_to_pose(src_pose2)
    
    abs_poses = []
    for rel_pose in rel_pose2:
        rel_tf = numpy_to_pose(rel_pose)
    
        abs_tf = src_tf @ rel_tf
    
        abs_pose = pose_to_numpy(abs_tf)
        abs_poses.append(abs_pose)
        
    # stack abs poses
    abs_pose = np.stack(abs_poses, axis=0)
    
    return abs_pose

def normalize_quaternions_inplace(quats):
    assert(quats.ndim == 2)
    
    
    for quat in quats:
        norm = np.linalg.norm(quat)
        quat /= norm
        
def drake_compute_rel_fk(pose_state, fk, get_rel_pose_fcn):
    # reshape fk into [N, 3] for fingers, xyz
    fk = fk.reshape(-1, 3)
    
    # expand each entry to start with wxyz = 1, 0, 0, 0
    fk_poses = np.concatenate([
        np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (fk.shape[0], 1)),
        fk
    ], axis=-1) # [N, 7]
    
    # compute rel fk
    rel_fk = get_rel_pose_fcn(
        src = pose_state,
        dst = fk_poses
    )
    
    # extract just the xyz's
    rel_fk = rel_fk[:, 4:7]
    
    # reshape to a long 2d vector
    rel_fk = rel_fk.reshape(1, -1)
    
    return rel_fk

def get_rel_yaw_pose(src, dst):
    """
    get rel pose but ONLY considering yaw rotation of src
    """
    src_yaw_only_pose = drake_compute_yaw_only_pose_from_pose(src)
    
    rel_pose = drake_compute_rel_pose(
        src_pose=src_yaw_only_pose,
        dst_pose=dst
    )
    
    return rel_pose
