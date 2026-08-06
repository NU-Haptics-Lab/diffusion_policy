import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.common.sarsa_sampler import EpisodeSampler


class AIETErlenmeyerFlaskSampler(EpisodeSampler):
    """
    Get a sample from an episode for the erlenmeyer flask BC policy.
    """

    def get_action_trajectory(self, ep_idx, key):
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx  # type: ignore
        return self.indices.get_sequence_by_train_indices_and_key(indices, key)

    def get_obs_sample(self, ep_idx):
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            obs_sample[obs_key] = self.get_key_sample(obs_key, ep_idx)
        return obs_sample

    def get_sample(self, ep_idx):
        """
        note: nns['action'] is hard-coded in diffusion_model and must be used
        """
        
        assert ep_idx >= 0
        assert ep_idx <= len(self) - 1

        sample = {}

        sample["obs"] = self.get_obs_sample(ep_idx)

        act_key = globals.CONFIG.action_key  # type: ignore
        sample["action"] = self.get_action_trajectory(ep_idx, act_key)
        
        # task id. 24 if the sample doesn't have a task id.
        try:
            sample["task_id"] = self.get_key_sample("task_id", ep_idx)
        except KeyError:
            sample["task_id"] = np.array([24], dtype=np.float32)

        # subtask id (e.g. 0 = normal, 1 = alignment-focused). 0 if the sample doesn't have one.
        try:
            sample["subtask_id"] = self.get_key_sample("subtask_id", ep_idx)
        except KeyError:
            sample["subtask_id"] = np.array([0], dtype=np.float32)

        return sample
