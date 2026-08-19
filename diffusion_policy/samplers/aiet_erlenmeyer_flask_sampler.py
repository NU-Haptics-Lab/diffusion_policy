import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.common.sarsa_sampler import EpisodeSampler


class AIETErlenmeyerFlaskSampler(EpisodeSampler):
    """
    Get a sample from an episode for the erlenmeyer flask BC policy.
    """

    # keys that have both a full 14x14 patch grid and a downsampled 7x7
    # variant in the zarr (see gen_dataset.py) -- which one backs the
    # obs_keys_to_load name is picked by config.patch_grid_size (14 or 7,
    # default 14), via resolve_rb_key below. shape_meta's declared shape for
    # these keys must match (see aiet_erlenmeyer_flask_3.yaml's
    # ${patch_grid_size} interpolation) since that's what sizes the model's
    # patch embedding/spatial-softmax layers.
    PATCH_GRID_KEYS = ["wrist_cam_patch_features", "overhead_roi_patch_features"]

    # obs keys that are precomputed derived signals rather than a literal-name
    # rb array: fixed-tau (no learning) spatial-softmax keypoint descriptors
    # over the full 14x14 patch grid -- see
    # add_spatial_softmax_keypoints.py -- computed once offline instead of
    # from raw patch grids at train time (much smaller to load: [T, 384, 2]
    # vs [T, 14, 14, 384]). Used in place of the corresponding
    # PATCH_GRID_KEYS entry when a config's obs_keys_to_load asks for
    # "*_keypoints" instead of the raw "*_patch_features" key -- see
    # aiet_erlenmeyer_flask_7.yaml. Independent of patch_grid_size (the
    # offline script always reduces the native 14x14 grid, never the
    # downsampled 7x7 one).
    KEYPOINT_RB_KEY_MAP = {
        "wrist_cam_keypoints": "wrist_cam_patch_features_spatial_softmax_xy",
        "overhead_roi_keypoints": "overhead_roi_patch_features_spatial_softmax_xy",
    }

    def get_action_trajectory(self, ep_idx, key):
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx  # type: ignore
        return self.indices.get_sequence_by_train_indices_and_key(indices, key)

    def get_obs_sample(self, ep_idx):
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            obs_sample[obs_key] = self.get_key_sample(self.resolve_rb_key(obs_key), ep_idx)
        return obs_sample

    def resolve_rb_key(self, obs_key):
        if obs_key in self.KEYPOINT_RB_KEY_MAP:
            return self.KEYPOINT_RB_KEY_MAP[obs_key]
        if obs_key in self.PATCH_GRID_KEYS and getattr(globals.CONFIG, "patch_grid_size", 14) == 7:  # type: ignore
            return obs_key + "_7x7"
        return obs_key

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

        # this episode's identity (rb_episode_end -- unique per episode within
        # a given rb_id, used as the key in DatasetSampler.ep_samplers too).
        # Lets a diagnostic look up e.g. "this episode's final palm pose" as a
        # proxy for the true alignment target, without needing a ground-truth
        # pellet-location label (see DiffusionModel's alignment-direction
        # check). Plain identity-normalized scalar, same treatment as
        # task_id/subtask_id/data_source.
        sample["ep_id"] = np.array([self.rb_episode_end], dtype=np.float32)

        return sample


class AIETErlenmeyerFlask3RealSampler(AIETErlenmeyerFlaskSampler):
    """
    Real-robot half of aiet_erlenmeyer_flask_3's cotraining with the sim
    alignment dataset. The real dataset has no pellet-location signal, so
    zero-fill it (shape-correct placeholder only -- the actual "don't use
    this" masking happens as a learned token zeroing in
    DexNexTransformerForDiffusion, driven by the data_source tag below, not
    by this raw value being zero).
    """

    SIM_ONLY_KEYS = ["target_pellet_location"]

    def get_obs_sample(self, ep_idx):
        # don't blindly call super().get_obs_sample() -- it iterates every key
        # in obs_keys_to_load, including sim-only keys not present in the real
        # replay buffer, which would KeyError before we get a chance to zero-fill
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            rb_key = self.resolve_rb_key(obs_key)
            if rb_key is None:
                shape = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
                obs_sample[obs_key] = np.zeros((1, *shape), dtype=np.float32)
            else:
                obs_sample[obs_key] = self.get_key_sample(rb_key, ep_idx)
        return obs_sample

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)
        sample["data_source"] = np.array([0], dtype=np.float32)  # 0 = real
        return sample

    def resolve_rb_key(self, obs_key):
        if obs_key in self.SIM_ONLY_KEYS:
            return None  # zero-filled, see get_obs_sample above
        return super().resolve_rb_key(obs_key)  # applies patch_grid_size substitution

    def get_constant_sample_fields(self):
        return {"data_source": 0.0}  # always 0 = real, see get_sample above


class AIETErlenmeyerFlask3SimSampler(EpisodeSampler):
    """
    Sim-alignment half of aiet_erlenmeyer_flask_3's cotraining with the real
    erlenmeyer-flask dataset. The sim zarr only has joint_positions and
    target_pellet_location -- joint_positions is the same joint convention as
    the real task's joint_states/joint_commands, so it's aliased to those
    shared keys rather than treated as a separate one. Every real-only key
    (vision, biotac) is zero-filled (shape-correct placeholder only -- actual
    masking happens via data_source-driven token zeroing in the model).
    """

    REAL_ONLY_KEYS = [
        "biotac_lh", "wrist_cam_features", "overhead_roi_features",
        "wrist_cam_patch_features", "overhead_roi_patch_features",
    ]

    def get_action_trajectory(self, ep_idx):
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx  # type: ignore
        # the sim zarr only has joint_positions -- use it directly regardless
        # of the cotrain-wide action_key (joint_commands), which doesn't exist here
        return self.indices.get_sequence_by_train_indices_and_key(indices, "joint_positions")

    def get_obs_sample(self, ep_idx):
        obs_sample = {}
        # shared key: sim's joint_positions is the same joint convention as
        # the real task's joint_states
        obs_sample["joint_states"] = self.get_key_sample("joint_positions", ep_idx)
        obs_sample["target_pellet_location"] = self.get_key_sample("target_pellet_location", ep_idx)

        for key in self.REAL_ONLY_KEYS:
            shape = globals.CONFIG.shape_meta[key].shape  # type: ignore
            obs_sample[key] = np.zeros((1, *shape), dtype=np.float32)

        return obs_sample

    def get_sample(self, ep_idx):
        assert ep_idx >= 0
        assert ep_idx <= len(self) - 1

        sample = {}
        sample["obs"] = self.get_obs_sample(ep_idx)
        sample["action"] = self.get_action_trajectory(ep_idx)
        # no real cotraining task/subtask concept for the sim data -- constant fallbacks
        sample["task_id"] = np.array([24], dtype=np.float32)
        sample["subtask_id"] = np.array([0], dtype=np.float32)
        sample["data_source"] = np.array([1], dtype=np.float32)  # 1 = sim

        return sample

    def resolve_rb_key(self, obs_key):
        if obs_key == "joint_states":
            return "joint_positions"  # shared key, see get_obs_sample above
        if obs_key in self.REAL_ONLY_KEYS:
            return None  # zero-filled, see get_obs_sample above
        return obs_key  # target_pellet_location

    def resolve_action_rb_key(self, action_key):
        # the sim zarr only has joint_positions -- ignore the cotrain-wide
        # action_key (joint_commands), which doesn't exist here (see
        # get_action_trajectory above)
        return "joint_positions"

    def get_constant_sample_fields(self):
        # always hardcoded in get_sample above, regardless of rb contents
        # (the sim rb has no task_id/subtask_id/data_source arrays at all)
        return {"task_id": 24.0, "subtask_id": 0.0, "data_source": 1.0}
