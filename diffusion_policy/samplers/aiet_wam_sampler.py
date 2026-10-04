import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.samplers.aiet_alignment_sim_3_sampler import AIETAlignmentSim3Sampler


class AIETWamSampler(AIETAlignmentSim3Sampler):
    """
    World-model sampler for aiet_wam_1.yaml. Unlike every aiet_erlenmeyer_flask_*/
    aiet_alignment_* sampler, the diffused target here is NOT an action --
    it's a trajectory (one row per action_rel_indices offset) of every one
    of WAM_OBS_KEYS' own FUTURE values (each flattened, concatenated in
    WAM_OBS_KEYS order), with one extra scalar appended per row:
    task-completion fraction (see _completion_fraction). The actual action
    trajectory is fed in as CONDITIONING instead, via a new "action_history"
    obs key -- literally AIETAlignmentSim3Sampler's own action encoding
    (absolute palm pose w/ sin/cos yaw + gripper_value, one 8-dim block per
    action_rel_indices offset, flattened) inherited unchanged from that
    class and reused here as an input rather than the diffused target (see
    _build_action_history, which is exactly get_action_trajectory's old
    computation, just renamed/repurposed). See aiet_wam_1.yaml's
    description for the full motivation.

    task-completion fraction = this waypoint's clipped episode-relative
    index / this episode's own final valid index -- 0 at episode start, 1 at
    the episode's own last frame. NOT a reliable completion signal for every
    bag: older bags kept recording after the task was actually done, so
    their episode length overstates how far a mid-episode waypoint really is
    from "done". This sampler computes it identically everywhere regardless
    -- see DatasetSampler's episode_min_bag_name (used by aiet_wam_1.yaml's
    dedicated wam_full_task stream, restricted to bags from toby_19 onward)
    for the stream actually meant to teach this output cleanly.
    """

    # every obs key whose FUTURE trajectory gets predicted (diffused) --
    # deliberately the exact same obs_keys_to_use as aiet_erlenmeyer_flask_16,
    # so this world model's targets are "everything the policy observes",
    # not a hand-picked subset. Order here is load-bearing: it's the exact
    # concatenation order build_action_chunk/get_action_trajectory produce,
    # which AIETWamBatchLoader's "action" normalizer must build its
    # per-block scale/offset in lockstep with.
    WAM_OBS_KEYS = [
        "palm_pose_xyz_rpy",
        "gripper_value",
        "biotac_lh",
        "wrist_cam_keypoints_dinov3l",
        "overhead_roi_keypoints_dinov3l",
        "env_cam_keypoints_dinov3l",
    ]

    def get_obs_sample(self, ep_idx):
        """
        Current-step WAM_OBS_KEYS (obs_keys_to_load must list exactly these
        -- NOT action_history, which isn't a raw rb key, see class
        docstring), plus the new "action_history" conditioning key.
        """
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            obs_sample[obs_key] = self.get_key_sample(self.resolve_rb_key(obs_key), ep_idx)
        obs_sample["action_history"] = self._build_action_history(ep_idx)[None, :]
        return obs_sample

    def _build_action_history(self, ep_idx):
        """
        [len(action_rel_indices) * 8] -- flattened (x,y,z,roll,pitch,sin_yaw,
        cos_yaw,gripper_value) at every action_rel_indices offset. Built via
        the inherited _gather_rel_palm_and_grip_by_offset (same computation
        AIETAlignmentSim3Sampler.get_action_trajectory used to diffuse
        directly), just fed to the model as conditioning here instead.
        """
        rel_palms, grips = self._gather_rel_palm_and_grip_by_offset(np.array([ep_idx]))
        palm = np.stack(rel_palms, axis=1)[0]  # [T, 7]
        grip = np.stack(grips, axis=1)[0]      # [T]
        trajectory = np.concatenate([palm, grip[:, None]], axis=-1).astype(np.float32)  # [T, 8]
        return trajectory.flatten()

    def _completion_fraction(self, ep_idx, offset):
        """
        Batched over ep_idx (accepts a scalar or an array) -- this waypoint's
        clipped episode-relative index / this episode's own final valid
        index. See class docstring for the reliability caveat.
        """
        episode_len = self.training_episode_end - self.training_episode_start
        final_idx = max(episode_len - 1, 1)
        idx = np.clip(np.asarray(ep_idx) + offset, 0, final_idx)
        return idx.astype(np.float32) / float(final_idx)

    def get_action_trajectory(self, ep_idx, key):
        """
        Overrides AIETAlignmentSim3Sampler.get_action_trajectory entirely --
        see class docstring. Delegates to build_action_chunk (single sample)
        so the slow per-sample path can never drift from the vectorized
        GPU-preload fast path (train_and_val._vectorized_load_dataset),
        which calls build_action_chunk directly and verifies its output
        against this path.
        """
        return self.build_action_chunk(np.array([ep_idx]))[0]

    def build_action_chunk(self, train_idx):
        """
        Batched world-model target -- see class docstring for the schema.
        Used by get_action_trajectory (single-sample case, above) AND by
        train_and_val._vectorized_load_dataset's GPU-preload fast path (see
        AIETAlignmentSim3Sampler.build_action_chunk's docstring for why this
        method needs to exist at all -- same reasoning applies here, just
        for a much wider per-step target).

        train_idx: [N] episode-relative train indices (NOT rb indices).
        Returns [N, len(action_rel_indices), D] where D = sum of
        WAM_OBS_KEYS' own flattened widths + 1 (completion fraction).
        """
        # np.array([]) (an empty episode's train indices) defaults to
        # dtype=float64 -- zarr's fancy-indexing dispatch requires an
        # integer dtype, same issue _gather_rel_palm_and_grip_by_offset
        # guards against.
        train_idx = np.asarray(train_idx, dtype=np.int64)

        rows_per_offset = []
        for offset in globals.CONFIG.action_rel_indices:  # type: ignore
            idx = train_idx + offset
            parts = []
            for obs_key in self.WAM_OBS_KEYS:
                rb_key = self.resolve_rb_key(obs_key)
                value = np.asarray(self.indices.get_sequence_by_train_indices_and_key(idx, rb_key))
                value = value.reshape(len(train_idx), -1).astype(np.float32)
                if obs_key == "palm_pose_xyz_rpy":
                    # same +-pi yaw discontinuity every aiet_alignment_*/
                    # aiet_erlenmeyer_flask_14+ action already sin/cos-encodes
                    # around -- applies just as much to predicting this as
                    # an observation as it did to predicting it as an action.
                    value = np.concatenate([
                        value[:, :5], np.sin(value[:, 5:6]), np.cos(value[:, 5:6]),
                    ], axis=-1)
                parts.append(value)
            parts.append(self._completion_fraction(train_idx, offset)[:, None])
            rows_per_offset.append(np.concatenate(parts, axis=-1))

        return np.stack(rows_per_offset, axis=1).astype(np.float32)  # [N, T, D]
