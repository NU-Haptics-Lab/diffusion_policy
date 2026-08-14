import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.samplers.aiet_erlenmeyer_flask_sampler import AIETErlenmeyerFlaskSampler


def _wrap_rpy_delta(rel_rpy):
    """Wrap an rpy delta (..., 3) into (-pi, pi] elementwise. Plain
    subtraction of two raw roll/pitch/yaw readings is only correct while
    neither angle has crossed the +-pi discontinuity between them -- e.g.
    yaw=+3.13 minus yaw=-3.13 is a real ~6.9 degree rotation, but naive
    subtraction gives 6.26 rad (~359 degrees). Wrapping the DELTA (not the
    raw angles) fixes this regardless of which side of the discontinuity
    either raw reading happens to fall on."""
    return (rel_rpy + np.pi) % (2 * np.pi) - np.pi


class AIETAlignmentSim3Sampler(AIETErlenmeyerFlaskSampler):
    """
    Sampler for aiet_alignment_sim_3.yaml -- the lh_forearm-only sim (no abb
    gofa attached). Single-stage diffusion transformer, no aux losses, no
    pellet-keypoint/localizer machinery.

    palm_pose_xyz_rpy is NOT a model input (deliberately excluded from
    obs_keys_to_use/obs_keys_to_load -- see aiet_alignment_sim_3.yaml) --
    the point is to force the policy to localize off the wrist-camera DINOv3
    patch tokens instead of just reading off its own current pose. The
    action's palm component is therefore relative to the CURRENT palm pose
    (the ep_idx step, i.e. action_rel_indices offset 0), elementwise
    xyz/rpy subtraction (rpy wrapped into (-pi, pi] via _wrap_rpy_delta, so a
    physically-tiny rotation that straddles the +-pi discontinuity -- e.g.
    yaw swinging through the search behavior's rotation range -- doesn't
    read as a near-360-degree delta), same convention as
    DiffusionModel.ActionRelativeToState -- but computed here (not via
    action_relative_to_state=true) because that mechanism reads the state to
    subtract off nobs['state'], which would require palm_pose_xyz_rpy to
    still be a model input.

    gripper_value stays absolute (not relative) -- it's a [0, 1] "how closed"
    scalar, not a pose, so there's no ambiguity/drift concern that relative
    encoding would help with.

    Action is [rel_palm_pose_xyz_rpy (6), gripper_value (1)] = 7-dim, built
    here by concatenating/subtracting the two rb keys, since combined.zarr
    has no single "action" field -- config.action_key is just a placeholder
    ("action") not present in the rb, matched by AIETAlignmentSim3BatchLoader's
    get_fitted_nns doing the same relative-pose computation when fitting
    normalizer stats.
    """

    def get_action_trajectory(self, ep_idx, key):
        indices = np.array(globals.CONFIG.action_rel_indices) + ep_idx  # type: ignore
        palm = self.indices.get_sequence_by_train_indices_and_key(indices, "palm_pose_xyz_rpy")
        grip = self.indices.get_sequence_by_train_indices_and_key(indices, "gripper_value")

        current_palm = self.indices.get_sequence_by_train_indices_and_key(
            np.array([ep_idx]), "palm_pose_xyz_rpy"
        )  # [1, 6]
        rel_palm = palm - current_palm  # broadcast over the trajectory dim
        rel_palm[:, 3:] = _wrap_rpy_delta(rel_palm[:, 3:])  # see _wrap_rpy_delta

        return np.concatenate([rel_palm, grip[:, None]], axis=-1).astype(np.float32)

    def _gather_rel_palm_and_grip_by_offset(self, train_idx):
        """
        Shared core of get_all_action_components/build_action_chunk below:
        for each offset in action_rel_indices, the (rel_palm [N, 6],
        gripper_value [N]) column it contributes -- via
        self.indices.get_sequence_by_train_indices_and_key, so this can't
        drift from get_action_trajectory's per-sample logic (same
        fill-back/fill-forward indexing, just batched over train_idx instead
        of one ep_idx at a time).

        Returns (rel_palms, grips), each a list of length len(action_rel_indices).
        """
        # np.array([]) (an empty episode's train indices) defaults to
        # dtype=float64 -- zarr's fancy-indexing dispatch requires an
        # integer dtype (is_integer_array), so a float array silently falls
        # through to BasicIndexer, which can't handle arrays at all and
        # raises. Force int64 explicitly so this can't happen regardless of
        # whether train_idx is empty.
        train_idx = np.asarray(train_idx, dtype=np.int64)
        current_palm = self.indices.get_sequence_by_train_indices_and_key(train_idx, "palm_pose_xyz_rpy")

        rel_palms = []
        grips = []
        for offset in globals.CONFIG.action_rel_indices:  # type: ignore
            future_palm = self.indices.get_sequence_by_train_indices_and_key(train_idx + offset, "palm_pose_xyz_rpy")
            rel_palm = future_palm - current_palm
            rel_palm[:, 3:] = _wrap_rpy_delta(rel_palm[:, 3:])  # see _wrap_rpy_delta
            rel_palms.append(rel_palm)
            grips.append(self.indices.get_sequence_by_train_indices_and_key(train_idx + offset, "gripper_value"))
        return rel_palms, grips

    def get_all_action_components(self):
        """
        Every valid sample this episode can produce -- used by
        AIETAlignmentSim3BatchLoader to fit the action normalizer over the
        whole dataset.

        Returns (rel_palm [N, 6], gripper_value [N]) where N = len(this
        episode's valid indices) * len(action_rel_indices) -- NOT one row
        per sample, since each sample contributes its whole action window.
        """
        all_indices = np.array(self.indices.get_all_train_indices())
        rel_palms, grips = self._gather_rel_palm_and_grip_by_offset(all_indices)

        rel_palm = np.concatenate(rel_palms, axis=0).astype(np.float32)
        grip = np.concatenate(grips, axis=0).astype(np.float32)
        return rel_palm, grip

    def build_action_chunk(self, train_idx):
        """
        Batched, per-episode equivalent of get_action_trajectory -- used by
        train_and_val._vectorized_load_dataset's GPU-preload fast path,
        which otherwise assumes the action is a single literal replay-buffer
        array it can read once and fancy-index (it can't here: our action is
        [rel_palm_pose_xyz_rpy, gripper_value], built from two DIFFERENT rb
        arrays, not an alias of one -- see resolve_action_rb_key's
        docstring). Declining that fast path entirely and falling back to
        one Python-level dataset[i] call per sample was correct but slow
        (see aiet_alignment_sim_3.yaml's preload_to_gpu comment) -- this lets
        it stay fast while still funneling through the exact same
        _gather_rel_palm_and_grip_by_offset get_all_action_components uses,
        so it can't silently drift from get_action_trajectory's output.

        train_idx: [N] episode-relative train indices (NOT rb indices --
        same convention get_action_trajectory's ep_idx uses).
        Returns [N, len(action_rel_indices), 7].
        """
        rel_palms, grips = self._gather_rel_palm_and_grip_by_offset(train_idx)
        rel_palm = np.stack(rel_palms, axis=1)  # [N, L, 6]
        grip = np.stack(grips, axis=1)  # [N, L]
        return np.concatenate([rel_palm, grip[:, :, None]], axis=-1).astype(np.float32)
