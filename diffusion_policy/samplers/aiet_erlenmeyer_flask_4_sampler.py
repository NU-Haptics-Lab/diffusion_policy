import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.samplers.aiet_alignment_sim_3_sampler import AIETAlignmentSim3Sampler


class AIETErlenmeyerFlask4Real1Sampler(AIETAlignmentSim3Sampler):
    """
    task_24 erlenmeyer-flask real leg of aiet_erlenmeyer_flask_4's 3-way
    cotrain (real-erlenmeyer + real-full-avatar-cotrain + sim-alignment).

    Reuses AIETAlignmentSim3Sampler's relative-palm-pose+gripper action
    construction unchanged: palm_pose_xyz_rpy and gripper_value now exist as
    literal arrays in task_24's zarr (added by
    avatar_intelligence/data_processing/common/add_palm_pose_fk.py and
    add_gripper_value.py), so no per-sample derivation is needed here, unlike
    the original plan's "derive gripper_value in the sampler" -- baking it
    into the zarr instead matches this repo's established
    precompute-everything convention and lets this class be a 3-line diff
    over AIETAlignmentSim3Sampler.

    Every obs key aiet_erlenmeyer_flask_4 declares (palm_pose_xyz_rpy,
    wrist_cam_patch_features/overhead_roi_patch_features, gripper_value,
    biotac_lh) exists natively in this zarr -- no zero-filling needed for
    this source.
    """

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)
        sample["data_source"] = np.array([0], dtype=np.float32)  # 0 = erlenmeyer real
        return sample

    def get_constant_sample_fields(self):
        return {"data_source": 0.0}


class AIETErlenmeyerFlask4Real2Sampler(AIETAlignmentSim3Sampler):
    """
    full_avatar_real_world_cotrain_dataset real leg of aiet_erlenmeyer_flask_4
    (left-avatar-half only -- right hand/arm, image_wrist_right, biotac_rh
    are all out of scope and never read).

    Like Real1Sampler, palm_pose_xyz_rpy and gripper_value are precomputed,
    literal top-level arrays in this zarr (via the same two add_*.py
    migration scripts, run with --source-key robot_joint_pos --source-cols
    0:30 to select the left arm+hand columns) -- the sampler never touches
    the raw 66-dim robot_joint_pos/robot_joint_action or the raw images
    (DINOv3 patch features were precomputed by add_dinov3_patch_features.py
    from image_wrist_left/image_left_cam, with an ROI crop applied to the
    latter before extraction, matching task_24's overhead_roi_* semantics).
    So this class needs no slicing/zero-filling either -- reuses
    AIETAlignmentSim3Sampler's action logic and AIETErlenmeyerFlaskSampler's
    patch_grid_size 7x7 substitution unchanged, exactly like Real1Sampler,
    differing only in data_source.
    """

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)
        sample["data_source"] = np.array([2], dtype=np.float32)  # 2 = full-avatar-cotrain real
        return sample

    def get_constant_sample_fields(self):
        return {"data_source": 2.0}


class AIETErlenmeyerFlask4SimSampler(AIETAlignmentSim3Sampler):
    """
    Sim-alignment leg of aiet_erlenmeyer_flask_4 -- same combined.zarr /
    action semantics as aiet_alignment_sim_3.yaml (AIETAlignmentSim3Sampler
    unmodified would suffice there), but `_4` needs 3 more adjustments on top
    of that reuse, all because combined.zarr's schema is narrower/differently
    named than the union `_4` declares:

    1. biotac_lh is a real-only obs key (present in both real legs, absent
       from sim) -- combined.zarr has no biotac_lh array at all, so it must
       be zero-filled.
    2. combined.zarr's single camera is named wrist_camera_patch_features
       (note: "wrist_camera_", not "wrist_cam_" -- a different name than
       task_24/full_avatar's wrist_cam_patch_features), and it has no
       precomputed _7x7 variant (unlike the two real sources) -- so instead
       of AIETErlenmeyerFlaskSampler.resolve_rb_key's usual "_7x7" suffix
       substitution, this pools the native 14x14 array down to 7x7 IN
       PYTHON at sample time when patch_grid_size == 7 (2x2 average pool,
       same math as ws_avatar_drake's add_7x7_pooled_patch_features.py --
       just not precomputed/baked into combined.zarr, since only this one
       cotrain config needs it pooled and it's cheap per-sample).
    3. combined.zarr has no overhead camera at all (single wrist-mounted
       camera only) -- overhead_roi_patch_features is zero-filled entirely
       for this source.

    All zero-filled/pooled values are shape-correct placeholders only --
    actual cross-source masking happens via data_source-driven token zeroing
    in the model (dexnex_transformer_for_diffusion.py's
    excluded_token_keys_by_data_source).
    """

    ZERO_FILL_KEYS = [
        "biotac_lh", "overhead_roi_patch_features", "overhead_roi_keypoints",
        "overhead_roi_keypoints_dinov3l",
        # combined.zarr has no third/env camera at all -- see
        # aiet_erlenmeyer_flask_10.yaml's token_keys_valid_only_for_task_ids
        # (this key is only genuinely present for real task_id 27 anyway, so
        # sim samples having it zero-filled+excluded is consistent with every
        # other real task_id, not sim-specific).
        "env_cam_keypoints_dinov3l",
    ]
    NATIVE_WRIST_PATCH_KEY = "wrist_camera_patch_features"  # always 14x14 -- no _7x7 variant in combined.zarr
    # precomputed fixed-tau spatial-softmax keypoints over NATIVE_WRIST_PATCH_KEY
    # -- combined.zarr uses "wrist_camera_" (not "wrist_cam_") naming, so this
    # can't go through the base class's KEYPOINT_RB_KEY_MAP. Assumes whatever
    # offline script adds this to combined.zarr follows
    # add_spatial_softmax_keypoints.py's f"{src_key}_spatial_softmax_xy"
    # naming convention against NATIVE_WRIST_PATCH_KEY -- update this constant
    # if that ends up named differently.
    NATIVE_WRIST_KEYPOINT_KEY = "wrist_camera_patch_features_spatial_softmax_xy"
    # dinov3-LARGE (1024-channel) counterpart, precomputed directly under this
    # name in combined.zarr (unlike NATIVE_WRIST_KEYPOINT_KEY above, this one
    # wasn't derived via add_spatial_softmax_keypoints.py's naming convention
    # -- confirmed by inspecting the zarr directly). See
    # aiet_erlenmeyer_flask_8.yaml, which cotrains with the real legs' own
    # wrist_cam_keypoints_dinov3l/overhead_roi_keypoints_dinov3l.
    NATIVE_WRIST_KEYPOINT_KEY_DINOV3L = "wrist_camera_keypoints_dinov3l"

    @staticmethod
    def _pool_2x2(patch_grid: np.ndarray) -> np.ndarray:
        """[N, 14, 14, D] -> [N, 7, 7, D] via non-overlapping 2x2 average pool."""
        n, h, w, d = patch_grid.shape
        return patch_grid.reshape(n, h // 2, 2, w // 2, 2, d).mean(axis=(2, 4))

    def get_obs_sample(self, ep_idx):
        obs_sample = {}
        for obs_key in globals.CONFIG.obs_keys_to_load:  # type: ignore
            if obs_key in self.ZERO_FILL_KEYS:
                shape = globals.CONFIG.shape_meta[obs_key].shape  # type: ignore
                obs_sample[obs_key] = np.zeros((1, *shape), dtype=np.float32)
            elif obs_key == "wrist_cam_patch_features":
                patch = self.get_key_sample(self.NATIVE_WRIST_PATCH_KEY, ep_idx).astype(np.float32)
                if getattr(globals.CONFIG, "patch_grid_size", 14) == 7:  # type: ignore
                    patch = self._pool_2x2(patch)
                obs_sample[obs_key] = patch
            elif obs_key == "wrist_cam_keypoints":
                obs_sample[obs_key] = self.get_key_sample(self.NATIVE_WRIST_KEYPOINT_KEY, ep_idx)
            elif obs_key == "wrist_cam_keypoints_dinov3l":
                obs_sample[obs_key] = self.get_key_sample(self.NATIVE_WRIST_KEYPOINT_KEY_DINOV3L, ep_idx)
            else:
                obs_sample[obs_key] = self.get_key_sample(obs_key, ep_idx)
        return obs_sample

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)
        sample["data_source"] = np.array([1], dtype=np.float32)  # 1 = sim
        # own dedicated task_id (distinct from the real legs' task_id==24),
        # so the (now-enabled) task embedding gives the model an explicit,
        # cheap signal to distinguish sim from real -- reusing task_id_emb
        # rather than adding new AdaLN/token machinery for a data_source
        # signal. num_tasks in the yaml must cover this id (>= 27).
        sample["task_id"] = np.array([26], dtype=np.float32)
        return sample

    def get_constant_sample_fields(self):
        return {"data_source": 1.0, "task_id": 26.0}

    def resolve_rb_key(self, obs_key):
        """
        Only matters for train_and_val._vectorized_load_dataset's fast-path
        probe (obs_key_map), which calls this directly instead of
        get_obs_sample -- get_obs_sample above is already correct on its own
        without this override. Base class's KEYPOINT_RB_KEY_MAP would
        resolve "wrist_cam_keypoints" to the WRONG name here (it assumes
        "wrist_cam_" naming; combined.zarr uses "wrist_camera_"), so this
        maps it explicitly instead of letting vectorization decline via a
        confusing not-found KeyError. "overhead_roi_keypoints" resolves to
        None like every other ZERO_FILL_KEYS entry -- see resolve_rb_key's
        docstring (None == zero-fill placeholder).
        """
        if obs_key == "wrist_cam_keypoints":
            return self.NATIVE_WRIST_KEYPOINT_KEY
        if obs_key == "wrist_cam_keypoints_dinov3l":
            return self.NATIVE_WRIST_KEYPOINT_KEY_DINOV3L
        if obs_key in self.ZERO_FILL_KEYS:
            return None
        return super().resolve_rb_key(obs_key)
