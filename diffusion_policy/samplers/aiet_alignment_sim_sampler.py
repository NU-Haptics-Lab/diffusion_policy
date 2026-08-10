import numpy as np
import diffusion_policy.globals as globals
from diffusion_policy.samplers.aiet_erlenmeyer_flask_sampler import AIETErlenmeyerFlaskSampler

# up to this many pellets are visible per sample -- see
# wrist_pellet_keypoints_px/wrist_pellet_keypoints_valid in combined.zarr
MAX_PELLET_KEYPOINTS = 4


class AIETAlignmentSimSampler(AIETErlenmeyerFlaskSampler):
    """
    Sampler for aiet_alignment_sim.yaml. Same obs/action handling as the
    shared base class (which aiet_alignment_sim used directly before this
    existed), plus three extra top-level sample fields -- NOT obs, since
    they're not policy inputs, only supervision targets for
    PelletLocalizer's aux losses (see batch_loss.BatchLoss):

      wrist_pellet_keypoints_px    [4, 2] float32 -- pixel (x, y) location of
                                    each visible pellet in wrist_camera_image,
                                    zero-padded past however many are valid
      wrist_pellet_keypoints_valid [4]    bool    -- which of the 4 slots are
                                    real pellets (1-4 valid per sample) vs padding
      target_pellet_valid          scalar bool    -- whether target_pellet_location
                                    (an obs key, see aiet_alignment_sim.yaml)
                                    is meaningful for this sample at all; used
                                    to mask the xyz-regression loss and as the
                                    has-pellet classifier's supervision target

    Falls back to "no pellet" if these keys are missing from the rb (they
    were added after the rest of this dataset existed, and this sampler
    class is aiet_alignment_sim-specific anyway, so the fallback is really
    just defensive, not expected to trigger in practice).
    """

    def get_sample(self, ep_idx):
        sample = super().get_sample(ep_idx)

        try:
            sample["wrist_pellet_keypoints_px"] = self.get_key_sample("wrist_pellet_keypoints_px", ep_idx)
        except KeyError:
            sample["wrist_pellet_keypoints_px"] = np.zeros((1, MAX_PELLET_KEYPOINTS, 2), dtype=np.float32)

        try:
            sample["wrist_pellet_keypoints_valid"] = self.get_key_sample("wrist_pellet_keypoints_valid", ep_idx)
        except KeyError:
            sample["wrist_pellet_keypoints_valid"] = np.zeros((1, MAX_PELLET_KEYPOINTS), dtype=np.float32)

        try:
            sample["target_pellet_valid"] = self.get_key_sample("target_pellet_valid", ep_idx)
        except KeyError:
            sample["target_pellet_valid"] = np.zeros((1,), dtype=np.float32)

        return sample
