import torch
from torch import nn

from diffusion_policy.model.components.spatial_softmax import SpatialSoftmax


class PelletLocalizer(nn.Module):
    """
    Predicts the target pellet's 3D position (and whether one is visible at
    all) from a wrist-camera DINOv3 patch grid plus lowdim state (joint
    positions, camera pose, ...). Sits upstream of the main diffusion
    transformer for aiet_alignment_sim: the transformer conditions only on
    (lowdim state, predicted pellet xyz), not on vision directly -- see
    DiffusionModel._maybe_run_pellet_localizer.

    Architecture: SpatialSoftmax (patch grid -> keypoint descriptor),
    concatenated with the lowdim inputs, through a shared MLP trunk, then
    two heads: xyz regression and a has-pellet classifier logit. The
    classifier exists because not every frame has a visible pellet (see
    wrist_pellet_keypoints_valid) -- xyz is meaningless supervision when
    there's nothing to localize, so BatchLoss's xyz regression loss is
    masked by the ground-truth has-pellet label (not the classifier's own
    prediction, to avoid a noisy self-gating signal early in training),
    while the classifier itself is supervised on every sample.

    Spatial softmax over mean-pooling or a mini-transformer+CLS-token: it's
    cheap, it's exactly what Finn et al. designed for (vision -> low-dim
    spatially-grounded bottleneck -> downstream regression), and it lets the
    intermediate keypoints ALSO be supervised against pixel-space ground
    truth (see diffusion_policy.losses.batch_loss.match_and_compute_keypoint_loss)
    alongside the xyz regression loss -- complementary signals through one
    bottleneck instead of one opaque one.
    """
    def __init__(self,
                 patch_h: int,
                 patch_w: int,
                 patch_dim: int,
                 lowdim_dim: int,
                 num_keypoints: int = 16,
                 hidden_dim: int = 128,
                 spatial_softmax_temperature_init: float = 1.0,
                 output_dim: int = 3,
                 ):
        super().__init__()
        self.spatial_softmax = SpatialSoftmax(patch_h, patch_w, patch_dim, num_keypoints, spatial_softmax_temperature_init)
        self.trunk = nn.Sequential(
            nn.Linear(num_keypoints * 2 + lowdim_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.xyz_head = nn.Linear(hidden_dim, output_dim)
        self.has_pellet_head = nn.Linear(hidden_dim, 1)

    def forward(self, patches: torch.Tensor, lowdim: torch.Tensor):
        """
        patches: (B, num_patches, patch_dim), num_patches == patch_h*patch_w
        lowdim:  (B, lowdim_dim) -- concatenated joint_positions/camera_pose/etc.

        Returns (pred_xyz, has_pellet_logit, keypoints), all gradient-attached:
          pred_xyz:         (B, output_dim)
          has_pellet_logit: (B,) -- raw logit, pass through sigmoid/BCEWithLogitsLoss
          keypoints:        (B, num_keypoints, 2) -- for the pixel-space aux loss
        """
        keypoints = self.spatial_softmax(patches)  # (B, num_keypoints, 2)
        x = torch.cat([keypoints.reshape(keypoints.shape[0], -1), lowdim], dim=-1)
        h = self.trunk(x)
        pred_xyz = self.xyz_head(h)
        has_pellet_logit = self.has_pellet_head(h).squeeze(-1)
        return pred_xyz, has_pellet_logit, keypoints
