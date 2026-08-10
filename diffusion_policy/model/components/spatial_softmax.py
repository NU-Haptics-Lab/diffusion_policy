import torch
from torch import nn


class SpatialSoftmax(nn.Module):
    """
    Standalone spatial-softmax reduction over a patch-token grid (Finn et
    al.): (B, num_patches, patch_dim) -> (B, num_keypoints, 2) normalized
    [-1,1]^2 coordinates.

    Same math as DexNexTransformerForDiffusion._spatial_softmax, extracted
    here so new code (e.g. PelletLocalizer) can reuse it without depending
    on the transformer's internals. Deliberately NOT used to refactor the
    transformer's own inline copy (ss_reduce/ss_temperature/ss_grid_*) --
    that would rename its parameters, silently breaking any checkpoint
    resumed with strict=False (aiet_erlenmeyer_flask_3's in-progress runs).
    """
    def __init__(self, H: int, W: int, patch_dim: int, num_keypoints: int, temperature_init: float = 1.0):
        super().__init__()
        self.reduce = nn.Linear(patch_dim, num_keypoints)
        self.temperature = nn.Parameter(torch.full((num_keypoints,), temperature_init))
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H*W, 2)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_patches, patch_dim), num_patches == H*W.
        Returns (B, num_keypoints, 2).
        """
        feat = self.reduce(x) / self.temperature
        weights = torch.softmax(feat, dim=1)  # softmax over spatial positions
        coords = torch.einsum('bpk,pc->bkc', weights, self.grid)
        return coords
