"""
DINOv3 image tokens for the dexnex diffusion transformer: raw camera images in, one flattened feature vector per camera
out (DiffusionModel adds each as a cond key, so the transformer's existing per-key Linear turns it into one token).

    image [B, H, W, 3] uint8
      -> DINOv3 (frozen, or fine-tuned in a second stage) -> patch tokens [B, g*g, D]   (g = img_size / 16)
      -> per-camera Linear D -> proj_dim (a 1x1 conv over the patch grid)              [B, g*g, proj_dim]
      -> dropout, flatten                                                              [B, g*g*proj_dim]

Flattening keeps one weight per (patch position, channel) in the transformer's per-key Linear: position-specific,
so it can learn where on the image the pellet / fingertips are, while each camera stays a single token (the patch grid
doesn't outnumber the proprioception tokens).

All cameras go through the backbone as one batch. The backbone runs in bf16 autocast; with freeze=True it runs under
no_grad and its parameters are excluded from the optimizer (requires_grad False). With freeze=False,
grad_checkpoint=True recomputes each block's activations in the backward pass (~1/10 the activation memory) and
backbone_lr_scale scales its learning rate (see DiffusionModel.optim_param_groups). The backbone is always kept in
eval mode (DINOv3 has no batch norm; this only pins its drop path / dropout off).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV3ImageTokens(nn.Module):
    def __init__(self,
                 image_keys: List[str],
                 repo_dir: str = "/home/omnid/dexnex/libraries/dinov3",
                 entrypoint: str = "dinov3_vitl16",
                 weights: str = "/media/dexnex_ssd/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                 img_size: int = 224,
                 proj_dim: int = 8,
                 dropout: float = 0.1,
                 freeze: bool = True,
                 grad_checkpoint: bool = False,
                 backbone_lr_scale: float = 0.1,
                 random_shift_px: int = 0,  # training-only random translation (pad + random crop, DrQ-style); 0 = off
                 ):
        super().__init__()
        self.image_keys = list(image_keys)
        self.img_size = img_size
        self.proj_dim = proj_dim
        self.freeze = freeze
        self.grad_checkpoint = grad_checkpoint
        self.backbone_lr_scale = backbone_lr_scale
        self.random_shift_px = random_shift_px

        self.backbone = torch.hub.load(repo_dir, entrypoint, source="local", weights=weights)
        self.backbone.eval()
        if freeze:
            self.backbone.requires_grad_(False)
        self.patch_size = self.backbone.patch_size
        assert img_size % self.patch_size == 0, (img_size, self.patch_size)
        self.grid = img_size // self.patch_size
        embed_dim = self.backbone.embed_dim

        self.proj = nn.ModuleDict({key: nn.Linear(embed_dim, proj_dim) for key in self.image_keys})
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def output_dims(self) -> Dict[str, int]:
        return {key: self.grid * self.grid * self.proj_dim for key in self.image_keys}

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # (see the module docstring)
        return self

    def backbone_parameters(self):
        return self.backbone.parameters()

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """[N, H, W, 3] uint8 (or float in [0, 255]) -> [N, 3, img_size, img_size] ImageNet-normalized float."""
        x = images.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True)
        if self.training and self.random_shift_px > 0:
            pad = self.random_shift_px
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
            n = x.shape[0]
            ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
            oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
            rows = (oy[:, None] + torch.arange(self.img_size, device=x.device)[None])  # [N, S]
            cols = (ox[:, None] + torch.arange(self.img_size, device=x.device)[None])
            x = x[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
                  rows[:, None, :, None], cols[:, None, None, :]]
        return (x - self.mean) / self.std

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv3's forward_features for one tensor, returning only the normalized patch tokens [N, g*g, D] --
        written out so each block can be gradient-checkpointed (forward_features has no hook for that)."""
        vit = self.backbone
        tokens, (H, W) = vit.prepare_tokens_with_masks(x, None)
        rope = vit.rope_embed(H=H, W=W) if vit.rope_embed is not None else None
        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        for blk in vit.blocks:
            if use_ckpt:
                tokens = checkpoint(lambda t, b=blk: b([t], [rope])[0], tokens, use_reentrant=False)
            else:
                tokens = blk([tokens], [rope])[0]
        # (DINOv3 normalizes patch tokens with self.norm whether or not its cls/patch norms are untied)
        return vit.norm(tokens[:, vit.n_storage_tokens + 1:])

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """images: key -> [B, H, W, 3] (uint8). Returns key -> [B, g*g*proj_dim] float32."""
        B = images[self.image_keys[0]].shape[0]
        x = self._preprocess(torch.cat([images[key] for key in self.image_keys], dim=0))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            if self.freeze:
                with torch.no_grad():
                    patches = self._patch_tokens(x)
            else:
                patches = self._patch_tokens(x)
        patches = patches.float()
        out = {}
        for i, key in enumerate(self.image_keys):
            p = self.proj[key](patches[i * B:(i + 1) * B])  # [B, g*g, proj_dim]
            out[key] = self.dropout(p).reshape(B, -1)
        return out
