from typing import Dict, List, Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from torch.nn.attention import sdpa_kernel, SDPBackend


logger = logging.getLogger(__name__)


class _CrossAttnLoggingDecoderLayer(nn.TransformerDecoderLayer):
    """
    Same as nn.TransformerDecoderLayer, but when `log_attn` is set True on the
    instance, captures the trajectory->obs cross-attention weights (averaged
    over heads) into `last_attn_weights` for diagnostics. need_weights=True
    disables the fused SDPA path for that one call, so this is left off by
    default and only turned on for periodic logging.
    """
    log_attn = False
    last_attn_weights = None

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=self.log_attn,
            average_attn_weights=True,
        )
        if self.log_attn:
            self.last_attn_weights = attn_weights.detach()
        return self.dropout1(x)


class DexNexTransformerForDiffusion(ModuleAttrMixin):
    """
    Obs tokens (including the diffusion timestep token) self-attend among
    themselves first, then the trajectory decoder cross-attends to that
    obs memory. No causal masking: the full action chunk is denoised at
    once, so there's no "future leakage" to guard against.

    Each observation key gets its own token (and its own input projection),
    rather than being fused into one obs vector before tokenization.
    """

    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dims: Optional[Dict[str, int]] = None,
            patch_group_dims: Optional[Dict[str, Tuple[int, int]]] = None,
            num_tasks: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            n_cond_layers: int = 4,
            p_drop_token: float = 0.0,
            droppable_token_keys: Optional[List[str]] = None,
        ) -> None:
        """
        cond_dims: obs key -> feature dim, for single-token (fused) obs keys.
        patch_group_dims: patch-group name -> (num_patches, patch_feature_dim),
            for obs keys that arrive as an unpooled spatial token grid (e.g.
            DINOv3 patch tokens per camera view). Each group shares ONE
            weight-tied projection across its patches (like a ViT patch embed
            -- patches are homogeneous except for position), plus its own
            learned per-patch positional embedding and a learned per-group
            camera-identity embedding added to every patch in that group.
        """
        super().__init__()

        assert n_cond_layers > 0, "obs self-attention requires n_cond_layers > 0"

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        obs_as_cond = cond_dims is not None and len(cond_dims) > 0
        self.cond_keys = list(cond_dims.keys()) if cond_dims is not None else []
        self.patch_group_names = list(patch_group_dims.keys()) if patch_group_dims is not None else []
        self.has_patch_groups = len(self.patch_group_names) > 0
        self.patch_group_num_patches = {
            group: num_patches for group, (num_patches, _) in (patch_group_dims or {}).items()
        }
        self.embed_task_id = num_tasks > 0
        T_cond = 1  # timestep token
        if self.embed_task_id:
            T_cond += 1  # task-id token
        if obs_as_cond:
            T_cond += len(self.cond_keys)
        T_cond += sum(self.patch_group_num_patches.values())

        # whole-token (modality) dropout: randomly zero an entire cond token,
        # forcing the model not to over-rely on any one obs key (or task_id).
        # The timestep token is structurally required for diffusion and is
        # never eligible for dropping, regardless of what's passed in. Note:
        # since patch groups contribute many *repeated*-name tokens, listing a
        # patch group name here makes each of its patches independently
        # eligible for dropping -- i.e. this already gives per-patch dropout
        # for free once patch groups are in use, no separate mechanism needed.
        all_token_names = (['timestep']
            + (['task_id'] if self.embed_task_id else [])
            + self.cond_keys
            + self.patch_group_names)
        if droppable_token_keys is None:
            droppable_token_keys = [name for name in all_token_names if name != 'timestep']
        else:
            unknown = set(droppable_token_keys) - set(all_token_names)
            assert not unknown, f"droppable_token_keys contains unknown token names: {unknown}"
            assert 'timestep' not in droppable_token_keys, "the timestep token is required for diffusion and cannot be dropped"
        self.p_drop_token = p_drop_token
        self.droppable_token_keys = set(droppable_token_keys)

        # trajectory embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # obs/cond embedding stem: one projection per obs key, each becomes its own token
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            assert cond_dims is not None
            self.cond_obs_emb = nn.ModuleDict({
                key: nn.Linear(cond_dims[key], n_emb) for key in self.cond_keys
            })

        # task-id token: task identity is categorical/nominal (no ordering between
        # tasks), so use a plain lookup embedding rather than a sinusoidal/linear
        # projection of the raw id.
        self.task_id_emb = None
        if self.embed_task_id:
            self.task_id_emb = nn.Embedding(num_tasks, n_emb)

        # patch-group embedding stem: one shared (weight-tied) projection per
        # group applied to every patch in that group, a learned per-patch
        # positional grid, and a learned per-group camera-identity embedding
        # broadcast to every patch in the group. Camera identity is fixed by
        # input structure (not per-sample varying like task_id), so a plain
        # Parameter per group is enough -- no lookup table needed.
        self.patch_emb = None
        self.patch_pos_emb = None
        self.patch_camera_emb = None
        if self.has_patch_groups:
            assert patch_group_dims is not None
            self.patch_emb = nn.ModuleDict({
                group: nn.Linear(patch_dim, n_emb)
                for group, (_, patch_dim) in patch_group_dims.items()
            })
            self.patch_pos_emb = nn.ParameterDict({
                group: nn.Parameter(torch.zeros(1, num_patches, n_emb))
                for group, num_patches in self.patch_group_num_patches.items()
            })
            self.patch_camera_emb = nn.ParameterDict({
                group: nn.Parameter(torch.zeros(1, 1, n_emb))
                for group in self.patch_group_names
            })

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # obs self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_cond_layers
        )

        # trajectory decoder, cross-attends to obs memory
        decoder_layer = _CrossAttnLoggingDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.obs_as_cond = obs_as_cond

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.ModuleDict,
            nn.ParameterDict,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DexNexTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
            if module.patch_pos_emb is not None:
                for p in module.patch_pos_emb.values():
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)
            if module.patch_camera_emb is not None:
                for p in module.patch_camera_emb.values():
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")
        no_decay.add("_dummy_variable")
        for group in self.patch_group_names:
            no_decay.add(f"patch_pos_emb.{group}")
            no_decay.add(f"patch_camera_emb.{group}")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self,
            learning_rate: float=1e-4,
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    @torch.no_grad()
    def task_embedding_stats(self):
        """
        Diagnostics for whether task-id conditioning is doing anything: if
        rows stay close to their random initialization relative to each
        other, the model hasn't learned to differentiate tasks yet.

        Returns None if task embedding isn't in use, else a dict with:
        - 'mean_row_norm': (scalar) average L2 norm of task embedding rows
        - 'mean_pairwise_distance': (scalar) average pairwise L2 distance
          between all distinct task rows
        """
        if self.task_id_emb is None:
            return None
        weight = self.task_id_emb.weight  # (num_tasks, n_emb)
        num_tasks = weight.shape[0]
        row_norms = weight.norm(dim=-1)
        pairwise = torch.cdist(weight, weight)  # (num_tasks, num_tasks)
        if num_tasks > 1:
            off_diag_mask = ~torch.eye(num_tasks, dtype=torch.bool, device=weight.device)
            mean_pairwise_distance = pairwise[off_diag_mask].mean()
        else:
            mean_pairwise_distance = torch.zeros((), device=weight.device)
        return {
            'mean_row_norm': row_norms.mean(),
            'mean_pairwise_distance': mean_pairwise_distance,
        }

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[Dict[str, torch.Tensor]]=None,
        patches: Optional[Dict[str, torch.Tensor]]=None,
        task_ids: Optional[torch.Tensor]=None,
        log_attn: bool=False, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: dict mapping obs key -> (B,cond_dims[key]); each key becomes its own token
        patches: dict mapping patch-group name -> (B,num_patches,patch_feature_dim);
            each patch becomes its own token, sharing one projection per group
        task_ids: (B,) long tensor of task indices; becomes its own token
        log_attn: if True, captures the last decoder layer's cross-attention
            weights (trajectory -> obs memory) for diagnostics; see
            `last_cross_attn_entropy` / `last_cross_attn_token_names` afterwards.
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # 2. obs tokens: timestep token + task-id token + one token per obs key, self-attend
        cond_embeddings = time_emb
        cond_token_names = ['timestep']
        if self.embed_task_id:
            assert self.task_id_emb is not None and task_ids is not None
            task_ids = torch.reshape(task_ids, [-1]).long()
            task_emb = self.task_id_emb(task_ids).unsqueeze(1)
            # (B,1,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, task_emb], dim=1)
            cond_token_names.append('task_id')
        if self.obs_as_cond:
            assert self.cond_obs_emb is not None and cond is not None
            key_tokens = [self.cond_obs_emb[key](cond[key]).unsqueeze(1) for key in self.cond_keys]
            # each (B,1,n_emb)
            cond_embeddings = torch.cat([cond_embeddings] + key_tokens, dim=1)
            cond_token_names.extend(self.cond_keys)

        if self.has_patch_groups:
            assert self.patch_emb is not None and self.patch_pos_emb is not None
            assert self.patch_camera_emb is not None and patches is not None
            for group in self.patch_group_names:
                p = self.patch_emb[group](patches[group])
                # (B,num_patches,n_emb)
                p = p + self.patch_pos_emb[group] + self.patch_camera_emb[group]
                cond_embeddings = torch.cat([cond_embeddings, p], dim=1)
                cond_token_names.extend([group] * self.patch_group_num_patches[group])

        if self.training and self.p_drop_token > 0:
            eligible = torch.tensor(
                [name in self.droppable_token_keys for name in cond_token_names],
                device=cond_embeddings.device)
            # (B, T_cond): keep unless (eligible AND unlucky)
            rand = torch.rand(cond_embeddings.shape[0], cond_embeddings.shape[1], device=cond_embeddings.device)
            keep_mask = (rand >= self.p_drop_token) | ~eligible.unsqueeze(0)
            cond_embeddings = cond_embeddings * keep_mask.unsqueeze(-1).to(cond_embeddings.dtype)

        tc = cond_embeddings.shape[1]
        cond_position_embeddings = self.cond_pos_emb[:, :tc, :]
        x_cond = self.drop(cond_embeddings + cond_position_embeddings)
        # the fused/efficient SDPA backward kernel has a known NaN-producing edge
        # case under near-saturated softmax; force the naive math backend for
        # stability. Negligible cost here since sequences are short.
        with sdpa_kernel(SDPBackend.MATH):
            memory = self.encoder(x_cond)
        # (B,T_cond,n_emb)

        assert torch.isfinite(memory).all(), "obs memory contains non-finite values"

        # 3. trajectory tokens cross-attend to obs memory
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + position_embeddings)
        # (B,T,n_emb)

        last_decoder_layer = self.decoder.layers[-1]
        if log_attn:
            last_decoder_layer.log_attn = True
        with sdpa_kernel(SDPBackend.MATH):
            x = self.decoder(
                tgt=x,
                memory=memory
            )
        # (B,T,n_emb)
        if log_attn:
            last_decoder_layer.log_attn = False
            # (B,T,T_cond) -> mean over queries -> (B,T_cond)
            attn_weights = last_decoder_layer.last_attn_weights
            eps = 1e-8
            entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            self.last_cross_attn_entropy = entropy.mean(dim=-1)  # (B,)
            self.last_cross_attn_token_names = cond_token_names
            self.last_cross_attn_weight_by_token = attn_weights.mean(dim=1)  # (B,T_cond)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        
        assert torch.isfinite(x).all(), "output contains non-finite values"
        return x


def test():
    # obs self-attention + trajectory cross-attention, no obs cond
    transformer = DexNexTransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        n_cond_layers=4,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

    # obs self-attention + trajectory cross-attention, with one token per obs key.
    # cond_dims mirrors how ObsEncoderMaker derives obs keys/shapes: from
    # globals.CONFIG.obs_keys_to_use and globals.CONFIG.shape_meta, which
    # must already be loaded (e.g. via hydra) by whatever calls this test.
    import diffusion_policy.globals as globals

    cond_dims = {
        key: globals.CONFIG.shape_meta[key].shape[-1] #type:ignore
        for key in globals.CONFIG.obs_keys_to_use #type:ignore
    }

    transformer = DexNexTransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dims=cond_dims,
        n_cond_layers=4,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = {key: torch.zeros((4, dim)) for key, dim in cond_dims.items()}
    out = transformer(sample, timestep, cond)
