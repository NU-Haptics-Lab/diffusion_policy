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


class _AdaLNDecoderLayer(_CrossAttnLoggingDecoderLayer):
    """
    DiT-style AdaLN-Zero timestep conditioning, as an alternative to giving
    timestep its own cross-attention token (which is what
    normalize_cond_tokens addresses instead -- see
    DexNexTransformerForDiffusion.__init__). Rather than competing for
    attention, the timestep embedding is projected (per-layer, via a
    zero-initialized MLP so training starts identical to no conditioning) into
    a scale/shift/gate triple for each of this layer's three sublayers
    (self-attn, cross-attn, feedforward): `modulate(norm(x)) = norm(x)*(1+scale)
    + shift`, with the sublayer's residual contribution multiplied by `gate`.
    Zero-initializing the final MLP layer means every gate starts at 0, so the
    layer is initially a no-op wrapper around the ordinary norm_first decoder
    layer and only learns to use timestep conditioning as training progresses
    -- the standard AdaLN-Zero stabilization trick.

    `adaln_emb` (B, n_emb) must be set externally on the instance before
    calling forward() -- nn.TransformerDecoder.forward has a fixed per-layer
    call signature with no room for extra conditioning args, so the caller
    sets this attribute on every layer right before invoking self.decoder(...).
    """
    adaln_emb: Optional[torch.Tensor] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.linear1.in_features
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)
        # marks this Linear so the outer model's generic _init_weights pass
        # (which re-inits every nn.Linear to normal_(std=0.02)) skips it --
        # otherwise it would clobber the zero-init this class depends on for
        # AdaLN-Zero's "starts as a no-op" property
        self.adaln_mlp[-1]._skip_generic_init = True

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
            tgt_key_padding_mask=None, memory_key_padding_mask=None,
            tgt_is_causal=None, memory_is_causal=False):
        assert self.adaln_emb is not None, "adaln_emb must be set on this layer before forward()"
        params = self.adaln_mlp(self.adaln_emb)  # (B, 9*d_model)
        (shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_mlp, scale_mlp, gate_mlp) = params.chunk(9, dim=-1)

        x = tgt
        x = x + gate_sa.unsqueeze(1) * self._sa_block(
            self._modulate(self.norm1(x), shift_sa, scale_sa),
            tgt_mask, tgt_key_padding_mask, is_causal=bool(tgt_is_causal))
        x = x + gate_ca.unsqueeze(1) * self._mha_block(
            self._modulate(self.norm2(x), shift_ca, scale_ca),
            memory, memory_mask, memory_key_padding_mask, is_causal=bool(memory_is_causal))
        x = x + gate_mlp.unsqueeze(1) * self._ff_block(
            self._modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


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
            spatial_softmax_group_dims: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
            spatial_softmax_temperature_init: float = 1.0,
            separate_spatial_softmax_by_data_source: bool = False,
            num_tasks: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            n_cond_layers: int = 4,
            p_drop_token: float = 0.0,
            droppable_token_keys: Optional[List[str]] = None,
            sim_only_token_keys: Optional[List[str]] = None,
            real_only_token_keys: Optional[List[str]] = None,
            excluded_token_keys_by_data_source: Optional[Dict[int, List[str]]] = None,
            num_data_sources: Optional[int] = None,
            normalize_cond_tokens: bool = True,
            use_adaln_timestep: bool = False,
            gaussian_noise_by_data_source: Optional[Dict[int, List[str]]] = None,
            gaussian_noise_std: float = 0.1,
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
        spatial_softmax_group_dims: patch-group name -> (H, W, patch_feature_dim,
            num_keypoints). An alternative to patch_group_dims for the same
            kind of raw spatial patch grid input, but instead of keeping every
            patch as its own token, reduces the whole grid to ONE token via a
            classic spatial softmax (Finn et al.): a learned per-position
            linear reduces the channel dim to num_keypoints "heatmaps", each
            heatmap is softmaxed over spatial position, then the expected
            (x,y) coordinate per keypoint is computed against a fixed
            normalized grid -- giving a (2*num_keypoints)-dim descriptor that
            plugs into the same single-token pathway as cond_dims.
        separate_spatial_softmax_by_data_source: if True, each spatial-softmax
            group gets its own ss_reduce/ss_temperature PER data_source
            (requires num_data_sources/excluded_token_keys_by_data_source to
            be set) instead of one set of weights shared across every
            cotraining source. Default False (shared) risks one source's
            easier-to-fit gradient dominating the shared reduction and
            starving the harder source's localization -- e.g. a precisely
            rendered sim domain converging to real spatial tracking while a
            noisier real domain's heatmap collapses to a flat, uninformative
            one (always predicting the image center regardless of true
            object position). Costs one extra Linear+temperature per group
            per data_source, and forward() computes every source's
            projection for every sample then selects per-sample by
            data_source (simpler than routing samples by source, and cheap
            since num_data_sources is small).
        sim_only_token_keys / real_only_token_keys: token names that should be
            hard-zeroed (after projection, deterministically, at both train
            and eval time -- NOT the same mechanism as p_drop_token, which is
            random and training-only) for samples whose data_source marks
            them as real (0) or sim (1) respectively, e.g. for cotraining on
            a real dataset lacking some sim-only observation, or vice versa.
            Binary-only (2 data sources) -- kept for backward compatibility
            with existing 2-source configs; internally converted to
            excluded_token_keys_by_data_source = {0: sim_only_token_keys,
            1: real_only_token_keys} when that param isn't given directly.
        excluded_token_keys_by_data_source: generalizes the above to any
            number of cotraining sources: data_source id -> token names that
            must be hard-zeroed for samples tagged with that id. E.g. for a
            3-way cotrain with data_source 0/1/2, a token only meaningful to
            source 1 would appear in this dict under keys 0 and 2. Takes
            precedence over sim_only_token_keys/real_only_token_keys if both
            are given.
        num_data_sources: how many distinct data_source ids can appear at
            runtime (ids are 0..num_data_sources-1). Required whenever a
            valid data_source id has NO entry in
            excluded_token_keys_by_data_source (e.g. a 3-source cotrain where
            only source 1 excludes anything -- inferring num_data_sources as
            max(dict keys)+1 would silently miss source 2 and crash at
            forward() with an out-of-bounds gather). Defaults to
            max(excluded_token_keys_by_data_source keys)+1 if omitted (or 2
            when only the legacy sim_only/real_only_token_keys are given) --
            fine when every source has at least one exclusion, wrong
            otherwise, so pass this explicitly for any cotrain with 3+
            sources.
        normalize_cond_tokens: if True (default), apply a per-token-name
            LayerNorm to each cond token's embedding (timestep, task_id, each
            cond_dims/spatial-softmax key, each patch group) right after its
            own projection, before concatenation into the cond sequence. This
            is separate from the encoder's internal norm_first LayerNorms:
            those are re-applied before every self-attention sublayer, but
            the encoder has no final `norm`, and the decoder's cross-attention
            only normalizes the query side (not `memory`, i.e. not these cond
            tokens) -- so without this, a token whose raw projection happens
            to produce a larger norm (e.g. timestep, always-present and never
            token-dropped) can dominate the QK^T dot product purely from
            magnitude, independent of its actual informativeness.
        use_adaln_timestep: if True, timestep is NOT given a cross-attention
            token at all -- instead each decoder layer gets its own AdaLN-Zero
            modulation (scale/shift/gate, zero-initialized) computed from the
            timestep embedding, applied around that layer's self-attn,
            cross-attn, and feedforward sublayers. See _AdaLNDecoderLayer.
            This is the more standard fix for timestep specifically (DiT-style)
            and is mutually exclusive with normalize_cond_tokens's handling of
            the timestep token, since there is no timestep token in this mode.
            Default False -- normalize_cond_tokens is the current default fix.
        gaussian_noise_by_data_source: data_source id -> list of cond_dims/
            spatial-softmax key names to add Gaussian noise to, for samples
            tagged with that data_source, at training time only. Applies
            AFTER the raw value is computed (i.e. after the spatial-softmax
            (x,y) reduction for those groups, not to the raw patch grid) and
            BEFORE its projection into a token -- so a spatial-softmax key's
            noise perturbs its keypoint coordinates directly. Motivation: a
            cotraining source whose non-vision signals (e.g. simulated,
            noise-free proprioception) are unrealistically clean relative to
            another source's (e.g. real sensor readings) can let the model
            solve that source's task without leaning on vision much at all,
            which then doesn't transfer to a source where vision genuinely
            matters. Only cond_dims/spatial-softmax keys are supported (not
            timestep, task_id, or raw un-pooled patch_group_dims tokens).
            None (default) disables this entirely.
        gaussian_noise_std: standard deviation of the noise added above,
            shared across every (data_source, key) pair enabled by
            gaussian_noise_by_data_source. Values are on very different
            natural scales (radians/meters for palm pose, a [0,1] scalar for
            gripper, normalized [-1,1] keypoint coordinates) so this is a
            blunt, single global knob -- tune per-key manually if this turns
            out to be too coarse.
        """
        super().__init__()

        assert n_cond_layers > 0, "obs self-attention requires n_cond_layers > 0"

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon

        self.patch_group_names = list(patch_group_dims.keys()) if patch_group_dims is not None else []
        self.has_patch_groups = len(self.patch_group_names) > 0
        self.patch_group_num_patches = {
            group: num_patches for group, (num_patches, _) in (patch_group_dims or {}).items()
        }

        self.spatial_softmax_group_dims = dict(spatial_softmax_group_dims or {})
        self.spatial_softmax_group_names = list(self.spatial_softmax_group_dims.keys())
        overlap = set(self.spatial_softmax_group_names) & set(self.patch_group_names)
        assert not overlap, f"a group can't be in both patch_group_dims and spatial_softmax_group_dims: {overlap}"

        # spatial-softmax-reduced groups become single-token cond keys, exactly
        # like cond_dims -- they just get their (2*num_keypoints)-dim input
        # computed internally from `patches` instead of supplied via `cond`.
        combined_cond_dims = dict(cond_dims or {})
        overlap = set(combined_cond_dims.keys()) & set(self.spatial_softmax_group_names)
        assert not overlap, f"a key can't be in both cond_dims and spatial_softmax_group_dims: {overlap}"
        for group, (_, _, _, num_keypoints) in self.spatial_softmax_group_dims.items():
            combined_cond_dims[group] = num_keypoints * 2
        self.cond_keys = list(combined_cond_dims.keys())
        self._cond_key_to_idx = {key: i for i, key in enumerate(self.cond_keys)}
        obs_as_cond = len(self.cond_keys) > 0

        self.use_adaln_timestep = use_adaln_timestep
        self.embed_task_id = num_tasks > 0
        T_cond = 0 if use_adaln_timestep else 1  # timestep token (absent when AdaLN-conditioned instead)
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
        all_token_names = ((['timestep'] if not use_adaln_timestep else [])
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

        # deterministic, always-on (train + eval) per-data-source token masking
        # -- e.g. a sim-only obs key must be zeroed for real-robot samples and
        # vice versa, regardless of self.training. Generalized to N cotrain
        # sources via a per-token, per-data-source allow-matrix (registered as
        # a buffer so it moves with .to(device)/.half() automatically):
        # allow_matrix[ds, i] == 1 iff token i is visible to samples tagged
        # data_source == ds. forward() does one gather (allow_matrix[data_source])
        # instead of per-key scalar arithmetic, so this isn't limited to 2 sources.
        if excluded_token_keys_by_data_source is None:
            excluded_token_keys_by_data_source = {}
            if sim_only_token_keys:
                excluded_token_keys_by_data_source[0] = list(sim_only_token_keys)  # excluded for real
            if real_only_token_keys:
                excluded_token_keys_by_data_source[1] = list(real_only_token_keys)  # excluded for sim
        for ds, keys in excluded_token_keys_by_data_source.items():
            unknown = set(keys) - set(all_token_names)
            assert not unknown, f"excluded_token_keys_by_data_source[{ds}] contains unknown token names: {unknown}"
        self.excluded_token_keys_by_data_source: Dict[int, List[str]] = {
            ds: list(keys) for ds, keys in excluded_token_keys_by_data_source.items()
        }
        if num_data_sources is not None:
            self.num_data_sources = num_data_sources
        else:
            self.num_data_sources = (max(excluded_token_keys_by_data_source.keys()) + 1
                if excluded_token_keys_by_data_source else 0)
        if self.num_data_sources > 0:
            # mirror forward()'s exact cond_token_names construction (patch
            # groups repeat their name once per patch, not once per group --
            # unlike all_token_names above, which is only used for
            # droppable_token_keys validation and doesn't need per-patch
            # granularity) so this matrix's column i lines up with
            # cond_embeddings[:, i] in forward().
            expanded_token_names = ((['timestep'] if not use_adaln_timestep else [])
                + (['task_id'] if self.embed_task_id else [])
                + self.cond_keys)
            for group in self.patch_group_names:
                expanded_token_names.extend([group] * self.patch_group_num_patches[group])

            allow_matrix = torch.ones(self.num_data_sources, len(expanded_token_names))
            for ds, keys in self.excluded_token_keys_by_data_source.items():
                for i, name in enumerate(expanded_token_names):
                    if name in keys:
                        allow_matrix[ds, i] = 0.0
            self.register_buffer("data_source_token_allow_matrix", allow_matrix, persistent=False)
        else:
            self.data_source_token_allow_matrix = None

        # per-(data_source, cond_key) Gaussian noise: column order matches
        # self.cond_keys exactly (not expanded_token_names above -- this only
        # ever applies to single-token cond_dims/spatial-softmax keys, never
        # timestep/task_id/raw patch-group tokens).
        self.gaussian_noise_std = gaussian_noise_std
        if gaussian_noise_by_data_source:
            for ds, keys in gaussian_noise_by_data_source.items():
                unknown = set(keys) - set(self.cond_keys)
                assert not unknown, \
                    f"gaussian_noise_by_data_source[{ds}] contains keys that aren't " \
                    f"cond_dims/spatial-softmax keys: {unknown}"
            noise_nds = max(self.num_data_sources, max(gaussian_noise_by_data_source.keys()) + 1)
            noise_allow = torch.zeros(noise_nds, max(len(self.cond_keys), 1))
            for ds, keys in gaussian_noise_by_data_source.items():
                for i, key in enumerate(self.cond_keys):
                    if key in keys:
                        noise_allow[ds, i] = 1.0
            self.register_buffer("gaussian_noise_allow_matrix", noise_allow, persistent=False)
        else:
            self.gaussian_noise_allow_matrix = None

        # trajectory embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # obs/cond embedding stem: one projection per obs key, each becomes its own token.
        # spatial-softmax-reduced groups share this exact pathway -- their
        # (2*num_keypoints)-dim descriptor is computed on the fly in forward()
        # and projected here just like any other single-token cond key.
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.ModuleDict({
                key: nn.Linear(combined_cond_dims[key], n_emb) for key in self.cond_keys
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

        # spatial-softmax stem: a learned per-position channel reduction to
        # num_keypoints "heatmaps" (ss_reduce), a learned per-keypoint
        # temperature (ss_temperature, initialized to a constant -- not
        # randomly, so it's excluded from the usual weight init below), and a
        # fixed (non-learnable) normalized [-1,1]^2 coordinate grid per group.
        self.ss_reduce = None
        self.ss_temperature = None
        self.separate_spatial_softmax_by_data_source = separate_spatial_softmax_by_data_source
        if self.spatial_softmax_group_names:
            if self.separate_spatial_softmax_by_data_source:
                assert self.num_data_sources > 0, \
                    "separate_spatial_softmax_by_data_source requires num_data_sources " \
                    "(or excluded_token_keys_by_data_source) to be set"
                self.ss_reduce = nn.ModuleDict({
                    f'{group}__ds{ds}': nn.Linear(patch_dim, num_keypoints)
                    for group, (_, _, patch_dim, num_keypoints) in self.spatial_softmax_group_dims.items()
                    for ds in range(self.num_data_sources)
                })
                self.ss_temperature = nn.ParameterDict({
                    f'{group}__ds{ds}': nn.Parameter(torch.full((num_keypoints,), spatial_softmax_temperature_init))
                    for group, (_, _, _, num_keypoints) in self.spatial_softmax_group_dims.items()
                    for ds in range(self.num_data_sources)
                })
            else:
                self.ss_reduce = nn.ModuleDict({
                    group: nn.Linear(patch_dim, num_keypoints)
                    for group, (_, _, patch_dim, num_keypoints) in self.spatial_softmax_group_dims.items()
                })
                self.ss_temperature = nn.ParameterDict({
                    group: nn.Parameter(torch.full((num_keypoints,), spatial_softmax_temperature_init))
                    for group, (_, _, _, num_keypoints) in self.spatial_softmax_group_dims.items()
                })
            for group, (H, W, _, _) in self.spatial_softmax_group_dims.items():
                ys, xs = torch.meshgrid(
                    torch.linspace(-1, 1, H),
                    torch.linspace(-1, 1, W),
                    indexing='ij'
                )
                grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H*W, 2)
                self.register_buffer(f'ss_grid_{group}', grid, persistent=False)

        # per-token-name LayerNorm, applied right after each token's own
        # projection (see normalize_cond_tokens docstring above) -- one shared
        # LayerNorm per patch group (applied identically to every patch in
        # that group), not per individual patch.
        self.normalize_cond_tokens = normalize_cond_tokens
        self.cond_token_norm = None
        if self.normalize_cond_tokens:
            norm_names = ((['timestep'] if not use_adaln_timestep else [])
                + (['task_id'] if self.embed_task_id else [])
                + self.cond_keys
                + self.patch_group_names)
            self.cond_token_norm = nn.ModuleDict({
                name: nn.LayerNorm(n_emb) for name in norm_names
            })

        # populated by _spatial_softmax on every forward() call, keyed by
        # group name -> (B, num_keypoints, 2) pre-flatten coords. Lets an
        # auxiliary loss (e.g. supervising keypoints against known landmark
        # pixel locations) reuse forward()'s own computation instead of
        # running the vision stem a second time -- same pattern as
        # last_cross_attn_entropy/last_cross_attn_token_names below.
        self._last_ss_keypoints: Dict[str, torch.Tensor] = {}

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
        decoder_layer_cls = _AdaLNDecoderLayer if use_adaln_timestep else _CrossAttnLoggingDecoderLayer
        decoder_layer = decoder_layer_cls(
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
            nn.SiLU,
            nn.Sequential)
        if isinstance(module, nn.Linear) and getattr(module, '_skip_generic_init', False):
            pass
        elif isinstance(module, (nn.Linear, nn.Embedding)):
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
        for group in self.spatial_softmax_group_names:
            no_decay.add(f"ss_temperature.{group}")

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

    def spatial_softmax_temperature_stats(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Diagnostics for ss_temperature: lower temperature sharpens the
        softmax over spatial positions (more peaked, closer to a hard
        argmax -- tighter separation between keypoints' predicted (x,y));
        higher temperature flattens it (closer to uniform, keypoints drift
        toward the grid centroid regardless of input -- the same "always
        centered" symptom a collapsed/undertrained reduction produces, so
        this is worth watching alongside spatial_softmax_group_dims's
        spatial_softmax_temperature_init).

        Returns None if no spatial-softmax groups are in use, else a dict
        keyed by the same names as self.ss_temperature (one per group, or
        one per "{group}__ds{data_source}" if
        separate_spatial_softmax_by_data_source), each a
        {'mean', 'min', 'max'} dict of scalars over that entry's
        num_keypoints values.
        """
        if self.ss_temperature is None:
            return None
        return {
            name: {
                'mean': param.detach().mean(),
                'min': param.detach().min(),
                'max': param.detach().max(),
            }
            for name, param in self.ss_temperature.items()
        }

    def _spatial_softmax(self, group: str, x: torch.Tensor,
            data_source: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classic spatial softmax (Finn et al.), collapsing an entire patch grid
        into one small descriptor instead of keeping every patch as its own
        token.

        x: (B, num_patches, patch_dim) where num_patches == H*W for this group.
        data_source: (B,) tensor, required when separate_spatial_softmax_by_data_source
            is True -- selects each sample's own per-data_source ss_reduce/
            ss_temperature. Every data_source's projection is computed for
            every sample (num_data_sources is small, so this is cheap), then
            gathered per-sample -- simpler than sorting samples by source,
            and correct even if a batch mixes sources.
        Returns (B, num_keypoints*2): for each of num_keypoints learned
        "heatmaps" (a per-position linear reduction of the channel dim),
        softmax over spatial position, then the expected (x,y) coordinate
        against a fixed normalized [-1,1]^2 grid.
        """
        assert self.ss_reduce is not None and self.ss_temperature is not None
        if self.separate_spatial_softmax_by_data_source:
            assert data_source is not None, \
                "data_source is required when separate_spatial_softmax_by_data_source is True"
            ds = torch.reshape(data_source, [-1]).long()  # (B,)
            batch_idx = torch.arange(x.shape[0], device=x.device)
            # (num_data_sources, B, num_patches, num_keypoints) -> gather each
            # sample's own source's projection
            feats = torch.stack(
                [self.ss_reduce[f'{group}__ds{d}'](x) for d in range(self.num_data_sources)], dim=0)
            feat = feats[ds, batch_idx]  # (B, num_patches, num_keypoints)
            temps = torch.stack(
                [self.ss_temperature[f'{group}__ds{d}'] for d in range(self.num_data_sources)], dim=0)
            temp = temps[ds]  # (B, num_keypoints)
            feat = feat / temp.unsqueeze(1)  # broadcast over the num_patches dim
        else:
            feat = self.ss_reduce[group](x)  # (B, num_patches, num_keypoints)
            feat = feat / self.ss_temperature[group]
        weights = torch.softmax(feat, dim=1)  # softmax over spatial positions
        grid = getattr(self, f'ss_grid_{group}')  # (num_patches, 2)
        coords = torch.einsum('bpk,pc->bkc', weights, grid)  # (B, num_keypoints, 2)
        self._last_ss_keypoints[group] = coords
        return coords.reshape(coords.shape[0], -1)  # (B, num_keypoints*2)

    def _maybe_add_gaussian_noise(self, key: str, value: torch.Tensor,
            data_source: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Adds gaussian_noise_std Gaussian noise to `value` (a cond_dims/
        spatial-softmax key's raw value, shape (B, dim)), per-sample, for
        whichever samples' data_source has `key` enabled in
        gaussian_noise_by_data_source. No-op at eval time, if noise isn't
        configured at all, or if data_source isn't provided.
        """
        if not self.training or self.gaussian_noise_allow_matrix is None or data_source is None:
            return value
        col = self._cond_key_to_idx[key]
        ds = torch.reshape(data_source, [-1]).long()
        allow = self.gaussian_noise_allow_matrix[ds, col]  # (B,)
        if not bool(allow.any()):
            return value
        noise = torch.randn_like(value) * self.gaussian_noise_std
        extra_dims = (1,) * (value.dim() - 1)
        return value + noise * allow.reshape(-1, *extra_dims).to(value.dtype)

    def get_last_spatial_softmax_keypoints(self, group: str) -> Optional[torch.Tensor]:
        """
        Returns the (B, num_keypoints, 2) per-keypoint [-1,1]^2 coordinates
        from the most recent forward() call's spatial-softmax reduction for
        `group`, gradient-attached (backprop through this reaches
        ss_reduce/ss_temperature exactly as it would through the normal
        forward path). None if forward() hasn't run yet, or `group` isn't a
        spatial-softmax group.
        """
        return self._last_ss_keypoints.get(group)

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[Dict[str, torch.Tensor]]=None,
        patches: Optional[Dict[str, torch.Tensor]]=None,
        task_ids: Optional[torch.Tensor]=None,
        data_source: Optional[torch.Tensor]=None,
        log_attn: bool=False, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: dict mapping obs key -> (B,cond_dims[key]); each key becomes its own token
        patches: dict mapping patch-group name -> (B,num_patches,patch_feature_dim);
            each patch becomes its own token (patch_group_dims groups) or gets
            reduced to one spatial-softmax token (spatial_softmax_group_dims groups)
        task_ids: (B,) long tensor of task indices; becomes its own token
        data_source: (B,) tensor, 0=real 1=sim; hard-zeroes sim_only_token_keys
            for real samples and real_only_token_keys for sim samples, always
            (train + eval), regardless of p_drop_token
        log_attn: if True, captures the last decoder layer's cross-attention
            weights (trajectory -> obs memory) for diagnostics; see
            `last_cross_attn_entropy` / `last_cross_attn_token_names` /
            `last_cross_attn_weight_by_token` / `last_cross_attn_timesteps`
            afterwards -- the last one lets you bucket per-token attention by
            noise level instead of only seeing a batch-wide average.
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
        raw_time_emb = self.time_emb(timesteps)  # (B,n_emb), used as AdaLN input when use_adaln_timestep
        if self.use_adaln_timestep:
            cond_embeddings = None
            cond_token_names = []
        else:
            time_emb = raw_time_emb.unsqueeze(1)
            # (B,1,n_emb)
            if self.normalize_cond_tokens:
                assert self.cond_token_norm is not None
                time_emb = self.cond_token_norm['timestep'](time_emb)
            # 2. obs tokens: timestep token + task-id token + one token per obs key, self-attend
            cond_embeddings = time_emb
            cond_token_names = ['timestep']
        if self.embed_task_id:
            assert self.task_id_emb is not None and task_ids is not None
            task_ids = torch.reshape(task_ids, [-1]).long()
            task_emb = self.task_id_emb(task_ids).unsqueeze(1)
            # (B,1,n_emb)
            if self.normalize_cond_tokens:
                assert self.cond_token_norm is not None
                task_emb = self.cond_token_norm['task_id'](task_emb)
            cond_embeddings = task_emb if cond_embeddings is None else torch.cat([cond_embeddings, task_emb], dim=1)
            cond_token_names.append('task_id')
        if self.obs_as_cond:
            assert self.cond_obs_emb is not None
            key_tokens = []
            for key in self.cond_keys:
                if key in self.spatial_softmax_group_names:
                    assert patches is not None
                    key_input = self._spatial_softmax(key, patches[key], data_source=data_source)
                else:
                    assert cond is not None
                    key_input = cond[key]
                key_input = self._maybe_add_gaussian_noise(key, key_input, data_source)
                key_emb = self.cond_obs_emb[key](key_input).unsqueeze(1)
                if self.normalize_cond_tokens:
                    assert self.cond_token_norm is not None
                    key_emb = self.cond_token_norm[key](key_emb)
                key_tokens.append(key_emb)
            # each (B,1,n_emb)
            cond_embeddings = torch.cat(key_tokens, dim=1) if cond_embeddings is None \
                else torch.cat([cond_embeddings] + key_tokens, dim=1)
            cond_token_names.extend(self.cond_keys)

        if self.has_patch_groups:
            assert self.patch_emb is not None and self.patch_pos_emb is not None
            assert self.patch_camera_emb is not None and patches is not None
            for group in self.patch_group_names:
                p = self.patch_emb[group](patches[group])
                # (B,num_patches,n_emb)
                p = p + self.patch_pos_emb[group] + self.patch_camera_emb[group]
                if self.normalize_cond_tokens:
                    assert self.cond_token_norm is not None
                    p = self.cond_token_norm[group](p)
                cond_embeddings = p if cond_embeddings is None else torch.cat([cond_embeddings, p], dim=1)
                cond_token_names.extend([group] * self.patch_group_num_patches[group])

        assert cond_embeddings is not None, \
            "at least one non-timestep cond source (task_id/cond_keys/patch groups) is required when use_adaln_timestep=True"

        if data_source is not None and self.data_source_token_allow_matrix is not None:
            # deterministic, always-on masking (unlike p_drop_token): some
            # tokens are structurally absent for a given data source, not
            # randomly dropped for regularization -- applied before p_drop_token
            # so random dropout only ever acts on tokens that are genuinely
            # available. Generalizes to N sources via a gather against the
            # per-source allow-matrix built in __init__ (column order matches
            # cond_token_names exactly), rather than the old ds/(1-ds) binary
            # arithmetic.
            ds = torch.reshape(data_source, [-1]).long()  # (B,), values in [0, num_data_sources)
            avail = self.data_source_token_allow_matrix[ds].to(cond_embeddings.dtype)  # (B, T_cond)
            cond_embeddings = cond_embeddings * avail.unsqueeze(-1)

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
        
        # with sdpa_kernel(SDPBackend.MATH):
        #     memory = self.encoder(x_cond)
        memory = self.encoder(x_cond)
        
        
        # (B,T_cond,n_emb)

        # assert torch.isfinite(memory).all(), "obs memory contains non-finite values"

        # 3. trajectory tokens cross-attend to obs memory
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + position_embeddings)
        # (B,T,n_emb)

        last_decoder_layer = self.decoder.layers[-1]
        if log_attn:
            last_decoder_layer.log_attn = True

        if self.use_adaln_timestep:
            for layer in self.decoder.layers:
                layer.adaln_emb = raw_time_emb

        # with sdpa_kernel(SDPBackend.MATH):
        #     x = self.decoder(
        #         tgt=x,
        #         memory=memory
        #     )
        x = self.decoder(
            tgt=x,
            memory=memory
        )

        if self.use_adaln_timestep:
            for layer in self.decoder.layers:
                layer.adaln_emb = None

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
            # per-sample diffusion timestep, so callers can bucket
            # last_cross_attn_weight_by_token by noise level (e.g. mean weight
            # per token within each timestep bucket) instead of only seeing an
            # average across the whole batch's mix of noise levels
            self.last_cross_attn_timesteps = timesteps.detach()  # (B,)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        
        # assert torch.isfinite(x).all(), "output contains non-finite values"
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
