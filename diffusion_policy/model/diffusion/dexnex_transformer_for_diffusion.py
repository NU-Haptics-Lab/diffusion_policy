from typing import Dict, Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

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
            num_tasks: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            n_cond_layers: int = 4
        ) -> None:
        super().__init__()

        assert n_cond_layers > 0, "obs self-attention requires n_cond_layers > 0"

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        obs_as_cond = cond_dims is not None and len(cond_dims) > 0
        self.cond_keys = list(cond_dims.keys()) if cond_dims is not None else []
        self.embed_task_id = num_tasks > 0
        T_cond = 1  # timestep token
        if self.embed_task_id:
            T_cond += 1  # task-id token
        if obs_as_cond:
            T_cond += len(self.cond_keys)

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
        decoder_layer = nn.TransformerDecoderLayer(
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

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[Dict[str, torch.Tensor]]=None,
        task_ids: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: dict mapping obs key -> (B,cond_dims[key]); each key becomes its own token
        task_ids: (B,) long tensor of task indices; becomes its own token
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
        if self.embed_task_id:
            assert self.task_id_emb is not None and task_ids is not None
            task_ids = torch.reshape(task_ids, [-1]).long()
            task_emb = self.task_id_emb(task_ids).unsqueeze(1)
            # (B,1,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, task_emb], dim=1)
        if self.obs_as_cond:
            assert self.cond_obs_emb is not None and cond is not None
            key_tokens = [self.cond_obs_emb[key](cond[key]).unsqueeze(1) for key in self.cond_keys]
            # each (B,1,n_emb)
            cond_embeddings = torch.cat([cond_embeddings] + key_tokens, dim=1)
        tc = cond_embeddings.shape[1]
        cond_position_embeddings = self.cond_pos_emb[:, :tc, :]
        x_cond = self.drop(cond_embeddings + cond_position_embeddings)
        memory = self.encoder(x_cond)
        # (B,T_cond,n_emb)

        # 3. trajectory tokens cross-attend to obs memory
        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory
        )
        # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
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
