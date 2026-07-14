from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


import diffusion_policy.globals as globals
import robomimic.models.base_nets as rmbn
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_nets import ObservationEncoder
from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax


def per_key_output_dims(obs_encoder: ObservationEncoder) -> "OrderedDict[str, int]":
    """
    Like ObservationEncoder.output_shape(), but returns the flat output dim
    of each obs key individually instead of summing them into one total.
    Pure shape math, no forward pass.
    """
    dims = OrderedDict()
    for k in obs_encoder.obs_shapes:
        feat_shape = obs_encoder.obs_shapes[k]
        if obs_encoder.obs_randomizers[k] is not None:
            feat_shape = obs_encoder.obs_randomizers[k].output_shape_in(feat_shape)
        if obs_encoder.obs_nets[k] is not None:
            feat_shape = obs_encoder.obs_nets[k].output_shape(feat_shape)
        if obs_encoder.obs_randomizers[k] is not None:
            feat_shape = obs_encoder.obs_randomizers[k].output_shape_out(feat_shape)
        dims[k] = int(np.prod(feat_shape))
    return dims


def encode_obs_per_key(obs_encoder: ObservationEncoder, obs_dict) -> "OrderedDict[str, torch.Tensor]":
    """
    Like ObservationEncoder.forward(), but returns each obs key's processed,
    flattened feature ([B, D_k]) individually instead of concatenating them
    into one [B, D] vector.
    """
    feats = OrderedDict()
    for k in obs_encoder.obs_shapes:
        x = obs_dict[k]
        if obs_encoder.obs_randomizers[k] is not None:
            x = obs_encoder.obs_randomizers[k].forward_in(x)
        if obs_encoder.obs_nets[k] is not None:
            x = obs_encoder.obs_nets[k](x)
            if obs_encoder.activation is not None:
                x = obs_encoder.activation(x)
        if obs_encoder.obs_randomizers[k] is not None:
            x = obs_encoder.obs_randomizers[k].forward_out(x)
        feats[k] = TensorUtils.flatten(x, begin_axis=1)
    return feats

class StateRandomizer(rmbn.Randomizer):
    def __init__(self,
                 noise_mag = 1e-2,
                 use_add_noise = False, # add some noise to each input
                 use_mask_input = False, # randomly mask some inputs
                 use_during_eval = False,
                 ):
        super().__init__()
        
        self.noise_mag = noise_mag
        self.use_add_noise = use_add_noise
        self.use_mask_input = use_mask_input
        self.use_during_eval = use_during_eval
        
    def get_mask_chance(self):
        return 0.01
        
    def forward_in(self, inputs):
        if not self.use_during_eval and not self.training:
            return inputs
        
        state = inputs
        
        if self.use_add_noise:
            # add some noise to the state
            noise = self.noise_mag * torch.randn(state.shape, device=state.device)
            
            s2 = state + noise
        else:
            s2 = state
            
        if self.use_mask_input:
            p = torch.rand_like(s2, device=s2.device)
            
            to_mask = p < self.get_mask_chance()
            
            s3 = s2.clone()
            shape = s3[to_mask].shape
            
            # mask vals randomized in the range [-1, 1]
            mask_val = torch.rand(shape, device=s3.device) * 2.0 - 1.0
            
            # mask some inputs
            s3[to_mask] = mask_val
        else:
            s3 = s2
    
        return s3
    
    def forward_out(self, inputs):
        # do nothing
        return inputs
    
    def output_shape_out(self, input_shape=None):
        return input_shape
    
    def output_shape_in(self, input_shape=None):
        return input_shape
        
    

def make_ob(
        shape: tuple,
        ch,
        cw
):
    image_randomizer = rmbn.CropRandomizer(input_shape=shape, crop_height=ch, crop_width=cw)
    
    nb_channels = shape[0]

    net = CascadingCNNSpatialSoftmax(nb_channels)

    return image_randomizer, net

class ObsEncoderMaker():
    """
    Wrap into a class so we can construct using OmegaConf
    """
    def __init__(self,
            obs_keys_to_use: list,
            rgbs: dict,
            lowdims: list,
            ch,
            cw
    ):
        self.obs_keys_to_use = obs_keys_to_use
        self.rgbs = rgbs
        self.lowdims = lowdims
        self.ch = ch
        self.cw = cw
        
    def setup(self):
            
        # confirm that all obs keys are accounted for
        for key in self.obs_keys_to_use:
            assert(key in self.rgbs or key in self.lowdims)
        
        self.obs_encoder = ObservationEncoder(feature_activation=torch.nn.Mish)

        # rgb image inputs
        for key, val in self.rgbs.items():
            # check with global control
            if key not in globals.CONFIG.obs_keys_to_use:
                continue
            
            image_randomizer, net = make_ob(val.shape, self.ch, self.cw)

            # register the network for processing the modality
            self.obs_encoder.register_obs_key(
                name=key,
                shape=val.shape,
                net=net,
                randomizer=image_randomizer
            )
            
        # flat inputs aka lowdim or low_dim inputs
        for key in self.lowdims:
            shape = globals.CONFIG.shape_meta[key].shape
            
            # check with global control
            if key not in globals.CONFIG.obs_keys_to_use:
                continue
            
            self.obs_encoder.register_obs_key(
                name=key,
                shape=shape,
                randomizer=StateRandomizer()
            )
            
        # finally, make it
        self.obs_encoder.make()
            
            
    def get(self):
        return self.obs_encoder
    
    def copy(self):
        # make a new one with the same config
        other = ObsEncoderMaker(
            obs_keys_to_use=self.obs_keys_to_use,
            rgbs=self.rgbs,
            lowdims=self.lowdims,
            ch=self.ch,
            cw=self.cw
        )
        other.setup()
        
        return other
    
class WeightedFeature:
    """
    
    """
    def __init__(self,
        weight: float
        ):
        self.weight = weight
    
    # TODO: rename eval to validate
    def compute_weighted_features(self, features):
        # this batch_loss should already be in val mode
        wfeatures = self.weight * features
        return wfeatures
    
def CalcMeanFeature(wfeaturess):
    catd = torch.cat(wfeaturess)
    out = torch.mean(catd)
    return out

class ObsEncoder:
    def w_average_the_history(self, nbatch):
        """
        encode obs history by calculating a weighted average
        """
        
        # obs's
        assert('obss' in nbatch)

        wnobs_featuress = []
        for nobs in nbatch['obss']:
            nobs_features = self.one_obs_encoder(nobs)

            # weight
            weighter = WeightedFeature(w)
            wnobs_features = weighter.compute_weighted_features(nobs_features)

            # add to list
            wnobs_featuress.append(wnobs_features)

        # get the avg obs
        nobs_features = CalcMeanFeature(wnobs_featuress)

        return nobs_features
    
    def encode_obs(self, nbatch):
        if self.average_the_history:
            nobs_features = self.w_average_the_history(nbatch)

        else:
            nobs = nbatch['obs']
            nobs_features = self.one_obs_encoder(nobs)

        return nobs_features
    
def test():
    import hydra
    from omegaconf import OmegaConf
    txt = """
_target_: diffusion_policy.model.obs_encoder.ObsEncoderMaker
rgbs:
    img:
        shape:
        - 3
        - 192
        - 192
    img2:
        shape:
        - 3
        - 192
        - 192
        
ch: 184
cw: 184

lowdims:
    state: # "state" came from the zarr generation script
        shape:
        - 35 # gofa (6), wr (2), th (5), ff (4), mf (4), biotacs (5), th-pos (3), ff-pos (3), mf-pos (3)
    """
    config = OmegaConf.create(txt)

    x: ObsEncoderMaker = hydra.utils.instantiate(config)
    
    obs_encoder: ObservationEncoder = x.get()
    sh = obs_encoder.output_shape() # no input_shape needed
    print("output shape: {}".format(sh))
    
    
if __name__ == "__main__":
    test()