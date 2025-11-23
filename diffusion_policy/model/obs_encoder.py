import torch
import torch.nn as nn


import diffusion_policy.globals as globals
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax

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
            rgbs: dict,
            lowdims: dict,
            ch,
            cw
    ):
        self.rgbs = rgbs
        self.lowdims = lowdims
        self.ch = ch
        self.cw = cw
        
        self.obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

        # rgb image inputs
        for key, val in rgbs.items():
            # check with global control
            if key not in globals.CONFIG.obs_keys_to_use:
                continue
            
            image_randomizer, net = make_ob(val.shape, ch, cw)

            # register the network for processing the modality
            self.obs_encoder.register_obs_key(
                name=key,
                shape=val.shape,
                net=net,
                randomizer=image_randomizer
            )
            
        # flat inputs aka lowdim or low_dim inputs
        for key, val in lowdims.items():
            # check with global control
            if key not in globals.CONFIG.obs_keys_to_use:
                continue
            
            self.obs_encoder.register_obs_key(
                name=key,
                shape=val.shape,
                randomizer=StateRandomizer()
            )
            
        # finally, make it
        self.obs_encoder.make()
            
            
    def get(self):
        return self.obs_encoder
    
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