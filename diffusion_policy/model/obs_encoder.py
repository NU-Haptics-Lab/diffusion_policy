import torch
import torch.nn as nn


import diffusion_policy.globals as globals
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax


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
            )
            
        # finally, make it
        self.obs_encoder.make()
            
            
    def get(self):
        return self.obs_encoder
    
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