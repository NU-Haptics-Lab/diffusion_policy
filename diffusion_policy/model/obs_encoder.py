import torch
import torch.nn as nn

import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
from diffusion_policy.model.components.dexnex_layers import CascadingCNNSpatialSoftmax


def make_ob(
        shape: tuple,
        ch,
        cw
):
    image_randomizer = rmbn.CropRandomizer(input_shape=shape, crop_height=ch, crop_width=cw)

    net = CascadingCNNSpatialSoftmax()

    return image_randomizer, net

class ObsEncoder:
    def __init__(self,
            obs_cfg: dict,
            ch: int,
            cw: int
    ):
        obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

        for key, val in obs_cfg.items():
            if val.type == "rgb":
                image_randomizer, net = make_ob(val, ch, cw)

                # register the network for processing the modality
                obs_encoder.register_obs_key(
                    name=key,
                    shape=camera1_shape,
                    net=net,
                    randomizer=image_randomizer
                )