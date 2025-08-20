import torch
import torch.nn as nn
from diffusion_policy.model.components.trunk import Trunk


class Tree(nn.Module):
    """
    a Tree-style nn module, where it has a common trunk and separate leafs (a.k.a. heads) for each dataset.
    """
    def __init__(self,
            trunk: Trunk,
            leafs: dict
            ):
        self.trunk = trunk
        self.leafs = leafs

    def forward(self, input, branch_key):
        x = self.trunk(input)

        branch = self.leafs[branch_key]
        x = branch(x)

        return x