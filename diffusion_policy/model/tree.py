import torch
import torch as nn
from diffusion_policy.model.trunk import Trunk


class Tree(nn.Module):
    """
    a Tree-style nn module, where it has a common trunk and separate branches (a.k.a. heads) for each dataset.
    """
    def __init__(self,
            trunk: Trunk,
            branches: dict
            ):
        self.trunk = trunk
        self.branches = branches

    def forward(self, input, branch_key):
        x = self.trunk(input)

        branch = self.branches[branch_key]
        x = branch(x)

        return x