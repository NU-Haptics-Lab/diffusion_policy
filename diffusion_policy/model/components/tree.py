import torch
import torch.nn as nn
from diffusion_policy.model.components.trunk import Trunk

from collections import defaultdict

class Tree(nn.Module):
    """
    a Tree-style nn module, where it has a common trunk and separate leafs (a.k.a. heads) for each dataset.
    """
    def __init__(self,
            rootcaps: dict,
            roots: dict,
            trunk: nn.Module,
            branches: dict,
            leafs: dict
            ):
        self.rootcaps = rootcaps
        self.roots = roots
        self.trunk = trunk
        self.branches = branches
        self.leafs = leafs

    def forward(self, inputs, rootcap=None, root=None, branch=None, leaf=None):
        x = inputs
        
        if len(self.rootcaps) != 0:
            x = self.rootcaps[rootcap](x)
        
        if len(self.roots) != 0:
            x = self.roots[root](x)
        
        # required
        x = self.trunk(x)
        
        if len(self.branches) != 0:
            x = self.branches[branch](x)

        if len(self.leafs) != 0:
            x = self.leafs[leaf](x)

        return x
    
    def forward_options(self, inputs, options: dict = None):
        """
        extract keywords and values from options, if they exist
        """
        if options == None:
            return self.forward(inputs)
        
        # default dict with None as the default value
        options2 = defaultdict(lambda: None)
        options2.update(options)

        return self.forward(inputs, 
            rootcap=options2["rootcap"], 
            root=options2["root"], branch=options2["branch"], leaf=options2["leaf"], 
            )