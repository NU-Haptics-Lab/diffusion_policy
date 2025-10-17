import torch
import torch.nn as nn
from diffusion_policy.model.components.trunk import Trunk

from collections import defaultdict

class Tree(nn.Module):
    """
    a Tree-style nn module, where it has a common trunk and separate leafs (a.k.a. heads) for each dataset.
    """
    def __init__(self,
            rootcaps: nn.ModuleDict = None,
            roots: nn.ModuleDict = None,
            trunk: nn.Module = None,
            branches: nn.ModuleDict = None,
            leafs: nn.ModuleDict = None,
            ):
        super().__init__()
        
        self.rootcaps = rootcaps
        self.roots = roots
        self.trunk = trunk
        self.branches = branches
        self.leafs = leafs
        
        self.use_trunk = False
        if self.trunk is not None:
            self.use_trunk = True

    def forward(self, inputs, use_trunk=False, rootcap=None, root=None, branch=None, leaf=None):
        x = inputs
        
        if rootcap is not None:
            assert(self.rootcaps is not None)
            x = self.rootcaps[rootcap](x)
        
        if root is not None:
            assert(self.roots is not None)
            x = self.roots[root](x)
        
        if self.use_trunk or use_trunk:
            assert(self.trunk is not None)
            x = self.trunk(x)
        
        if branch is not None:
            assert(self.branches is not None)
            x = self.branches[branch](x)

        if leaf is not None:
            assert(self.leafs is not None)
            x = self.leafs[leaf](x)
            
            # if x._version == 1:
            #     print("hi")
            #     pass

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