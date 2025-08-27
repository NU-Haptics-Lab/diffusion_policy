import torch
import torch.nn as nn


class Leaf(nn.Module):
    """
    A leaf of a neural net. Should be added onto the trunk of a neural net. Useful when co-training and a different "head" is desired for each dataset (a.k.a. task) 
    """
    pass