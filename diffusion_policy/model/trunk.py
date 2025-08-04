import torch
import torch as nn


class Trunk(nn.Module):
    """
    The trunk of a neural net model. This is the common part of a neural net which all data is passed through. Useful with co-training.
    """