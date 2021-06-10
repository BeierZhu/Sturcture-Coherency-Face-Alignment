import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GrapConv(nn.Module):
    """docstring for GrapConv
        Input:  in_nodes x C
        Output: out_nodes x C 
    """
    def __init__(self, in_nodes, out_nodes, in_channels, out_channels, adj, bias=True):
        super(GrapConv, self).__init__()
        self.arg = arg
        