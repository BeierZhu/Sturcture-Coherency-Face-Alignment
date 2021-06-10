import torch
import torch.nn as nn

from models.backbones.preact_resnet import PreActBlock
from models.backbones.preact_resnet_ibn import IBN_a
from models.backbones.preact_resnet_ibn import PreActResNetIBN
from models.backbones.downsample import Downsample
from models.backbones.preact_resnet_coordconv_ibn import CoordConv

from utils.log_helper import cprint

class PreActBlockAntiAliased(PreActBlock):
    def __init__(self, in_planes, planes, stride=1, filter_size=3):
        super(PreActBlockAntiAliased, self).__init__(in_planes, planes, stride=stride)
        self.conv1 = CoordConv(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1:
            self.conv2 = nn.Sequential(
                        Downsample(filt_size=filter_size, stride=stride, channels=planes),
                        CoordConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                            Downsample(filt_size=filter_size, stride=stride, channels=in_planes),
                            CoordConv(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False))

class PreActBlockAntiAliasedIBN_a(PreActBlockAntiAliased):
    """docstring for PreActBlockAntiAliased"""
    def __init__(self, in_planes, planes, stride=1, filter_size=3):
        super(PreActBlockAntiAliasedIBN_a, self).__init__(in_planes, planes, stride=stride)
        self.bn1 = IBN_a(in_planes)        

class PreActResNetAntiAliased(PreActResNetIBN):
    """docstring for PreActResNetAntiAliased"""
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False, filter_size=3, pool_only=True):
        super(PreActResNetAntiAliased, self).__init__(blocks=blocks, num_blocks=num_blocks, num_feats=num_feats, is_color=is_color, receptive_keep=receptive_keep)

        if not pool_only:
            self.conv1 = CoordConv(3, num_feats[0], kernel_size=7, stride=1, padding=3, bias=False)

        if pool_only:
            self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), 
                                        Downsample(filt_size=filter_size, stride=2, channels=num_feats[0]))
        else:
            self.maxpool = nn.Sequential(Downsample(filt_size=filter_size, stride=2, channels=num_feats[0]),
                                        nn.MaxPool2d(kernel_size=2, stride=1), 
                                        Downsample(filt_size=filter_size, stride=2, channels=num_feats[0]))


def PreActResNetAntiAliased34(is_color, pretrained_path=None, receptive_keep=False):
    cprint.yellow("build PreActResNetAntiAliased34 ......")
    blocks = [PreActBlockAntiAliasedIBN_a, PreActBlockAntiAliasedIBN_a, PreActBlockAntiAliasedIBN_a, PreActBlockAntiAliased]
    return PreActResNetAntiAliased(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)