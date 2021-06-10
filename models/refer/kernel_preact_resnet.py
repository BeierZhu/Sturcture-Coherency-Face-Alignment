import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kernel_conv import KernelConv2d
from models.backbones.preact_resnet import PreActResNet
from utils.log_helper import cprint


class KernelPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(KernelPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = KernelConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = KernelConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                KernelConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

def KernelPreActResNet18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating KernelPreActResNet18 ......")
    return PreActResNet(block=KernelPreActBlock, num_blocks=[2,2,2,2], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)