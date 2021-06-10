import torch
import torch.nn as nn
import torch.nn.functional as F

from models.downsample import Downsample

class AntiAliasMaxPool2d(nn.MaxPool2d):
    pass

class AntiAliasConvBN2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AntiAliasConvBN2dReLU, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if stride != 1:
            self.downsample = Downsample(filt_size=3, stride=stride, channels=out_channels)

    def forward(self, x):
        if hasattr(self, 'downsample'):
            return F.relu(self.bn(self.downsample(self.conv2d(x))))
        else:
            return F.relu(self.bn(self.conv2d(x))) 

# implement later
class AntiAliasAvgPool2d(nn.AvgPool2d):
    pass

