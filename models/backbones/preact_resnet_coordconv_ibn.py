import torch
import torch.nn as nn
from models.backbones.preact_resnet import PreActBlock
from models.backbones.preact_resnet import PreActResNet
from utils.log_helper import cprint

import logging
logger = logging.getLogger('FLD')

# IBN-a
class IBN_a(nn.Module):
    def __init__(self, planes):
        super(IBN_a, self).__init__()
        half1 = planes // 2
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class PreActBlockIBN_a(PreActBlock):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockIBN_a, self).__init__(in_planes, planes, stride=stride)
        self.bn1 = IBN_a(in_planes)
        self.conv1 = CoordConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = CoordConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                CoordConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
  

class PreActResNetIBN(PreActResNet):
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = CoordConv(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(blocks[0], num_feats[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], num_feats[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], num_feats[2], num_blocks[2], stride=2)
        stride = 1 if receptive_keep else 2
        self.layer4 = self._make_layer(blocks[3], num_feats[3], num_blocks[3], stride=stride)

        self.num_out_feats = [num_feat*blocks[3].expansion for num_feat in num_feats]
        self.downsample_ratio = 16*stride

        self._init_weights()


def PreActResNet34CoordConv(is_color, pretrained_path=None, receptive_keep=False):
    logger.debug("build PreActResNet34IBN_aCoordConv ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret