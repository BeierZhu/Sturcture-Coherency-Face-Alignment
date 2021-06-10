import torch
import torch.nn as nn
from models.backbones.preact_resnet import PreActBlock
from models.backbones.preact_resnet import PreActBottleneck
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


class PreActBottleneckIBN_a(PreActBottleneck):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckIBN_a, self).__init__(in_planes, planes, stride=stride)
        self.bn1 = IBN_a(in_planes)


class PreActBlockIBN_b(PreActBlock):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockIBN_a, self).__init__(in_planes, planes, stride=stride)
        self.in2 = nn.InstanceNorm2d(planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.in2(self.bn2(out))))
        out += shortcut
        return out
    

class PreActResNetIBN(PreActResNet):
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
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


def PreActResNet18IBN(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet18IBN ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a]
    return PreActResNetIBN(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def LightPreActResNet18IBN_a(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    logger.debug("build LightPreActResNet18IBN_a ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[32, 64, 128, 256], 
                        is_color=is_color, receptive_keep=receptive_keep)


def PreActResNet18IBN_a(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    logger.debug("build PreActResNet18IBN_a ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet18IBN_b(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet18IBN_b ......")
    blocks = [PreActBlockIBN_b, PreActBlockIBN_b, PreActBlockIBN_b, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet34IBN(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet34IBN_a ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet34IBN_a(is_color, pretrained_path=None, receptive_keep=False):
    logger.debug("build PreActResNet34IBN_a ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def LightPreActResNet34IBN_a(is_color, pretrained_path=None, receptive_keep=False):
    logger.debug("build LightPreActResNet34IBN_a ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[32, 64, 128, 256], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet34IBN_b(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet34IBN_b ......")
    blocks = [PreActBlockIBN_b, PreActBlockIBN_b, PreActBlockIBN_b, PreActBlock]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet50IBN_a(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet50IBN_a ......")
    blocks = [PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneck]
    return PreActResNetIBN(blocks=blocks, num_blocks=[3,4,6,3], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)


if __name__ == '__main__':
    from thop import profile
    model = PreActResNet34IBN_a(is_color=True)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, input=(input))
    print(flops, params)
