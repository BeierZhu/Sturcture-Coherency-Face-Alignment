'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.log_helper import cprint

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_feats, is_color=True, receptive_keep=False):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_feats[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_feats[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_feats[2], num_blocks[2], stride=2)
        stride = 1 if receptive_keep else 2
        self.layer4 = self._make_layer(block, num_feats[3], num_blocks[3], stride=stride)

        self.num_out_feats = [num_feat*block.expansion for num_feat in num_feats]
        self.downsample_ratio = 16*stride

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict


def PreActResNet18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet18 ......")
    return PreActResNet(block=PreActBlock, num_blocks=[2,2,2,2], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)

def LightPreActResNet18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating LightPreActResNet18 ......")
    return PreActResNet(block=PreActBlock, num_blocks=[2,2,2,2], 
                        num_feats=[32, 64, 128, 256], is_color=is_color, receptive_keep=receptive_keep)

def LightPreActResNetBottleNeck18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating LightPreActResNetBottleNeck18 ......")
    return PreActResNet(block=PreActBottleneck, num_blocks=[2,2,2,2], 
                        num_feats=[32, 64, 128, 256], is_color=is_color, receptive_keep=receptive_keep)

def LighterPreActResNet18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating LighterPreActResNet18 ......")
    return PreActResNet(block=PreActBlock, num_blocks=[2,2,2,2], 
                        num_feats=[16, 32, 64, 128], is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet34(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet34 ......")
    return PreActResNet(block=PreActBlock, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet50(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet50 ......")
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet101(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet101 ......")
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,4,23,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet152(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResNet152 ......")
    return PreActResNet(block=PreActBottleneck, num_blocks=[3,8,36,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)
